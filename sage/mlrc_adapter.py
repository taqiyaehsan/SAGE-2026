"""Real-task adapter: the MLRC Machine-Unlearning world for the skeptic-gate loop.

This is the REAL counterpart to synthetic.py. It exposes the SAME three callables
the task-agnostic gates expect:

    propose(history)         -> Candidate           (OpenAI proposes a new MyMethod.py)
    evaluate(candidate, seed)-> float                (write file, run MLRC eval, parse score)
    on_accept(candidate, dec)-> None                 (snapshot + adopt as new incumbent)

Design choices (see PROGRESS.md / HANDOFF.md):
  * KEEP/DISCARD is self-contained via in-memory + snapshot files, NOT MLRC's git
    state, so the runner never depends on the upstream repo's working tree.
  * evaluate() always RESTORES the current best file after a run, so disk == best
    incumbent between steps. The causal gate can therefore call evaluate() k times
    on the same candidate (k independent noisy re-runs) with no extra bookkeeping.
  * A broken proposal is NOT silently re-prompted: it goes to the real eval and a
    crash maps to a very low score, so greedy faithfully "wastes an eval then
    discards". This is exactly the waste the coherence gate (Gate 1) later prevents,
    so we record static_ok per proposal to quantify it.

Higher score is better (MLRC "Final Score" / npz total_score).
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from gates import Fidelity, FULL
from mlrc_background_knowledge import (
    build_enriched_messages,
    load_cross_run_history,
)

# --- paths -----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_DIR = REPO_ROOT / "MLRC-Bench" / "MLAgentBench" / "benchmarks_base" / "machine_unlearning" / "env"
METHOD_FILE = ENV_DIR / "methods" / "MyMethod.py"
NPZ_FILE = ENV_DIR / "dev_results" / "my_method_results.npz"

CRASH_SCORE = -10.0  # measured score assigned to a crashed / unparseable eval


# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    code: str                       # full proposed MyMethod.py content
    intent: str                     # the proposer's one-line stated intent
    rationale: str = ""             # 2-4 sentence explanation of the approach
    static_ok: bool = True          # passed cheap parse/signature check
    static_reason: str = ""         # why static check failed (if it did)
    truth: Optional[dict] = None    # unused on the real task (synthetic-only field)


# ---------------------------------------------------------------------------
# Cheap static coherence check (Gate-1 building block; logged even under greedy)
# ---------------------------------------------------------------------------

REQUIRED_SIG = "def run(self, net, retain_loader, forget_loader, val_loader)"


def static_check(code: str) -> tuple[bool, str]:
    """Parse + signature sanity. Returns (ok, reason). Does NOT run anything."""
    if "class MyMethod" not in code:
        return False, "missing class MyMethod"
    if "BaseMethod" not in code:
        return False, "missing BaseMethod inheritance"
    if "def run(self" not in code:
        return False, "missing run() method"
    # normalise whitespace for a tolerant signature match
    flat = re.sub(r"\s+", " ", code)
    if re.sub(r"\s+", " ", REQUIRED_SIG) not in flat:
        return False, "run() signature changed"
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e.msg} (line {e.lineno})"
    return True, ""


# ---------------------------------------------------------------------------
# Proposer (OpenAI). Provider-agnostic surface: swap _call_llm to change vendor.
# ---------------------------------------------------------------------------

TASK_BRIEF = textwrap.dedent("""\
    You are improving a MACHINE UNLEARNING method for the MLRC-Bench benchmark.

    Goal: after "forgetting" a forget-set, the unlearned model should behave like a
    model retrained from scratch WITHOUT that data, while keeping accuracy on the
    retain/test data. Dev phase uses CIFAR-10 + a pretrained resnet18.

    The score = forgetting_quality * (retain_acc_ratio) * (test_acc_ratio), higher
    is better. The baseline (~0.054) just fine-tunes on the retain set for 1 epoch
    (catastrophic forgetting). You can usually beat it with smarter unlearning.

    HARD RULES (a violation makes the eval crash or be disqualified):
      - Output a COMPLETE Python file defining `class MyMethod(BaseMethod)` with
        EXACTLY this signature:
          def run(self, net, retain_loader, forget_loader, val_loader):
      - Modify `net` IN PLACE and end run() with `net.eval()`. (A return is ignored.)
      - Imports allowed: torch, torch.nn, torch.optim, numpy, copy, math, and
        `from methods.BaseMethod import BaseMethod`. NO internet, NO new files.
      - Keep this device line and move every tensor/model to DEVICE:
          DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
      - APPROXIMATE unlearning only. Do NOT retrain from scratch on the retain set;
        any retraining must be far cheaper than full retraining (keep it to a couple
        of epochs / passes so one eval stays a few minutes on CPU/MPS).
      - Loader batches may be tuples (inputs, targets) for CIFAR. Handle that.

    Ideas you may use: gradient ascent on the forget set + descent on retain,
    selective layer re-init, parameter noise/pruning, fisher/EWC-style penalties,
    relabel-and-finetune, knowledge-distillation toward a retain-only signal, etc.

    Respond with a JSON object:
    {
      "intent": "<one concise sentence summarizing the change>",
      "rationale": "<2-4 sentences explaining WHY this approach should improve the score: what mechanism drives forgetting, how it preserves retain/test accuracy, and what inspired the design (e.g. which expert method or prior failure)>",
      "code": "<full file>"
    }
    Put the entire file in "code" as a single string. No markdown fences.
""")


def build_messages(current_code: str, best_score: float, history: list[dict]) -> list[dict]:
    hist_lines = []
    for h in history[-8:]:
        tag = "ACCEPTED" if h["accepted"] else "rejected"
        hist_lines.append(f"  - [{tag}] score={h['score']:.4f}  intent: {h['intent']}")
    hist_txt = "\n".join(hist_lines) if hist_lines else "  (none yet)"
    user = textwrap.dedent(f"""\
        Current best Final Score: {best_score:.4f}

        What has been tried so far (most recent last):
        {hist_txt}

        Here is the CURRENT best MyMethod.py (your edit should improve on it):
        ```python
        {current_code}
        ```

        Propose ONE new MyMethod.py that you believe will raise the Final Score.
        Make a meaningful, non-trivial change. Return the JSON object described.
    """)
    return [
        {"role": "system", "content": TASK_BRIEF},
        {"role": "user", "content": user},
    ]


class OpenAIProposer:
    def __init__(self, model: str = "gpt-4.1-mini", temperature: float = 0.7,
                 use_background: bool = True, use_cross_run_history: bool = True,
                 focus_best: bool = False):
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent / ".env")
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.use_background = use_background
        self.use_cross_run_history = use_cross_run_history
        self.focus_best = focus_best
        self._cross_run_history = None

    def _get_cross_run_history(self) -> list[dict]:
        if self._cross_run_history is None:
            self._cross_run_history = load_cross_run_history()
        return self._cross_run_history

    def propose(self, current_code: str, best_score: float, history: list[dict]) -> Optional[Candidate]:
        if self.use_background or self.use_cross_run_history:
            messages = build_enriched_messages(
                current_code=current_code,
                best_score=best_score,
                session_history=history,
                task_brief=TASK_BRIEF,
                use_background=self.use_background,
                use_cross_run_history=self.use_cross_run_history,
                cross_run_history=self._get_cross_run_history() if self.use_cross_run_history else None,
                focus_best=self.focus_best,
            )
        else:
            messages = build_messages(current_code, best_score, history)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            obj = json.loads(raw)
            code = obj.get("code", "")
            intent = (obj.get("intent", "") or "").strip()[:500]
            rationale = (obj.get("rationale", "") or "").strip()[:1000]
        except Exception as e:  # noqa: BLE001 - any API/parse failure -> skip proposal
            print(f"  [proposer] error: {type(e).__name__}: {e}")
            return None
        code = _strip_fences(code)
        ok, reason = static_check(code)
        return Candidate(code=code, intent=intent, rationale=rationale,
                         static_ok=ok, static_reason=reason)


def _strip_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        code = re.sub(r"^```[a-zA-Z]*\n", "", code)
        code = re.sub(r"\n```$", "", code)
    return code.strip() + "\n"


# ---------------------------------------------------------------------------
# The world: file management + real eval
# ---------------------------------------------------------------------------

class MLRCWorld:
    """Owns the MyMethod.py file, runs the real eval, and snapshots accepted edits.

    `snapshot_dir` receives a copy of EVERY proposal (accepted, rejected, crashed)
    plus the running JSONL log -- this is the honest-discard record AND the source
    of accepted edits for the replication audit (HANDOFF step 14).
    """

    def __init__(self, proposer, snapshot_dir: Path, eval_timeout: float = 1200.0,
                 mock_eval: bool = False, mock_llm: bool = False, mock_sigma: float = 0.02,
                 fidelity: Fidelity = FULL, env_dir: Optional[Path] = None):
        self.proposer = proposer
        self.snapshot_dir = snapshot_dir
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.eval_timeout = eval_timeout
        self.mock_eval = mock_eval
        self.mock_llm = mock_llm
        self.mock_sigma = mock_sigma
        # cost lever: the active fidelity. `num_models` (if set) is pushed to the
        # MLRC eval via the MU_NUM_MODELS env var (fewer models = cheaper, noisier);
        # `sigma_mult` inflates the mock eval's noise so the trade is visible offline.
        self.fidelity = fidelity
        self.num_models = fidelity.params.get("num_models")
        self.sigma_mult = float(fidelity.params.get("sigma_mult", 1.0))
        self._mock_rng = np.random.default_rng(0)

        self.env_dir = env_dir or ENV_DIR
        self.method_file = self.env_dir / "methods" / "MyMethod.py"
        self.npz_file = self.env_dir / "dev_results" / "my_method_results.npz"

        self.best_code = self.method_file.read_text()  # baseline file = starting incumbent
        self.best_score = float("-inf")
        self.history: list[dict] = []
        self.step_counter = 0
        self.eval_calls = 0

    # -- proposal stream ----------------------------------------------------
    def propose(self, _rng=None) -> Optional[Candidate]:
        if self.mock_llm:
            return self._mock_propose()
        return self.proposer.propose(self.best_code, self.best_score, self.history)

    def _mock_propose(self) -> Candidate:
        # A trivially valid edit (tweaks the LR) so the harness can be exercised free.
        lr = round(float(self._mock_rng.uniform(0.0005, 0.01)), 5)
        code = self.best_code
        if "lr=" in code:
            code = re.sub(r"lr=[0-9.]+", f"lr={lr}", code, count=1)
        ok, reason = static_check(code)
        return Candidate(code=code, intent=f"[mock] set lr={lr}",
                         static_ok=ok, static_reason=reason)

    # -- noisy real evaluation ---------------------------------------------
    def evaluate(self, candidate: Candidate, seed: int) -> float:
        """Write candidate, run the real MLRC eval, parse score, RESTORE best.
        Each call is one independent noisy eval (~3 min on MPS). Costs 1 budget unit."""
        self.eval_calls += 1
        if self.mock_eval:
            return self._mock_evaluate(candidate)
        self.method_file.write_text(candidate.code)
        try:
            score, detail = self._run_eval()
        finally:
            self.method_file.write_text(self.best_code)  # always restore incumbent on disk
        candidate.truth = detail  # stash last eval detail for logging
        return score

    def _mock_evaluate(self, candidate: Candidate) -> float:
        if not candidate.static_ok:
            return CRASH_SCORE
        # Pretend baseline ~0.054 with a small bump for "good-looking" edits + noise.
        # The cheaper fidelity inflates noise by sigma_mult (the cost-lever trade).
        base = 0.054 + self.mock_rng_bump(candidate)
        return float(base + self._mock_rng.normal(0.0, self.mock_sigma * self.sigma_mult))

    def mock_rng_bump(self, candidate: Candidate) -> float:
        return 0.005 if "ascent" in candidate.code.lower() else 0.0

    def _run_eval(self) -> tuple[float, dict]:
        # Remove any stale result so a crash can't read a previous run's score.
        try:
            self.npz_file.unlink()
        except FileNotFoundError:
            pass
        env = dict(os.environ)
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        if self.num_models is not None:  # cost lever -> MLRC evaluation.py reads this
            env["MU_NUM_MODELS"] = str(self.num_models)
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, "-u", "main.py", "-m", "my_method", "-p", "dev"],
                cwd=str(self.env_dir), env=env,
                stdout=None, stderr=None,
                timeout=self.eval_timeout,
            )
        except subprocess.TimeoutExpired:
            return CRASH_SCORE, {"crash": "timeout", "wall_s": time.time() - t0}
        wall = time.time() - t0
        detail = {"returncode": proc.returncode, "wall_s": wall}
        if self.npz_file.exists():
            try:
                with np.load(self.npz_file) as d:
                    score = float(d["total_score"])
                detail["score_source"] = "npz"
                return score, detail
            except Exception as e:  # noqa: BLE001
                detail["npz_error"] = str(e)
        detail["crash"] = "no_score"
        return CRASH_SCORE, detail

    @staticmethod
    def _parse_stdout(out: str) -> dict:
        d = {}
        for key, pat in [("forget_quality", r"Forgetting Quality:\s*([-\d.]+)"),
                         ("retain_ratio", r"Retain Accuracy \(RAU/RAR\):\s*([-\d.]+)"),
                         ("test_ratio", r"Test Accuracy \(TAU/TAR\):\s*([-\d.]+)")]:
            m = re.search(pat, out)
            if m:
                d[key] = float(m.group(1))
        return d

    # -- commit on accept ---------------------------------------------------
    def on_accept(self, candidate: Candidate, decision) -> None:
        self.best_code = candidate.code
        self.best_score = float(np.mean(decision.candidate_scores)) if decision.candidate_scores else self.best_score
        self.method_file.write_text(self.best_code)  # disk now holds the new incumbent

    # -- snapshot every proposal (for audit + honest discards) --------------
    def snapshot(self, candidate: Candidate, status: str) -> str:
        fn = self.snapshot_dir / f"prop_{self.step_counter:03d}_{status}.py"
        fn.write_text(candidate.code)
        return str(fn.relative_to(REPO_ROOT))

    # -- coherence predicate for Gate 1 (used by CoherenceWrapper later) ----
    def is_broken(self, candidate: Candidate) -> bool:
        return not candidate.static_ok
