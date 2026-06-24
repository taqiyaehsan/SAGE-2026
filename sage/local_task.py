"""Local MLRC-style task adapter: the agent EDITS primary code; the harness runs it.

This is the local-dataset counterpart of mlrc_adapter.py (machine unlearning). The
shape is identical, so the SAME task-agnostic gates.py drives it:

    propose(history)          -> Candidate    (LLM rewrites the method file)
    evaluate(candidate, seed) -> float         (run_method.py subprocess, parse score)
    on_accept(candidate, dec) -> None          (adopt as new incumbent)
    is_broken(candidate)      -> bool          (Gate-1 coherence predicate)

Each task is a directory under tasks/<name>/ holding:
    background.md        -- the brief the proposer reads (the approach space)
    baseline_method.py   -- the PRIMARY CODE: a working baseline the agent edits

The agent owns the model + training inside MyMethod; the harness (run_method.py)
owns data, the held-out test split, the seed, CPU-only execution, a timeout, and
scoring. A broken/garbage/hanging edit becomes CRASH_SCORE -- the exact waste the
coherence gate later prevents, which is why we record static_ok per proposal.

Higher score is better (held-out classification accuracy).
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from gates import (Budget, Fidelity, FULL, GreedyPolicy, CausalPolicy,
                   CoherenceWrapper, run_loop)

HERE = Path(__file__).resolve().parent
TASKS_DIR = HERE / "tasks"
RUN_METHOD = HERE / "run_method.py"
CRASH_SCORE = -1e6  # universally-worst score for a crashed/invalid/hanging eval
                    # (below any accuracy in [0,1] AND any plausible R^2)


# ---------------------------------------------------------------------------
# Task spec: a dataset + its primary code + its noise regimes
# ---------------------------------------------------------------------------

@dataclass
class TaskSpec:
    name: str
    time_limit: float          # per-eval wall-clock cap (s); exceeded => crash score
    regimes: list              # [(label, Fidelity{train_frac}), ...] noise dial
    metric: str = "accuracy"   # "accuracy" (classification) or "r2" (regression)

    @property
    def dir(self) -> Path:
        return TASKS_DIR / self.name

    @property
    def background(self) -> str:
        return (self.dir / "background.md").read_text()

    @property
    def baseline_code(self) -> str:
        return (self.dir / "baseline_method.py").read_text()


# train_frac is the noise dial: smaller resample => noisier (and slightly cheaper).
TASKS: dict[str, TaskSpec] = {
    "fashionmnist": TaskSpec(
        "fashionmnist", time_limit=60.0,
        regimes=[("low  (full data)", Fidelity("full", 1.0, {"train_frac": 1.0})),
                 ("med  (25% data)", Fidelity("med", 1.0, {"train_frac": 0.25})),
                 ("high (8% data)", Fidelity("high", 1.0, {"train_frac": 0.08}))]),
    "magic": TaskSpec(
        "magic", time_limit=60.0,
        regimes=[("low  (full data)", Fidelity("full", 1.0, {"train_frac": 1.0})),
                 ("med  (15% data)", Fidelity("med", 1.0, {"train_frac": 0.15})),
                 ("high (4% data)", Fidelity("high", 1.0, {"train_frac": 0.04}))]),
    # SPURIOUS-CORRELATION stress test (IRM Colored MNIST). Same noise-dial regimes
    # as fashionmnist; the interesting axis here is val-vs-test divergence, not the
    # train-subsample noise (see tasks/colored_mnist/background.md + task_data.py).
    "colored_mnist": TaskSpec(
        "colored_mnist", time_limit=120.0,   # CNNs on 2-ch 28x28 are slow single-thread
        regimes=[("low  (full data)", Fidelity("full", 1.0, {"train_frac": 1.0})),
                 ("med  (25% data)", Fidelity("med", 1.0, {"train_frac": 0.25})),
                 ("high (8% data)", Fidelity("high", 1.0, {"train_frac": 0.08}))]),
    # REGRESSION template (sklearn diabetes). metric="r2"; a worked example showing
    # the same pipeline handles regression. Teammates copy this shape for their tasks.
    "example_regression": TaskSpec(
        "example_regression", time_limit=60.0, metric="r2",
        regimes=[("low  (full data)", Fidelity("full", 1.0, {"train_frac": 1.0})),
                 ("med  (40% data)", Fidelity("med", 1.0, {"train_frac": 0.40})),
                 ("high (15% data)", Fidelity("high", 1.0, {"train_frac": 0.15}))]),
}


def get_task(name: str) -> TaskSpec:
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; choose from {list(TASKS)}")
    return TASKS[name]


# ---------------------------------------------------------------------------
# Candidate + cheap static coherence check (Gate-1 building block)
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    code: str
    intent: str
    static_ok: bool = True
    static_reason: str = ""
    truth: Optional[dict] = None   # last eval detail (for logging); synthetic-only field


def static_check(code: str) -> tuple[bool, str]:
    """Parse + interface sanity via the AST (tolerant of type annotations and
    whitespace). Does NOT run anything -- that is the expensive eval the gate
    is deciding whether to spend a budget unit on."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e.msg} (line {e.lineno})"
    cls = next((n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == "MyMethod"), None)
    if cls is None:
        return False, "missing class MyMethod"
    base_names = {b.id for b in cls.bases if isinstance(b, ast.Name)}
    if "BaseMethod" not in base_names:
        return False, "MyMethod must inherit BaseMethod"
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    params = lambda fn: [a.arg for a in fn.args.args]
    if "fit" not in methods:
        return False, "missing fit()"
    if params(methods["fit"])[:4] != ["self", "X", "y", "seed"]:
        return False, f"fit() params {params(methods['fit'])} != [self, X, y, seed]"
    if "predict" not in methods:
        return False, "missing predict()"
    if params(methods["predict"])[:2] != ["self", "X"]:
        return False, f"predict() params {params(methods['predict'])} != [self, X]"
    return True, ""


# ---------------------------------------------------------------------------
# Proposer (OpenAI): the LLM agent rewrites the primary method code
# ---------------------------------------------------------------------------

def _system_brief(task: TaskSpec) -> str:
    return textwrap.dedent(f"""\
        You are an autonomous ML researcher improving a method for a downstream
        task. Below is the task background; edit the provided PRIMARY CODE to raise
        held-out accuracy.

        ===== TASK BACKGROUND =====
        {task.background}
        ===========================

        Respond with a JSON object: {{"intent": "<one concise sentence>", "code":
        "<the COMPLETE edited method file as one string>"}}. No markdown fences.
    """)


def _user_msg(current_code: str, best_score: float, history: list[dict]) -> str:
    lines = []
    for h in history[-8:]:
        tag = "ACCEPTED" if h.get("accepted") else "rejected"
        sc = h.get("score")
        lines.append(f"  - [{tag}] score={sc:.4f}  intent: {h.get('intent','')}"
                     if sc is not None else f"  - [{tag}] intent: {h.get('intent','')}")
    hist_txt = "\n".join(lines) if lines else "  (none yet)"
    bs = f"{best_score:.4f}" if best_score > float("-inf") else "(not yet measured)"
    return textwrap.dedent(f"""\
        Current best held-out accuracy: {bs}

        What has been tried (most recent last):
        {hist_txt}

        CURRENT best method file (your edit should improve on it):
        ```python
        {current_code}
        ```

        Propose ONE new complete method file with a meaningful, non-trivial change.
        Return the JSON object described.""")


def _strip_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        code = re.sub(r"^```[a-zA-Z]*\n", "", code)
        code = re.sub(r"\n```$", "", code)
    return code.strip() + "\n"


class OpenAIProposer:
    def __init__(self, task: TaskSpec, model: str = "gpt-4.1-mini",
                 temperature: float = 0.7):
        from dotenv import load_dotenv
        load_dotenv(HERE / ".env")
        from openai import OpenAI
        self.client = OpenAI()
        self.task = task
        self.model = model
        self.temperature = temperature

    def propose(self, current_code: str, best_score: float,
                history: list[dict]) -> Optional[Candidate]:
        try:
            resp = self.client.chat.completions.create(
                model=self.model, temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": _system_brief(self.task)},
                          {"role": "user", "content": _user_msg(current_code, best_score, history)}])
            obj = json.loads(resp.choices[0].message.content)
            code = _strip_fences(obj.get("code", ""))
            intent = (obj.get("intent", "") or "").strip()[:300]
        except Exception as e:  # noqa: BLE001
            print(f"  [proposer] error: {type(e).__name__}: {e}")
            return None
        ok, reason = static_check(code)
        return Candidate(code=code, intent=intent, static_ok=ok, static_reason=reason)


# ---------------------------------------------------------------------------
# The world: writes candidate code, runs the harness subprocess, parses score
# ---------------------------------------------------------------------------

class LocalTaskWorld:
    def __init__(self, task: TaskSpec, proposer, snapshot_dir: Path,
                 fidelity: Fidelity = FULL, mock_llm: bool = False):
        self.task = task
        self.proposer = proposer
        self.snapshot_dir = snapshot_dir
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.fidelity = fidelity
        self.train_frac = float(fidelity.params.get("train_frac", 1.0))
        self.mock_llm = mock_llm
        self.best_code = task.baseline_code   # primary code = starting incumbent
        self.best_score = float("-inf")
        self.history: list[dict] = []
        self.accepted_records: list[dict] = []
        self.step_counter = 0
        self.eval_calls = 0
        self._mock_rng = np.random.default_rng(0)
        self._eval_file = self.snapshot_dir / "_current_eval_method.py"

    # -- proposal stream ----------------------------------------------------
    def propose(self, _rng=None) -> Optional[Candidate]:
        self.step_counter += 1
        if self.mock_llm:
            return self._mock_propose()
        return self.proposer.propose(self.best_code, self.best_score, self.history)

    def _mock_propose(self) -> Candidate:
        """A trivially valid edit (tweak the first lr) so the harness runs free."""
        lr = round(float(self._mock_rng.uniform(0.02, 0.2)), 4)
        code = re.sub(r"lr=[0-9.]+", f"lr={lr}", self.best_code, count=1)
        ok, reason = static_check(code)
        return Candidate(code=code, intent=f"[mock] set lr={lr}",
                         static_ok=ok, static_reason=reason)

    # -- noisy real evaluation (one budget unit) ---------------------------
    def evaluate(self, candidate: Candidate, seed: int, split: str = "val") -> float:
        self.eval_calls += 1
        if not candidate.static_ok:
            return CRASH_SCORE  # don't even run code that failed the cheap check
        self._eval_file.write_text(candidate.code)
        cmd = [sys.executable, str(RUN_METHOD), "--task", self.task.name,
               "--method", str(self._eval_file), "--seed", str(seed),
               "--frac", str(self.train_frac), "--split", split,
               "--metric", self.task.metric]
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True,
                                  timeout=self.task.time_limit)
        except subprocess.TimeoutExpired:
            candidate.truth = {"crash": "timeout", "wall_s": time.time() - t0}
            return CRASH_SCORE
        out = proc.stdout.strip().splitlines()
        result = {}
        for line in reversed(out):           # last JSON line is the result
            try:
                result = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        if "score" in result:
            candidate.truth = result
            return float(result["score"])
        candidate.truth = {"crash": result.get("error", "no_score"),
                           "stderr_tail": proc.stderr[-400:]}
        return CRASH_SCORE

    # -- commit on accept ---------------------------------------------------
    def on_accept(self, candidate: Candidate, decision) -> None:
        prev = self.best_code
        self.best_code = candidate.code
        mean = float(np.mean(decision.candidate_scores)) if decision.candidate_scores else None
        if mean is not None:
            self.best_score = mean
        self.accepted_records.append({
            "step": self.step_counter, "intent": candidate.intent,
            "code_before": prev, "code_after": candidate.code,
            "apparent_scores": list(decision.candidate_scores), "apparent_mean": mean,
        })
        self.history.append({"intent": candidate.intent, "score": mean, "accepted": True})

    def is_broken(self, candidate: Candidate) -> bool:
        return not candidate.static_ok

    def snapshot(self, candidate: Candidate, status: str) -> str:
        fn = self.snapshot_dir / f"prop_{self.step_counter:03d}_{status}.py"
        fn.write_text(candidate.code)
        return str(fn.relative_to(HERE))


# ---------------------------------------------------------------------------
# Run one arm of the pipeline (same shape as hpo_task.run_arm / synthetic.run_arm)
# ---------------------------------------------------------------------------

def _build_policy(arm: str, world: LocalTaskWorld, eval_cost: float):
    if arm == "greedy":
        return GreedyPolicy(eval_cost=eval_cost)
    if arm == "causal":
        return CausalPolicy(k0=2, k_max=6, z=1.0, eval_cost=eval_cost)
    if arm == "coh+greedy":
        return CoherenceWrapper(GreedyPolicy(eval_cost=eval_cost), world.is_broken)
    if arm == "coh+causal":
        return CoherenceWrapper(CausalPolicy(k0=2, k_max=6, z=1.0, eval_cost=eval_cost),
                                world.is_broken)
    raise ValueError(arm)


def run_arm(task_name: str, arm: str, budget_units: float, outer_seed: int,
            fidelity: Fidelity = FULL, *, mock_llm: bool = False,
            model: str = "gpt-4.1-mini", snapshot_dir: Optional[Path] = None) -> dict:
    """One propose->gate->keep arm on a local task. Proposer is the real OpenAI
    agent unless mock_llm=True (free, for harness testing/CI)."""
    task = get_task(task_name)
    import numpy as _np
    prng = _np.random.default_rng(outer_seed * 1000 + 7)
    snap = snapshot_dir or (HERE / "results" / f"local_{task_name}" / f"{arm}_seed{outer_seed}")
    proposer = None if mock_llm else OpenAIProposer(task, model=model)
    world = LocalTaskWorld(task, proposer, snap, fidelity=fidelity, mock_llm=mock_llm)
    eval_cost = fidelity.cost
    policy = _build_policy(arm, world, eval_cost)
    budget = Budget(budget_units)

    # Baseline incumbent band: 2 real evals of the primary code.
    base = Candidate(code=task.baseline_code, intent="baseline", static_ok=True)
    base_scores = []
    for s in range(2):
        if budget.can_afford(eval_cost):
            base_scores.append(world.evaluate(base, 5_000 + s))
            budget.charge(eval_cost)
    if not base_scores:
        base_scores = [0.0]
    world.best_score = float(np.mean(base_scores))

    logs = run_loop(world.propose, world.evaluate, policy, budget,
                    world.on_accept, base_scores, rng=prng)
    return {
        "task": task_name, "arm": arm, "outer_seed": outer_seed,
        "budget_units": budget_units, "fidelity": fidelity.name,
        "n_steps": len(logs), "n_accepted": len(world.accepted_records),
        "n_culled": sum(1 for L in logs if L.culled), "eval_calls": world.eval_calls,
        "budget_spent": budget.spent, "base_scores": base_scores,
        "final_code": world.best_code, "final_score": world.best_score,
        "accepted_records": world.accepted_records, "history": world.history,
    }
