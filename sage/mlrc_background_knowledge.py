"""Background knowledge and cross-run memory for the MLRC autoresearcher.

Provides two enrichments to the GPT prompt:

1. **Expert background** — loads competition participants' methods from
   MLRC-Bench's ``background.txt`` so GPT has proven unlearning strategies
   to draw from instead of re-inventing from scratch.

2. **Cross-run history** — scans all previous ``results/mlrc_*/results.jsonl``
   files and builds a persistent memory of every proposal ever tried: intent,
   score, accepted/rejected, and detailed metrics.  GPT sees what has already
   been tried across ALL runs and avoids repeating failed approaches.

Usage from mlrc_adapter.py:

    from mlrc_background_knowledge import (
        load_background_text,
        load_cross_run_history,
        build_enriched_messages,
    )
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKGROUND_PATH = (
    REPO_ROOT
    / "MLRC-Bench"
    / "MLAgentBench"
    / "benchmarks_base"
    / "machine_unlearning"
    / "scripts"
    / "background.txt"
)
RESULTS_ROOT = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# 1. Expert background from competition participants
# ---------------------------------------------------------------------------

def load_background_text(path: Path = BACKGROUND_PATH) -> Optional[str]:
    """Read background.txt and return its contents, or None if missing."""
    if path.exists():
        return path.read_text().strip()
    return None


def format_background_section(text: Optional[str] = None) -> str:
    """Format background.txt content as a prompt section."""
    if text is None:
        text = load_background_text()
    if not text:
        return ""
    return textwrap.dedent(f"""\

    EXPERT METHODS FROM COMPETITION PARTICIPANTS (use these as inspiration):
    {text}
    """)


# ---------------------------------------------------------------------------
# 2. Cross-run persistent history
# ---------------------------------------------------------------------------

def load_cross_run_history(
    results_root: Path = RESULTS_ROOT,
    exclude_run_id: Optional[str] = None,
) -> list[dict]:
    """Scan all previous mlrc_* run directories and extract every proposal.

    Returns a list of dicts sorted by timestamp, each containing:
        run_id, step, intent, score, accepted, status,
        forget_quality, retain_ratio, test_ratio, is_crash
    """
    history = []
    if not results_root.exists():
        return history

    for run_dir in sorted(results_root.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("mlrc_"):
            continue
        if exclude_run_id and run_dir.name == exclude_run_id:
            continue

        jsonl = run_dir / "results.jsonl"
        if not jsonl.exists():
            continue

        meta = {}
        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        for line in jsonl.read_text().strip().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") != "proposal":
                continue

            detail = rec.get("eval_detail") or {}
            history.append({
                "run_id": run_dir.name,
                "arm": meta.get("arm", "unknown"),
                "step": rec.get("step"),
                "intent": rec.get("intent", ""),
                "rationale": rec.get("rationale", ""),
                "score": rec.get("mean_score"),
                "accepted": rec.get("accepted", False),
                "status": rec.get("status", ""),
                "forget_quality": detail.get("forget_quality"),
                "retain_ratio": detail.get("retain_ratio"),
                "test_ratio": detail.get("test_ratio"),
                "is_crash": rec.get("is_crash", False),
            })

    return history


def format_history_section(history: Optional[list[dict]] = None,
                           focus_best: bool = False) -> str:
    """Format cross-run history into a prompt section for GPT.

    When focus_best=True, separates top-performing accepted methods (with full
    intent + rationale) from the rest and tells the LLM to extend them.
    When focus_best=False (default), shows a flat list of all attempts."""
    if history is None:
        history = load_cross_run_history()
    if not history:
        return ""

    if not focus_best:
        lines = []
        for h in history:
            tag = "ACCEPTED" if h["accepted"] else "rejected"
            if h["is_crash"]:
                tag = "CRASHED"
            score_str = f"{h['score']:.4f}" if h["score"] is not None else "crash"
            detail_parts = []
            if h.get("forget_quality") is not None:
                detail_parts.append(f"fq={h['forget_quality']:.4f}")
            if h.get("retain_ratio") is not None:
                detail_parts.append(f"retain={h['retain_ratio']:.4f}")
            if h.get("test_ratio") is not None:
                detail_parts.append(f"test={h['test_ratio']:.4f}")
            detail_str = f" ({', '.join(detail_parts)})" if detail_parts else ""
            lines.append(
                f"  - [{tag}] score={score_str}{detail_str}  intent: {h['intent']}"
            )
        return textwrap.dedent(f"""\

        HISTORY FROM ALL PREVIOUS RUNS (do NOT repeat failed approaches):
        {chr(10).join(lines)}
        """)

    scored = [h for h in history if h["score"] is not None and not h["is_crash"]]
    best_accepted = sorted(
        [h for h in scored if h["accepted"]],
        key=lambda h: h["score"],
        reverse=True,
    )

    top_lines = []
    for rank, h in enumerate(best_accepted[:5], 1):
        detail_parts = []
        if h.get("forget_quality") is not None:
            detail_parts.append(f"fq={h['forget_quality']:.4f}")
        if h.get("retain_ratio") is not None:
            detail_parts.append(f"retain={h['retain_ratio']:.4f}")
        if h.get("test_ratio") is not None:
            detail_parts.append(f"test={h['test_ratio']:.4f}")
        detail_str = f" ({', '.join(detail_parts)})" if detail_parts else ""
        entry = f"  {rank}. score={h['score']:.4f}{detail_str}\n     intent: {h['intent']}"
        if h.get("rationale"):
            entry += f"\n     rationale: {h['rationale']}"
        top_lines.append(entry)

    rest_lines = []
    top_set = set(id(h) for h in best_accepted[:5])
    for h in history:
        if id(h) in top_set:
            continue
        tag = "ACCEPTED" if h["accepted"] else "rejected"
        if h["is_crash"]:
            tag = "CRASHED"
        score_str = f"{h['score']:.4f}" if h["score"] is not None else "crash"
        rest_lines.append(f"  - [{tag}] score={score_str}  intent: {h['intent']}")

    sections = []
    if top_lines:
        sections.append(
            "TOP METHODS FROM PREVIOUS RUNS (extend these — the intent and rationale\n"
            "describe the approach in enough detail to reconstruct the code):\n"
            + "\n".join(top_lines)
        )
    if rest_lines:
        sections.append(
            "OTHER PREVIOUS ATTEMPTS (do NOT repeat failed approaches):\n"
            + "\n".join(rest_lines)
        )

    return "\n\n" + "\n\n".join(sections) + "\n"


# ---------------------------------------------------------------------------
# 3. Enriched message builder
# ---------------------------------------------------------------------------

def build_enriched_messages(
    current_code: str,
    best_score: float,
    session_history: list[dict],
    task_brief: str,
    use_background: bool = True,
    use_cross_run_history: bool = True,
    cross_run_history: Optional[list[dict]] = None,
    focus_best: bool = False,
) -> list[dict]:
    """Build GPT messages with optional background knowledge and cross-run memory.

    Parameters
    ----------
    current_code : str
        The current best MyMethod.py source.
    best_score : float
        The current best score.
    session_history : list[dict]
        History from the CURRENT run (same as the original build_messages).
    task_brief : str
        The base system prompt (TASK_BRIEF from mlrc_adapter).
    use_background : bool
        Include expert methods from background.txt.
    use_cross_run_history : bool
        Include proposals from all previous runs.
    cross_run_history : list[dict] or None
        Pre-loaded cross-run history (avoids re-scanning on every call).
    focus_best : bool
        When True, highlight top methods and instruct the LLM to extend the
        current best. When False (default), show flat history and allow any approach.
    """

    # --- system prompt: task brief + optional background ---
    system_parts = [task_brief]
    if use_background:
        bg = format_background_section()
        if bg:
            system_parts.append(bg)

    system_content = "\n".join(system_parts)

    # --- cross-run history section ---
    cross_run_section = ""
    if use_cross_run_history:
        cross_run_section = format_history_section(cross_run_history,
                                                   focus_best=focus_best)

    # --- current session history ---
    session_lines = []
    for h in session_history[-8:]:
        tag = "ACCEPTED" if h["accepted"] else "rejected"
        session_lines.append(
            f"  - [{tag}] score={h['score']:.4f}  intent: {h['intent']}"
        )
    session_txt = "\n".join(session_lines) if session_lines else "  (none yet)"

    if focus_best:
        code_label = ("CURRENT BEST MyMethod.py -- this is the incumbent that "
                      "achieved the best score so far:")
        strategy_block = (
            "STRATEGY: Focus on EXTENDING the current best method. This code "
            "already outperformed every previous alternative -- do not discard it. "
            "Keep the core approach (architecture, training loop, key "
            "hyperparameters) and make ONE targeted improvement. For example: add "
            "or tune regularization, adjust learning rate / schedule, add a layer "
            "or change layer sizes, improve data preprocessing, switch optimizer, "
            "or add gradient clipping. Radical rewrites that abandon the current "
            "approach risk losing the gains it already earned.\n"
        )
        inspire_line = (
            "Draw inspiration from the expert methods if provided, but integrate "
            "ideas INTO the current best method rather than replacing it wholesale."
        )
    else:
        code_label = "CURRENT MyMethod.py:"
        strategy_block = ""
        inspire_line = (
            "Draw inspiration from the expert methods if provided. "
            "Make a meaningful, non-trivial change."
        )

    user = textwrap.dedent(f"""\
        Current best Final Score: {best_score:.4f}
        {cross_run_section}
        What has been tried THIS run (most recent last):
        {session_txt}

        {code_label}
        ```python
        {current_code}
        ```

        {strategy_block}CRITICAL -- KEEP IT FAST: Your method MUST finish within 30 minutes on a
        single GPU. Methods that exceed this time limit are KILLED and score -10
        (worst possible). Most previous timeouts came from too many epochs or
        expensive per-sample operations. Rules of thumb:
          - Use at most 2-3 epochs / passes over the data.
          - Avoid nested loops over individual samples (use batched operations).
          - Avoid computing per-sample gradients or Hessians.
          - If adding a new loss term, keep it lightweight (KL div, MSE on logits).
          - Test mentally: if retain_loader has ~4000 batches, will your loop finish
            in time?

        Do NOT repeat any approach from the history above that scored poorly.
        {inspire_line}
        Return the JSON object described.
    """)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user},
    ]
