"""Step 6 runner: the vanilla greedy autoresearch loop on MLRC Machine Unlearning.

    propose (OpenAI) -> evaluate (real MLRC eval) -> accept rule -> keep/discard -> log

The ACCEPT DECISION is delegated to the SAME policies in gates.py that drive the
synthetic control, so the only thing that differs across arms (greedy vs gated) is
the policy object. This driver adds the rich, real-task logging (intent, score,
wall-time, snapshot path, crash cause) that the generic gates.run_loop doesn't carry.

Usage:
    python run_mlrc.py --arm greedy --budget 6
    python run_mlrc.py --mock-llm --mock-eval --budget 4      # free harness smoke test
    python run_mlrc.py --arm greedy --budget 1                # one real iteration

Each real eval is ~3 min on MPS. Budget unit = one eval (one seed). The baseline
measurement and every gate re-run are charged to the same budget (equal-budget rule).
"""

from __future__ import annotations

import argparse
import json
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gates import (Budget, Incumbent, GreedyPolicy, CausalPolicy, CoherenceWrapper,
                   Fidelity, FULL)
from mlrc_adapter import MLRCWorld, OpenAIProposer, Candidate, REPO_ROOT, METHOD_FILE

RESULTS_ROOT = REPO_ROOT / "skeptic_gate" / "results"
BASELINE_FILE = REPO_ROOT / "skeptic_gate" / "baseline_MyMethod.py"

# Cost-lever presets for this task. "cheap" runs the unlearning eval over 3 inner
# models instead of 10 (faster, noisier) and is charged 0.3 budget units/eval, so
# the same budget funds ~3x more evals. Validate the 0.3 weight against measured
# wall-clock on the GPU before trusting it for equal-budget claims.
FIDELITIES = {
    "full": FULL,
    "cheap": Fidelity(name="cheap", cost=0.3, params={"num_models": 3, "sigma_mult": 1.6}),
}


def _load_saved_baseline(results_root: Path) -> float | None:
    """Scan previous runs for a baseline score. Returns the most recent one, or None."""
    if not results_root.exists():
        return None
    best = None
    best_time = ""
    for run_dir in results_root.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith("mlrc_"):
            continue
        summary = run_dir / "summary.json"
        if not summary.exists():
            continue
        try:
            data = json.loads(summary.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        score = data.get("baseline_score")
        started = data.get("started", "")
        if score is not None and score > 0 and started >= best_time:
            best = score
            best_time = started
    return best


def build_policy(arm: str, world: MLRCWorld, eval_cost: float = 1.0):
    if arm == "greedy":
        return GreedyPolicy(eval_cost=eval_cost)
    if arm == "causal":
        return CausalPolicy(k0=2, k_max=6, z=1.0, eval_cost=eval_cost)
    if arm == "coh+greedy":
        return CoherenceWrapper(GreedyPolicy(eval_cost=eval_cost), world.is_broken)
    if arm == "coh+causal":
        return CoherenceWrapper(CausalPolicy(k0=2, k_max=6, z=1.0, eval_cost=eval_cost),
                                world.is_broken)
    raise ValueError(f"unknown arm: {arm}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="greedy",
                    choices=["greedy", "causal", "coh+greedy", "coh+causal"])
    ap.add_argument("--budget", type=float, default=6.0,
                    help="total eval-units (one FULL eval = 1 unit, ~3 min)")
    ap.add_argument("--fidelity", default="full", choices=list(FIDELITIES),
                    help="cost lever: 'cheap' = fewer inner models, faster+noisier, 0.3 units/eval")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0, help="outer seed (logging + RNG label)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--eval-timeout", type=float, default=1800.0)
    ap.add_argument("--mock-llm", action="store_true", help="skip OpenAI; trivial valid edits")
    ap.add_argument("--mock-eval", action="store_true", help="skip real eval; synthetic score")
    ap.add_argument("--no-background", action="store_true",
                    help="disable expert background.txt in GPT prompt")
    ap.add_argument("--no-history", action="store_true",
                    help="disable cross-run history in GPT prompt")
    ap.add_argument("--focus-best", action="store_true",
                    help="highlight top methods from previous runs and instruct "
                         "the LLM to extend the current best (default: show flat "
                         "history and allow any approach)")
    ap.add_argument("--no-reuse-baseline", action="store_true",
                    help="force a fresh baseline eval instead of reusing a saved one")
    ap.add_argument("--reset-from", default=str(BASELINE_FILE),
                    help="reset MyMethod.py from this canonical baseline before the run "
                         "(ensures every arm starts identical); '' to skip")
    ap.add_argument("--env-dir", default=None,
                    help="path to a separate env/ copy for parallel runs (avoids "
                         "MyMethod.py write conflicts)")
    args = ap.parse_args()

    # Resolve env_dir for parallel isolation (each run gets its own MyMethod.py).
    env_dir = Path(args.env_dir) if args.env_dir else None
    method_file = (env_dir / "methods" / "MyMethod.py") if env_dir else METHOD_FILE

    # Reset the edit target to the canonical baseline so every arm starts from the
    # SAME incumbent (a prior accepting run leaves its method on disk otherwise).
    if args.reset_from and Path(args.reset_from).exists():
        method_file.write_text(Path(args.reset_from).read_text())
        print(f"reset {method_file.name} from {Path(args.reset_from).name}")

    run_id = args.run_id or f"mlrc_{args.arm}_s{args.seed}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    run_dir = RESULTS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "results.jsonl"
    log_f = log_path.open("w")

    def emit(rec: dict):
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    fidelity = FIDELITIES[args.fidelity]
    proposer = None if args.mock_llm else OpenAIProposer(
        args.model, args.temperature,
        use_background=not args.no_background,
        use_cross_run_history=not args.no_history,
        focus_best=args.focus_best,
    )
    world = MLRCWorld(proposer, snapshot_dir=run_dir / "proposals",
                      eval_timeout=args.eval_timeout,
                      mock_eval=args.mock_eval, mock_llm=args.mock_llm,
                      fidelity=fidelity, env_dir=env_dir)
    policy = build_policy(args.arm, world, eval_cost=fidelity.cost)
    budget = Budget(args.budget)

    meta = {
        "run_id": run_id, "arm": args.arm, "policy": policy.name, "budget": args.budget,
        "model": args.model, "temperature": args.temperature, "seed": args.seed,
        "fidelity": fidelity.name, "fidelity_cost": fidelity.cost,
        "fidelity_params": fidelity.params,
        "mock_llm": args.mock_llm, "mock_eval": args.mock_eval,
        "started": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"== run {run_id} | arm={args.arm} | budget={args.budget} ==")

    # --- baseline incumbent: reuse saved score or measure fresh ----
    base_score = None
    if not args.no_reuse_baseline:
        base_score = _load_saved_baseline(RESULTS_ROOT)
    if base_score is not None:
        print(f"reusing saved baseline score = {base_score:.4f}")
        world.best_score = base_score
        incumbent = Incumbent(scores=[base_score])
        emit({"step": -1, "kind": "baseline", "score": base_score,
              "wall_s": 0.0, "budget_spent_after": 0.0,
              "detail": {"score_source": "cached"}})
    else:
        base_cand = Candidate(code=world.best_code, intent="baseline", static_ok=True)
        print("measuring baseline (no saved score found) ...")
        t0 = time.time()
        base_score = world.evaluate(base_cand, seed=5000 + args.seed)
        budget.charge(fidelity.cost)
        world.best_score = base_score
        incumbent = Incumbent(scores=[base_score])
        emit({"step": -1, "kind": "baseline", "score": base_score,
              "wall_s": time.time() - t0, "budget_spent_after": budget.spent,
              "detail": base_cand.truth})
        print(f"baseline score = {base_score:.4f}  ({time.time()-t0:.0f}s)")

    # --- the loop -----------------------------------------------------------
    seed_counter = 10_000 + args.seed * 1000
    accepted_count = culled_count = crash_count = 0
    while budget.remaining() > 1e-9:
        world.step_counter += 1
        step = world.step_counter
        t_step = time.time()
        cand = world.propose()
        if cand is None:
            emit({"step": step, "kind": "propose_failed"})
            print(f"[{step}] proposal failed; stopping.")
            break
        seed_counter += 100

        # --- BEFORE eval: show what GPT proposed ---
        print(f"\n{'='*70}")
        print(f"  STEP {step}/{int(budget.total) - 1}  |  budget spent: {budget.spent:.1f}/{budget.total:.0f}")
        print(f"{'='*70}")
        print(f"  LLM PROPOSAL: {cand.intent}")
        if cand.rationale:
            print(f"  RATIONALE:")
            for line in textwrap.wrap(cand.rationale, width=64):
                print(f"    {line}")
        static_status = "PASS" if cand.static_ok else f"FAIL ({cand.static_reason})"
        print(f"  Static check: {static_status}")
        print(f"  Evaluating...", flush=True)

        dec = policy.decide(cand, world.evaluate, incumbent, budget, seed_counter)

        status = "accept" if dec.accepted else ("cull" if dec.culled else "reject")
        is_crash = bool(cand.truth and cand.truth.get("crash"))
        if is_crash:
            crash_count += 1
        if dec.accepted:
            world.on_accept(cand, dec)
            if dec.candidate_scores:
                incumbent.scores = list(dec.candidate_scores)
            accepted_count += 1
        if dec.culled:
            culled_count += 1

        snap = world.snapshot(cand, status)
        mean_score = float(np.mean(dec.candidate_scores)) if dec.candidate_scores else None

        # --- AFTER eval: show results ---
        detail = cand.truth or {}
        print(f"  {'-'*66}")
        if is_crash:
            print(f"  RESULT: CRASHED — {detail.get('crash', 'unknown error')}")
        else:
            ms_str = f"{mean_score:.4f}" if mean_score is not None else "n/a"
            print(f"  RESULT: score = {ms_str}  (current best = {world.best_score:.4f})")
            if detail.get("forget_quality") is not None:
                print(f"    Forgetting Quality:  {detail['forget_quality']:.4f}")
            if detail.get("retain_ratio") is not None:
                print(f"    Retain Acc Ratio:    {detail['retain_ratio']:.4f}")
            if detail.get("test_ratio") is not None:
                print(f"    Test Acc Ratio:      {detail['test_ratio']:.4f}")
        verdict = "*** ACCEPTED ***" if dec.accepted else ("CULLED" if dec.culled else "REJECTED")
        print(f"  VERDICT: {verdict}  ({dec.reason})")
        if dec.accepted:
            print(f"  >> New best score: {world.best_score:.4f}")
        print(f"  Wall time: {time.time() - t_step:.0f}s")
        print(f"{'='*70}", flush=True)

        rec = {
            "step": step, "kind": "proposal", "status": status, "accepted": dec.accepted,
            "reason": dec.reason, "intent": cand.intent, "rationale": cand.rationale,
            "static_ok": cand.static_ok, "static_reason": cand.static_reason,
            "scores": dec.candidate_scores, "mean_score": mean_score,
            "delta_hat": dec.delta_hat, "se_hat": dec.se_hat,
            "units_spent": dec.units_spent, "budget_spent_after": budget.spent,
            "incumbent_best": world.best_score, "is_crash": is_crash,
            "eval_detail": cand.truth, "snapshot": snap, "wall_s": time.time() - t_step,
        }
        emit(rec)
        world.history.append({"intent": cand.intent,
                              "score": mean_score if mean_score is not None else -99.0,
                              "accepted": dec.accepted})

        if dec.units_spent == 0.0 and not dec.culled:
            print("policy could not afford an action; stopping.")
            break

    summary = {
        **meta, "finished": datetime.now(timezone.utc).isoformat(),
        "baseline_score": base_score, "best_score": world.best_score,
        "improvement": world.best_score - base_score,
        "n_proposals": world.step_counter, "n_accepted": accepted_count,
        "n_culled": culled_count, "n_crashed": crash_count,
        "eval_calls": world.eval_calls, "budget_spent": budget.spent,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log_f.close()
    print("\n== summary ==")
    print(f"baseline {base_score:.4f} -> best {world.best_score:.4f} "
          f"(+{summary['improvement']:.4f})")
    print(f"proposals={world.step_counter} accepted={accepted_count} "
          f"culled={culled_count} crashed={crash_count} eval_calls={world.eval_calls}")
    print(f"logs: {log_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
