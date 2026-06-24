"""Replication audit on the REAL task (HANDOFF step 14 — the centerpiece).

Take the changes the GREEDY agent actually ACCEPTED, re-run each one many times
as fresh independent evals, and report how many "improvements" survive. A greedy
win is luck if, on re-testing, its mean score falls back inside the baseline's
noise band (i.e. it is not actually above baseline).

This uses the SAME real eval as the loop (MLRCWorld.evaluate, fresh npz each call)
and NO API calls. It must NOT run while a greedy/gated loop is running — both write
methods/MyMethod.py and the eval npz.

Definitions (pre-registered):
  baseline_mean, baseline_std : from a baseline-noise run (default: results/baseline_noise_n8)
  band = z * baseline_std     : the noise band around baseline (z=2 by default)
  For each accepted change C:
    accepted_at = the single (lucky) score greedy saw when it accepted C
    retest_mean, retest_std = stats over `reps` fresh evals of C
    SURVIVES if retest_mean > baseline_mean + band   (a real improvement over baseline)
    VANISHES otherwise                                (the gain was noise / over-claimed)

Usage:
    python replication_audit_real.py --run real_greedy_b8 --reps 15
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path

from mlrc_adapter import MLRCWorld, Candidate, REPO_ROOT

RESULTS_ROOT = REPO_ROOT / "skeptic_gate" / "results"


def load_accepted(run_dir: Path) -> list[dict]:
    """Return accepted proposals: [{step, intent, accepted_at, snapshot_path}]."""
    accepted = []
    log = run_dir / "results.jsonl"
    for line in log.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("kind") == "proposal" and rec.get("accepted"):
            snap = REPO_ROOT / rec["snapshot"]
            accepted.append({
                "step": rec["step"], "intent": rec.get("intent", ""),
                "accepted_at": rec.get("mean_score"),
                "snapshot": snap,
            })
    return accepted


def load_baseline(baseline_run: str) -> tuple[float, float]:
    p = RESULTS_ROOT / baseline_run / "summary.json"
    d = json.loads(p.read_text())
    return d["mean"], d["stdev"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="greedy run id under results/ to audit")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--baseline-run", default="baseline_noise_n8")
    ap.add_argument("--z", type=float, default=2.0, help="noise-band width in baseline std")
    args = ap.parse_args()

    run_dir = RESULTS_ROOT / args.run
    accepted = load_accepted(run_dir)
    base_mean, base_std = load_baseline(args.baseline_run)
    band = args.z * base_std
    survive_threshold = base_mean + band

    print(f"== replication audit: {args.run} ==")
    print(f"baseline: mean={base_mean:.4f} std={base_std:.4f}  "
          f"survive if retest_mean > {survive_threshold:.4f} (mean+{args.z}σ)")
    print(f"greedy accepted {len(accepted)} change(s); re-running each {args.reps}x\n")

    if not accepted:
        print("Greedy accepted NOTHING -> no wins to audit. Honest Layer-1 finding: "
              "the agent found no above-baseline change at this setting.")
        out = {"run": args.run, "n_accepted": 0, "baseline_mean": base_mean,
               "baseline_std": base_std, "results": []}
        (run_dir / "replication_audit_real.json").write_text(json.dumps(out, indent=2))
        return

    # mock_llm=True so no API; real eval path only.
    world = MLRCWorld(proposer=None, snapshot_dir=run_dir / "audit_tmp", mock_llm=True)

    results = []
    for a in accepted:
        code = a["snapshot"].read_text()
        cand = Candidate(code=code, intent=a["intent"], static_ok=True)
        scores = []
        for r in range(args.reps):
            t0 = time.time()
            s = world.evaluate(cand, seed=10_000 + r)
            scores.append(s)
            print(f"  [step {a['step']}] rep {r+1}/{args.reps}: {s:.4f} ({time.time()-t0:.0f}s)")
        rmean = st.mean(scores)
        rstd = st.pstdev(scores)
        survives = rmean > survive_threshold
        results.append({
            "step": a["step"], "intent": a["intent"],
            "accepted_at": a["accepted_at"],
            "retest_mean": rmean, "retest_std": rstd,
            "retest_min": min(scores), "retest_max": max(scores),
            "survives": survives, "scores": scores,
        })
        verdict = "SURVIVES" if survives else "VANISHES"
        print(f"  -> step {a['step']}: accepted_at={a['accepted_at']:.4f} "
              f"retest_mean={rmean:.4f}±{rstd:.4f}  {verdict}\n")

    n_kept = len(results)
    n_survive = sum(1 for r in results if r["survives"])
    n_vanish = n_kept - n_survive
    out = {
        "run": args.run, "reps": args.reps, "z": args.z,
        "baseline_mean": base_mean, "baseline_std": base_std,
        "survive_threshold": survive_threshold,
        "n_accepted": n_kept, "n_survive": n_survive, "n_vanish": n_vanish,
        "vanish_fraction": n_vanish / n_kept if n_kept else None,
        "results": results,
    }
    (run_dir / "replication_audit_real.json").write_text(json.dumps(out, indent=2))
    print("== audit summary ==")
    print(f"greedy kept {n_kept} 'win(s)'; under re-testing {n_vanish} VANISH, "
          f"{n_survive} survive ({100*n_vanish/n_kept:.0f}% vanish)" if n_kept else "no accepts")
    print(f"saved: {(run_dir / 'replication_audit_real.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
