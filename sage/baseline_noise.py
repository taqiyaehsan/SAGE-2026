"""Characterize the TRUE per-eval noise of the baseline MyMethod on the real task.

Runs the UNCHANGED baseline eval N times as fresh, INDEPENDENT draws (the npz is
deleted before each run inside MLRCWorld.evaluate), so we measure genuine run-to-run
wobble -- not a cached/re-read score. This decides whether the real task is already
in the gate-relevant noise regime or whether we must lower NUM_MODELS (Step 6b).

Usage:  python baseline_noise.py --n 8
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from datetime import datetime, timezone
from pathlib import Path

from mlrc_adapter import MLRCWorld, Candidate, REPO_ROOT

RESULTS_ROOT = REPO_ROOT / "skeptic_gate" / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    run_id = args.run_id or f"baseline_noise_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    run_dir = RESULTS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "evals.jsonl"
    log_f = log_path.open("w")

    # mock_llm=True so we never touch the API; we only use the real eval path.
    world = MLRCWorld(proposer=None, snapshot_dir=run_dir / "tmp",
                      mock_eval=False, mock_llm=True)
    base = Candidate(code=world.best_code, intent="baseline", static_ok=True)

    print(f"== baseline noise: {args.n} fresh evals | run {run_id} ==")
    scores = []
    for i in range(args.n):
        t0 = time.time()
        s = world.evaluate(base, seed=i)          # deletes npz -> fresh independent draw
        wall = time.time() - t0
        scores.append(s)
        rec = {"i": i, "score": s, "wall_s": wall, "detail": base.truth}
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        d = base.truth or {}
        print(f"[{i+1}/{args.n}] score={s:.4f}  fq={d.get('forget_quality')}  "
              f"RAU/RAR={d.get('retain_ratio')}  TAU/TAR={d.get('test_ratio')}  ({wall:.0f}s)")

    summary = {
        "run_id": run_id, "n": len(scores), "scores": scores,
        "mean": st.mean(scores), "stdev": st.pstdev(scores),
        "min": min(scores), "max": max(scores), "range": max(scores) - min(scores),
        "cv_pct": (st.pstdev(scores) / st.mean(scores) * 100) if st.mean(scores) else None,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log_f.close()
    print("\n== summary ==")
    print(f"mean={summary['mean']:.4f}  std={summary['stdev']:.4f}  "
          f"min={summary['min']:.4f}  max={summary['max']:.4f}  "
          f"range={summary['range']:.4f}  CV={summary['cv_pct']:.1f}%")
    print(f"logs: {log_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
