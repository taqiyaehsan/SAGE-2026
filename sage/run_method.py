"""Harness: train ONE method file on a task and report its score.

This is the trusted side of the loop (the analog of MLRC's evaluation.py). It is
invoked as a subprocess by local_task.py for every eval, so a method that crashes,
hangs (killed by the parent's timeout), or returns garbage cannot corrupt the
driver -- it just yields a crash score. The harness OWNS:
  * data + the held-out test split (the method never sees test),
  * the seed, CPU-only + single-thread execution (stationarity),
  * a seeded random resample of TRAIN (the noise dial: smaller frac => noisier),
  * scoring.
The METHOD owns only its model + training.

Standalone use (for testing the baseline -- no LLM, no network):
    python run_method.py --task fashionmnist \
        --method tasks/fashionmnist/baseline_method.py --seed 0 --frac 1.0 --split val
Prints a single JSON line: {"score": <accuracy>, ...} or {"error": "..."}.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def _load_method_class(method_path: str):
    """Import a method file by path and return its MyMethod class. The file does
    `from base_method import BaseMethod`, so skeptic_gate/ must be importable."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("candidate_method", method_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # may raise -> caught by caller
    return mod.MyMethod


def _resample_train(n: int, frac: float, seed: int):
    """Seeded random TRAIN subset (no replacement). frac>=1 -> use all (still noisy
    for stochastic methods via their seeded init; deterministic ones are stable)."""
    if frac >= 1.0:
        return torch.arange(n)
    k = max(8, int(round(frac * n)))
    g = torch.Generator().manual_seed(seed + 777)
    return torch.randperm(n, generator=g)[:k]


def _score(metric: str, pred, yev) -> float:
    """Higher is better for BOTH metrics, so the gates work unchanged.
    classification -> accuracy in [0,1]; regression -> R^2 (can be negative)."""
    pred = torch.as_tensor(pred).reshape(-1)
    if pred.shape[0] != yev.shape[0]:
        raise ValueError(f"predict returned {pred.shape[0]} preds for {yev.shape[0]} inputs")
    if metric == "r2":
        pred = pred.float(); y = yev.float()
        ss_res = ((y - pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum().clamp_min(1e-12)
        return float(1.0 - ss_res / ss_tot)
    return float((pred.long() == yev).float().mean())  # accuracy


def evaluate(task: str, method_path: str, seed: int, frac: float, split: str,
             metric: str = "accuracy") -> dict:
    import task_data
    torch.set_num_threads(1)
    data = task_data.load_task(task)
    Xtr, ytr = data["X_tr"], data["y_tr"]
    Xev, yev = data[f"X_{split[:2]}"], data[f"y_{split[:2]}"]  # 'val'->va, 'test'->te
    sub = _resample_train(Xtr.shape[0], frac, seed)
    MyMethod = _load_method_class(method_path)
    t0 = time.time()
    m = MyMethod()
    m.fit(Xtr[sub], ytr[sub], seed)
    score = _score(metric, m.predict(Xev), yev)
    return {"score": score, "metric": metric, "n_train": int(sub.shape[0]),
            "split": split, "seed": seed, "frac": frac, "wall_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--metric", default="accuracy", choices=["accuracy", "r2"])
    ap.add_argument("--out", default=None, help="optional path to write the JSON result")
    a = ap.parse_args()
    try:
        result = evaluate(a.task, a.method, a.seed, a.frac, a.split, a.metric)
    except Exception as e:  # noqa: BLE001 - any failure in agent code -> crash score upstream
        import traceback
        result = {"error": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc()[-1500:]}
    line = json.dumps(result)
    if a.out:
        Path(a.out).write_text(line)
    print(line)
    sys.exit(0 if "score" in result else 3)


if __name__ == "__main__":
    main()
