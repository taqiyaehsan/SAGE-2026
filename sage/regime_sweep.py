"""Noise-dial REGIME SWEEP on a FIXED candidate pool (NO LLM calls).

The headline experiment for the skeptic gate. We reuse the methods the agent
already wrote (from results/study_<task>/llm.json), inject EVALUATION noise at
several levels, replay greedy AND causal at each level, and audit every accept
against the FULL-EVAL TRUTH. Output: how greedy's false-positives (accepted gains
that vanish vs the truth) grow with noise, and how the causal gate suppresses
them -- the regime curve.

Noise model (default = "eval"): the skeptic's premise is *noisy evaluation*, so we
inject noise where it belongs. Each method is trained ONCE on full data; we cache
its per-example correctness on the full val set. A noisy eval at level E = score on
a random size-E subset of val. This is:
  * UNBIASED -- E[noisy score] = true val accuracy, so the truth ranking is
    preserved by construction. A vanished gain is PURELY noise-driven (no
    regime-shift confound, unlike shrinking the train set).
  * the honest model of a CHEAP eval (small eval set => noisy => the cost lever).
  * binomial: std ~ sqrt(p(1-p)/E), so small E reliably swamps small true gaps.

Alt mode ("train"): seeded train-subsample (run_method's frac dial) -- realistic
(cheap training is noisier) but confounds variance with a small-data regime shift.
Reported as a secondary variant.

Truth = each method's FULL-eval accuracy. A gain "vanishes" iff it is <= 0 vs truth
-- the honest definition of a false positive. BOOTSTRAP: at each level we redraw the
noisy evals over R trials and report the MEAN false-accepts, so the curve is not one
lucky draw. NO new agent calls => greedy vs causal is a pure paired policy ablation.

Run:  python regime_sweep.py <task> [eval|train] [N_SEEDS] [R_TRIALS] [LEVELS]
  eval  LEVELS = val-subset sizes,  e.g. 2000,500,200,100,50,25
  train LEVELS = train fracs,       e.g. 0.5,0.25,0.1,0.05,0.02
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import task_data
import run_method
import local_task as LT
from gates import GreedyPolicy, CausalPolicy, Budget, run_loop
from study import _ReplayWorld

HERE = Path(__file__).resolve().parent
SEED0 = 1234


def load_pool(task: str):
    p = HERE / f"results/study_{task}/llm.json"
    d = json.loads(p.read_text())
    methods = sorted(d["methods"], key=lambda m: m["idx"])
    return d, methods


def fit_correctness(task: str, methods: list[dict], metric: str, tmp: Path):
    """Train each pooled method ONCE on full train; return (per-example correctness
    matrix over full val [n_methods x n_val], truth vector). For regression the
    'correctness' row is the per-example squared-error-based contribution; we keep
    it simple and only support classification eval-noise here (accuracy)."""
    torch.set_num_threads(1)
    data = task_data.load_task(task)
    Xtr, ytr, Xva, yva = data["X_tr"], data["y_tr"], data["X_va"], data["y_va"]
    rows, truth = [], []
    for m in methods:
        f = tmp / f"cand_{m['idx']}.py"
        f.write_text(m["code"])
        cls = run_method._load_method_class(str(f))
        t0 = time.time()
        try:
            mm = cls()
            mm.fit(Xtr, ytr, SEED0)
            pred = torch.as_tensor(mm.predict(Xva)).reshape(-1).long()
            correct = (pred == yva).float().numpy()
        except Exception:                          # noqa: BLE001
            correct = np.zeros(yva.shape[0])
        rows.append(correct)
        truth.append(float(correct.mean()))
        print(f"    fit idx {m['idx']}: true val acc={correct.mean():.4f}  [{time.time()-t0:.0f}s]")
    return np.array(rows), np.array(truth)


def eval_matrix(correct: np.ndarray, n_seeds: int, E: int, rng) -> np.ndarray:
    """n_seeds noisy evals per method: each = accuracy on a random size-E val subset."""
    n_val = correct.shape[1]
    E = min(E, n_val)
    mat = np.empty((correct.shape[0], n_seeds))
    for s in range(n_seeds):
        idx = rng.integers(0, n_val, size=E)        # subsample-with-replacement
        mat[:, s] = correct[:, idx].mean(axis=1)
    return mat


def train_matrix(task: str, methods: list[dict], n_seeds: int, frac: float,
                 metric: str, tmp: Path) -> np.ndarray:
    """Secondary mode: re-fit each method on a seeded train subsample (frac)."""
    torch.set_num_threads(1)
    data = task_data.load_task(task)
    Xtr, ytr, Xva, yva = data["X_tr"], data["y_tr"], data["X_va"], data["y_va"]
    n = Xtr.shape[0]
    rows = []
    for m in methods:
        f = tmp / f"cand_{m['idx']}.py"; f.write_text(m["code"])
        cls = run_method._load_method_class(str(f))
        vals = []
        for s in range(n_seeds):
            sub = run_method._resample_train(n, frac, SEED0 + s)
            try:
                mm = cls(); mm.fit(Xtr[sub], ytr[sub], SEED0 + s)
                vals.append(run_method._score(metric, mm.predict(Xva), yva))
            except Exception:                       # noqa: BLE001
                vals.append(float(LT.CRASH_SCORE))
        rows.append(vals)
    return np.array(rows)


def replay_audit(mat: np.ndarray, methods: list[dict], truth: np.ndarray,
                 arm: str, trial_seed: int) -> dict:
    stream = [{"idx": m["idx"], "intent": m["intent"]} for m in methods[1:]]
    world = _ReplayWorld(mat, stream)
    policy = GreedyPolicy() if arm == "greedy" else CausalPolicy(k0=2, k_max=6, z=1.0)
    budget = Budget(len(stream) * 8 + 4)
    run_loop(world.propose, world.evaluate, policy, budget, world.on_accept,
             list(mat[0]), rng=np.random.default_rng(trial_seed))
    n_van, final_idx = 0, 0
    for a in world.accepts:
        if float(truth[a["idx"]] - truth[a["prev_idx"]]) <= 0.0:
            n_van += 1
        final_idx = a["idx"]
    return {"n_accepted": len(world.accepts), "n_vanished": n_van,
            "final_idx": final_idx, "final_true_acc": float(truth[final_idx])}


def sweep(task: str, mode: str, n_seeds: int, r_trials: int, levels: list) -> dict:
    d, methods = load_pool(task)
    metric = d["metric"]
    tmp = HERE / f"results/study_{task}/regime_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"REGIME SWEEP [{task}] mode={mode} metric={metric}  pool={len(methods)} "
          f"methods (NO LLM)  n_seeds={n_seeds} R={r_trials}")
    correct = truth = None
    if mode == "eval":
        print("  fitting each method once (full data) -> per-example correctness...")
        correct, truth = fit_correctness(task, methods, metric, tmp)
    else:
        # truth from full-data study run
        truth = np.array([float(m["acc"]) for m in methods])
    print("  truth (true acc) by idx: "
          + "  ".join(f"{i}:{truth[i]:.3f}" for i in range(len(truth))))
    points = []
    for lv in levels:
        t0 = time.time()
        rng = np.random.default_rng(0)
        if mode == "eval":
            mat = eval_matrix(correct, n_seeds, int(lv), rng)
        else:
            mat = train_matrix(task, methods, n_seeds, float(lv), metric, tmp)
        noise = float(np.mean([np.std(mat[i], ddof=1) for i in range(1, mat.shape[0])]))
        res = {"greedy": [], "causal": []}
        for t in range(r_trials):
            if mode == "eval":
                mt = eval_matrix(correct, n_seeds, int(lv), rng)
            else:
                cols = rng.integers(0, n_seeds, size=n_seeds)
                mt = mat[:, cols]
            for arm in ("greedy", "causal"):
                res[arm].append(replay_audit(mt, methods, truth, arm, trial_seed=t))
        row = {"level": lv, "noise_std": noise}
        for arm in ("greedy", "causal"):
            van = np.array([r["n_vanished"] for r in res[arm]])
            acc = np.array([r["final_true_acc"] for r in res[arm]])
            row[arm] = {"mean_vanished": float(van.mean()),
                        "fp_rate": float((van > 0).mean()),
                        "max_vanished": int(van.max()),
                        "mean_final_acc": float(acc.mean())}
        points.append(row)
        print(f"  level={str(lv):<6} noise(std)={noise:.4f}  "
              f"greedy FP-rate={row['greedy']['fp_rate']:.2f} "
              f"(mean {row['greedy']['mean_vanished']:.2f})  |  "
              f"causal FP-rate={row['causal']['fp_rate']:.2f} "
              f"(mean {row['causal']['mean_vanished']:.2f})  "
              f"[{time.time()-t0:.0f}s]")
    out = {"task": task, "mode": mode, "n_seeds": n_seeds, "r_trials": r_trials,
           "truth": truth.tolist(), "points": points}
    (HERE / f"results/study_{task}/regime_{mode}.json").write_text(json.dumps(out, indent=2))
    cpath = HERE / f"results/study_{task}/regime_{mode}.csv"
    with cpath.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "mode", "level", "noise_std",
                    "greedy_fp_rate", "greedy_mean_vanished", "greedy_max_vanished",
                    "greedy_mean_final_acc",
                    "causal_fp_rate", "causal_mean_vanished", "causal_max_vanished",
                    "causal_mean_final_acc"])
        for p in points:
            g, c = p["greedy"], p["causal"]
            w.writerow([task, mode, p["level"], f"{p['noise_std']:.6f}",
                        f"{g['fp_rate']:.4f}", f"{g['mean_vanished']:.4f}",
                        g["max_vanished"], f"{g['mean_final_acc']:.6f}",
                        f"{c['fp_rate']:.4f}", f"{c['mean_vanished']:.4f}",
                        c["max_vanished"], f"{c['mean_final_acc']:.6f}"])
    print(f"  saved -> results/study_{task}/regime_{mode}.json + .csv")
    return out


def plot(task: str, out: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"  (plot skipped: {e})"); return
    pts = out["points"]; mode = out["mode"]
    x = [p["noise_std"] for p in pts]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(x, [p["greedy"]["mean_vanished"] for p in pts], "o-", label="greedy")
    ax[0].plot(x, [p["causal"]["mean_vanished"] for p in pts], "s-", label="causal")
    ax[0].set_xlabel("eval noise (mean seed std)"); ax[0].set_ylabel("false accepts (mean)")
    ax[0].set_title(f"Skeptic regime curve ({mode} noise)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(x, [p["greedy"]["mean_final_acc"] for p in pts], "o-", label="greedy")
    ax[1].plot(x, [p["causal"]["mean_final_acc"] for p in pts], "s-", label="causal")
    ax[1].set_xlabel("eval noise (mean seed std)"); ax[1].set_ylabel("final method TRUE acc")
    ax[1].set_title("Does skepticism cost real gains?"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout()
    fp = HERE / f"results/study_{task}/regime_curve_{mode}.png"
    fig.savefig(fp, dpi=130); print(f"  saved -> {fp}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        print("usage: python regime_sweep.py <task> [eval|train] [N_SEEDS] [R_TRIALS] [LEVELS]")
        sys.exit(0)
    task = argv[0]
    mode = argv[1] if len(argv) > 1 else "eval"
    n_seeds = int(argv[2]) if len(argv) > 2 else 8
    r_trials = int(argv[3]) if len(argv) > 3 else 200
    if len(argv) > 4:
        levels = [float(x) if mode == "train" else int(x) for x in argv[4].split(",")]
    else:
        levels = ([2000, 1000, 500, 200, 100, 50, 25] if mode == "eval"
                  else [0.5, 0.25, 0.1, 0.05, 0.02])
    out = sweep(task, mode, n_seeds, r_trials, levels)
    plot(task, out)
