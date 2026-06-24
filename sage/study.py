"""Replay study for the code-editing agent: the scientifically-legit greedy-vs-
causal comparison + the Pareto frontier over the methods the agent actually wrote.

Why "replay": in a live closed loop greedy and causal diverge (different incumbents
=> different proposals), so a difference can't be cleanly attributed to the gate.
Here we instead:

  1. generate_pool  -- run the code-editing agent ONCE; record EVERY coherent
     method it writes (baseline -> ... -> CNN/MLP/...). This is the candidate stream.
  2. score_matrix   -- re-score each method over S seeds (val), plus a one-touch
     held-out TEST number, wall-clock, and FLOPs (FlopCounterMode -> works on the
     agent's arbitrary code). This is the trusted measurement matrix.
  3. replay         -- run the SAME gates.py greedy AND causal over the IDENTICAL
     candidate stream and IDENTICAL per-seed measurements. Only the accept rule
     differs => a clean paired ablation (pure policy isolation).
  4. audit          -- of each arm's accepts, how many gains VANISH against the
     full-seed truth (the replication audit, on real measurements).
  5. pareto         -- 3-axis frontier (accuracy up, stability=std down, cost down;
     cost = FLOPs, with wall-clock as context) over the distinct methods. No
     auto-pick: report the frontier so the user chooses by their priorities.

Works for classification (accuracy) and regression (R^2) -- metric comes from the
TaskSpec, and the gates only ever see a scalar where higher is better.

Run:  python study.py <task> [llm] [N_PROPOSALS] [N_SEEDS]
  tasks: fashionmnist | magic | example_regression | (any registered task)
"""

from __future__ import annotations

import csv
import json
import signal
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import task_data
import run_method
import local_task as LT
from gates import Budget, GreedyPolicy, CausalPolicy, Incumbent, run_loop

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
SEED0 = 90_000


# ---------------------------------------------------------------------------
# 1. Generate the candidate pool: the agent writes a sequence of methods.
# ---------------------------------------------------------------------------

def generate_pool(spec: LT.TaskSpec, n_proposals: int, model: str, mock_llm: bool,
                  snapshot_dir: Path) -> list[dict]:
    """Run the code-editing agent with the PROPOSED skeptic (the CAUSAL accept gate)
    driving the incumbent, recording EVERY coherent method it writes (baseline
    included). The incumbent the agent is shown only advances when a gain clears the
    causal noise band -- so the exploration reflects the proposed system, and greedy
    is left as a pure comparison baseline replayed later (Part B). The trusted scoring
    for the ablation/Pareto still happens in the matrix; this loop's evals only steer
    generation."""
    proposer = None if mock_llm else LT.OpenAIProposer(spec, model=model)
    world = LT.LocalTaskWorld(spec, proposer, snapshot_dir, mock_llm=mock_llm)
    pool = [{"idx": 0, "code": spec.baseline_code, "intent": "baseline"}]
    policy = CausalPolicy(k0=2, k_max=6, z=1.0)
    evaluate = lambda c, s: world.evaluate(c, s)
    # seed the incumbent with the baseline's OWN measured scores (>=2 for a band)
    base_cand = LT.Candidate(code=spec.baseline_code, intent="baseline")
    incumbent = Incumbent(scores=[evaluate(base_cand, 5000 + j) for j in range(policy.k0)])
    world.best_score = incumbent.mean
    budget = Budget(n_proposals * policy.k_max + 10)   # plenty for the inner seeds
    idx = 1
    for step in range(n_proposals):
        cand = world.propose()
        if cand is None:
            break
        if not cand.static_ok:
            world.history.append({"intent": cand.intent, "score": None, "accepted": False})
            print(f"    step {step}: CULLED ({cand.static_reason})")
            continue
        dec = policy.decide(cand, evaluate, incumbent, budget, seed0=6000 + step * 10)
        pool.append({"idx": idx, "code": cand.code, "intent": cand.intent}); idx += 1
        mean_sc = (sum(dec.candidate_scores) / len(dec.candidate_scores)
                   if dec.candidate_scores else None)
        if dec.accepted:
            world.best_code = cand.code
            incumbent.scores = dec.candidate_scores      # incumbent adopts its seed scores
            world.best_score = incumbent.mean
        world.history.append({"intent": cand.intent, "score": mean_sc, "accepted": dec.accepted})
        sc = f"{mean_sc:.4f}" if mean_sc is not None else "n/a"
        print(f"    step {step}: {'ACCEPT' if dec.accepted else 'reject'} "
              f"({dec.reason}, {len(dec.candidate_scores)} seeds) mean={sc}  {cand.intent[:50]}")
    return pool


# ---------------------------------------------------------------------------
# 2. Score matrix: re-score each method over S seeds (in-process; the code is
#    already vetted by the coherence gate + one subprocess eval in generation).
# ---------------------------------------------------------------------------

@contextmanager
def _time_limit(seconds: float):
    """Wall-clock cap for in-process scoring via SIGALRM (main thread, Unix). A method
    that exceeds it raises TimeoutError -> caught like any crash -> CRASH_SCORE. This
    is the scoring-stage analog of the per-eval timeout generation already enforces:
    without it, ONE pathological method (e.g. a huge/slow CNN that timed out in
    generation) hangs the whole study indefinitely."""
    def _handler(signum, frame):
        raise TimeoutError(f"scoring exceeded {seconds:.0f}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _load_class(code: str, tmp: Path):
    tmp.write_text(code)
    return run_method._load_method_class(str(tmp))


def _measure_flops(cls, Xtr, ytr) -> int:
    from torch.utils.flop_counter import FlopCounterMode
    fc = FlopCounterMode(display=False)
    try:
        with fc:
            cls().fit(Xtr, ytr, SEED0)
        return int(fc.get_total_flops())
    except Exception:
        return -1


def score_matrix(spec: LT.TaskSpec, pool: list[dict], n_seeds: int,
                 tmpdir: Path) -> tuple[np.ndarray, list[dict]]:
    data = task_data.load_task(spec.name)
    Xtr, ytr = data["X_tr"], data["y_tr"]
    metric = spec.metric
    mat, meta = [], []
    for c in pool:
        # A method can pass the (static) coherence gate yet crash at RUNTIME (e.g.
        # torch.zeros(..., generator=g)). Generation catches this via the subprocess
        # harness; here we run in-process, so guard each method and score a crash as
        # CRASH_SCORE rather than aborting the whole study (the "automated debugging"
        # story: broken edits fail loudly but cheaply, they don't kill the run).
        try:
            # Per-method wall-clock cap (the scoring-stage analog of generation's
            # per-eval timeout): a method that hangs/runs absurdly long is scored as a
            # crash instead of stalling the whole study. ~3x the per-eval limit covers
            # n_seeds fits + test + FLOPs for any healthy method.
            with _time_limit(spec.time_limit * 3):
                cls = _load_class(c["code"], tmpdir / f"cand_{c['idx']}.py")
                vals, walls = [], []
                for s in range(n_seeds):
                    m = cls(); t0 = time.time(); m.fit(Xtr, ytr, SEED0 + s)
                    walls.append(time.time() - t0)
                    vals.append(run_method._score(metric, m.predict(data["X_va"]), data["y_va"]))
                mt = cls(); mt.fit(Xtr, ytr, SEED0 + 999)
                test = run_method._score(metric, mt.predict(data["X_te"]), data["y_te"])
                flops = _measure_flops(cls, Xtr, ytr)
        except Exception as e:  # noqa: BLE001 - any failure in agent code -> crash score
            print(f"    [score_matrix] idx {c['idx']} crashed: {type(e).__name__}: {e}")
            vals = [float(LT.CRASH_SCORE)] * n_seeds
            walls = [0.0]; test = float(LT.CRASH_SCORE); flops = -1
        mat.append(vals)
        meta.append({"idx": c["idx"], "intent": c["intent"], "code": c["code"],
                     "acc": float(np.mean(vals)),
                     "stability": float(np.std(vals, ddof=1)) if n_seeds > 1 else 0.0,
                     "test": float(test), "wall_ms": float(np.median(walls) * 1000),
                     "flops": flops})
    return np.array(mat), meta


# ---------------------------------------------------------------------------
# 3. Replay the SAME gates over the SAME candidates + measurements.
# ---------------------------------------------------------------------------

@dataclass
class _Cand:
    idx: int
    intent: str
    static_ok: bool = True
    truth: Optional[dict] = None


class _ReplayWorld:
    """Feeds gates.run_loop the precomputed matrix instead of live training, and
    tracks the incumbent-index trajectory so the audit knows what each accept
    replaced."""
    def __init__(self, mat: np.ndarray, stream: list[dict]):
        self.mat = mat
        self.stream = stream
        self._iter = iter(stream)
        self._draw = defaultdict(int)      # per-candidate column cursor
        self.incumbent_idx = 0             # baseline
        self.accepts: list[dict] = []

    def propose(self, _rng=None):
        try:
            c = next(self._iter)
        except StopIteration:
            return None
        return _Cand(idx=c["idx"], intent=c["intent"])

    def evaluate(self, cand: _Cand, seed: int) -> float:
        j = self._draw[cand.idx] % self.mat.shape[1]   # distinct real samples
        self._draw[cand.idx] += 1
        return float(self.mat[cand.idx][j])

    def on_accept(self, cand: _Cand, decision) -> None:
        self.accepts.append({"idx": cand.idx, "prev_idx": self.incumbent_idx,
                             "apparent": float(np.mean(decision.candidate_scores))
                             if decision.candidate_scores else None})
        self.incumbent_idx = cand.idx


def replay(arm: str, mat: np.ndarray, pool: list[dict]) -> dict:
    """Replay one accept policy over the fixed candidate stream (pool[1:]); the
    baseline (idx 0) is the initial incumbent. Returns accepts + audit."""
    stream = pool[1:]
    world = _ReplayWorld(mat, stream)
    policy = GreedyPolicy() if arm == "greedy" else CausalPolicy(k0=2, k_max=6, z=1.0)
    budget = Budget(len(stream) * 8 + 4)               # plenty; proposer ends the loop
    base_scores = list(mat[0])                          # baseline's real samples
    run_loop(world.propose, world.evaluate, policy, budget, world.on_accept,
             base_scores, rng=np.random.default_rng(0))
    # replication audit: truth = full-seed mean; a win "vanishes" if true gain <= 0
    truth = mat.mean(axis=1)
    rows = []
    for a in world.accepts:
        gain = float(truth[a["idx"]] - truth[a["prev_idx"]])
        rows.append({**a, "true_after": float(truth[a["idx"]]),
                     "true_before": float(truth[a["prev_idx"]]),
                     "true_gain": gain, "survives": bool(gain > 0.0)})
    n = len(rows); surv = sum(r["survives"] for r in rows)
    return {"arm": arm, "n_accepted": n, "n_vanished": n - surv, "n_survive": surv,
            "accepts": rows}


# ---------------------------------------------------------------------------
# 4. 3-axis Pareto frontier over the distinct methods (no auto-pick).
# ---------------------------------------------------------------------------

def pareto(meta: list[dict]) -> list[dict]:
    def dom(a, b):  # a dominates b: >= acc, <= stability, <= flops, strict somewhere
        ge = a["acc"] >= b["acc"] and a["stability"] <= b["stability"] and a["flops"] <= b["flops"]
        gt = a["acc"] > b["acc"] or a["stability"] < b["stability"] or a["flops"] < b["flops"]
        return ge and gt
    return [r for r in meta if not any(dom(o, r) for o in meta if o is not r)]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_study(task: str, *, use_llm: bool = False, model: str = "gpt-4.1-mini",
              n_proposals: int = 8, n_seeds: int = 8) -> dict:
    t0 = time.time()
    spec = LT.get_task(task)
    kind = f"LLM agent ({model})" if use_llm else "mock proposer"
    print("=" * 80)
    print(f"STUDY [{task}]  metric={spec.metric}  proposer={kind}")
    print("=" * 80)
    snap = RESULTS_DIR / f"study_{task}" / "snapshots"; snap.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_DIR / f"study_{task}" / "tmp"; tmp.mkdir(parents=True, exist_ok=True)

    print("1) agent writes methods (recording every coherent one)...")
    pool = generate_pool(spec, n_proposals, model, not use_llm, snap)
    print(f"   pool = {len(pool)} methods (incl. baseline)")

    print(f"2) scoring matrix over {n_seeds} seeds (val) + test + FLOPs...")
    mat, meta = score_matrix(spec, pool, n_seeds, tmp)

    print("3) replaying SAME gates (greedy vs causal) over identical candidates...")
    arms = {a: replay(a, mat, pool) for a in ("greedy", "causal")}

    # crashed methods (CRASH_SCORE) must not sit on the frontier: their stub
    # stability=0 / flops=-1 look "optimal" on two axes. Exclude failures first.
    valid = [r for r in meta if r["acc"] > LT.CRASH_SCORE / 2]
    front = pareto(valid)
    front_idx = {r["idx"] for r in front}

    higher = "R^2" if spec.metric == "r2" else "acc"
    print(f"\n  --- replay (scientifically-legit paired ablation) ---")
    for a in ("greedy", "causal"):
        r = arms[a]
        print(f"  {a:7s}: accepted {r['n_accepted']}, VANISH on re-test {r['n_vanished']}, "
              f"survive {r['n_survive']}")

    print(f"\n  --- PARETO FRONTIER over the methods the agent wrote (no auto-pick) ---")
    print(f"  {'idx':>3s} {higher+'(val)':>10s} {'stab':>7s} {'GFLOPs':>9s} {'wall(ms)':>9s} "
          f"{'test':>7s}  method")
    for r in sorted(meta, key=lambda r: -r["acc"]):
        mark = "*" if r["idx"] in front_idx else " "
        print(f" {mark}{r['idx']:>3d} {r['acc']:10.4f} {r['stability']:7.4f} "
              f"{r['flops']/1e9:9.2f} {r['wall_ms']:9.0f} {r['test']:7.4f}  {r['intent'][:42]}")
    print(f"  (* = on frontier; {len(front)} of {len(meta)} methods non-dominated)")

    out = {"task": task, "metric": spec.metric, "proposer": kind,
           "n_proposals": n_proposals, "n_seeds": n_seeds, "pool_size": len(pool),
           "replay": arms, "frontier_idx": sorted(front_idx), "methods": meta,
           "wall_s": time.time() - t0}
    outdir = RESULTS_DIR / f"study_{task}"
    tag = "llm" if use_llm else "mock"
    (outdir / f"{tag}.json").write_text(json.dumps(out, indent=2))
    _write_csvs(outdir, tag, task, spec.metric, meta, front_idx, arms)
    print(f"\n  saved -> {outdir}  (wall {out['wall_s']:.1f}s)")
    return out


def _write_csvs(outdir: Path, tag: str, task: str, metric: str, meta: list[dict],
                front_idx: set, arms: dict) -> None:
    """Dump every number to CSV: the per-method score/Pareto table, and the per-accept
    replay audit for both gates (greedy baseline vs causal)."""
    mpath = outdir / f"methods_{tag}.csv"
    with mpath.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "idx", "intent", "metric", "acc_val_mean", "stability_std",
                    "test", "flops", "gflops", "wall_ms", "on_frontier"])
        for r in sorted(meta, key=lambda r: -r["acc"]):
            w.writerow([task, r["idx"], r["intent"], metric, f"{r['acc']:.6f}",
                        f"{r['stability']:.6f}", f"{r['test']:.6f}", r["flops"],
                        f"{r['flops']/1e9:.4f}", f"{r['wall_ms']:.1f}",
                        int(r["idx"] in front_idx)])
    rpath = outdir / f"replay_{tag}.csv"
    with rpath.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "arm", "n_accepted", "n_vanished", "n_survive",
                    "accept_idx", "prev_idx", "apparent", "true_before", "true_after",
                    "true_gain", "survives"])
        for arm in ("greedy", "causal"):
            a = arms[arm]
            if not a["accepts"]:
                w.writerow([task, arm, a["n_accepted"], a["n_vanished"], a["n_survive"],
                            "", "", "", "", "", "", ""])
            for acc in a["accepts"]:
                w.writerow([task, arm, a["n_accepted"], a["n_vanished"], a["n_survive"],
                            acc["idx"], acc["prev_idx"],
                            f"{acc.get('apparent', float('nan')):.6f}",
                            f"{acc['true_before']:.6f}", f"{acc['true_after']:.6f}",
                            f"{acc['true_gain']:.6f}", int(acc["survives"])])
    print(f"  csv  -> {mpath.name}, {rpath.name}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python study.py <task> [llm] [N_PROPOSALS] [N_SEEDS]")
    else:
        task = argv[0]
        use_llm = "llm" in argv[1:]
        nums = [a for a in argv[1:] if a != "llm"]
        npr = int(nums[0]) if len(nums) > 0 else 8
        ns = int(nums[1]) if len(nums) > 1 else 8
        run_study(task, use_llm=use_llm, n_proposals=npr, n_seeds=ns)
