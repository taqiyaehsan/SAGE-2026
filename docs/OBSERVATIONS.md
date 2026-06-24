# Observation: the local evaluation is non-stationary

## Short version

The identical, unmodified baseline `MyMethod.py` was evaluated many times on one
Apple-Silicon (MPS) laptop. The score it produced depended on **when** the eval ran:
tightly clustered within a short time window, but shifting by ~100× across windows,
with the eval also slowing down as the scores dropped. This is **system-state drift,
not clean i.i.d. measurement noise**, and it has direct consequences for how the
benchmark numbers can (and cannot) be used.

## The data

All from the same byte-for-byte baseline (1-epoch fine-tune on the retain set).
Raw per-eval logs are in `results/baseline_observation/`.

| window | n | final score             | forget_quality | utility ratios | eval wall-time |
|--------|---|-------------------------|----------------|----------------|----------------|
| A      | 8 | 0.116–0.117 (CV ≈ 0.3%) | ~0.116         | ~1.00          | ~180 s         |
| B      | 1 | 0.054                   | ~0.054         | ~1.00          | ~180 s         |
| C      | 3 | 0.0008–0.0012           | ~0.004         | ~1.00          | ~290 s         |

- Window A: `window_A_n8_evals.jsonl` / `window_A_n8_summary.json`
- Window C: `window_C_recheck_evals.jsonl`

The utility ratios (retain/test accuracy vs. the retrained reference) stay ~1.0
throughout; the entire swing is in the **forgetting-quality** term.

## Why this happens (best current explanation)

One `eval` is not inference on a fixed model. It is:

1. `NUM_MODELS = 10` independent **unlearning runs** — each deep-copies the
   pretrained model and **trains** it (the baseline does an epoch of SGD on the
   retain set), then runs inference; and
2. a scoring step that **trains** membership-inference attack classifiers to
   distinguish the unlearned models from the retrained reference.

The training in step 1 is where nondeterminism enters: BatchNorm running-stat
updates and floating-point reductions on MPS (with the CPU-fallback path) are
sensitive to runtime conditions. After hours of back-to-back evals the machine
appears to enter a different performance state — the eval wall-time rises from
~180 s to ~290 s, consistent with **thermal throttling or device contention** —
and in that state the ten unlearned models train differently, driving the
forgetting score toward zero.

The data-split RNG and the retain-loader shuffle are seeded, so those are *not* the
cause; the unseeded pieces (forget-loader shuffle, the reference-noise draw) are too
small to explain a 100× swing. The training path is the remaining explanation.

## Consequences

1. **A single score on this machine is not trustworthy.** The same code yields
   0.001, 0.054, or 0.117 with nothing changed but elapsed time.
2. **Sequential arm comparisons are confounded.** If arm X runs during a "high"
   window and arm Y during a "low" window, the difference reflects the **clock**,
   not the arm. Any greedy-vs-skeptic comparison on this laptop is therefore not
   reliable.
3. **It is not the noise model the skeptic gate assumes.** The causal gate is built
   for i.i.d. noise around a stable true value; non-stationary drift violates that.
4. **It explains earlier confusion.** An apparent "baseline level shift" (0.054 vs
   0.117) and a tempting "+0.046 improvement" accepted on one eval are both
   artifacts of this drift, not real effects.

## Recommendations

- Treat laptop/MPS numbers as **qualitative** only.
- Run quantitative comparisons on a **CUDA** machine, which should be far more
  stationary.
- Add determinism controls before comparing arms:
  - `torch.manual_seed(...)`, `torch.use_deterministic_algorithms(True)`,
    `PYTHONHASHSEED`, and a fixed cuDNN configuration;
  - **interleave or randomize the order** in which arms are evaluated so any
    residual drift averages out instead of aligning with one arm;
  - re-measure the incumbent adjacent in time to each candidate, so a paired
    comparison cancels slow drift.
- Re-run the baseline-noise characterization (`skeptic_gate/baseline_noise.py`) in
  the target environment and confirm the distribution is stationary **before**
  trusting any arm comparison.

## How to reproduce the check

```bash
cd skeptic_gate
../.venv/bin/python baseline_noise.py --n 8
# wait, then run again later (or under load) and compare the two summaries
../.venv/bin/python baseline_noise.py --n 8
```

If the two summaries disagree by more than their stated spread, the environment is
non-stationary and arm comparisons there should not be trusted.
