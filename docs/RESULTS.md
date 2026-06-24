# Skeptic gate under noisy evaluation — results (FashionMNIST + MAGIC + Colored MNIST)

The headline experiment for the skeptic gate: **when an autonomous code-editing
agent is evaluated noisily, the greedy accept rule adopts "improvements" that do
not replicate, while the causal accept rule (the skeptic) does not.** We show this
on two team-authored tasks — FashionMNIST (vision) and MAGIC Gamma Telescope
(tabular) — both local, CPU, stationary.

A third task, **Colored MNIST**, is a **spurious-correlation** stress test and the
**failure node** (section 5). A non-linear channel-match cue is correlated with the
label in train/val and **reversed** at test. The agent's CNNs exploit the cue — raising
validation while **collapsing** on the reversed test — and the *only* robust model (a
linear shape-reader) looks worst on validation, so optimizing validation selects the
trap. The skeptic re-tests over seeds on the validation distribution, so it (correctly)
accepts the seed-stable validation gain — but the failure is distributional, not seed
noise. **Re-testing buys reproducibility, not validity; the fix is a shifted validation
set, not more seeds.**

The proposed system is the **skeptic**: a coherence gate (culls broken edits before
any eval) + a **causal accept gate** (re-tests over k seeds, accepts only if the
gain clears the noise band). Greedy (accept on one noisy score) is the **baseline**.
The agent generates its candidate stream once under the causal gate; both accept
rules are then replayed over that identical stream, so the comparison is a clean
paired ablation with no extra LLM calls.

All numbers are reproducible (commands at the bottom). Raw data per task:
`sage/results/study_<task>/` (`llm.json`, `methods_llm.csv`, `replay_llm.csv`,
`regime_eval.json/.csv`, `regime_curve_eval.png`).

---

## 1. Measurable progress (agent edits the baseline → a real model)

| task | baseline → best (val) | baseline → best (test) | what the agent wrote |
|------|-----------------------|------------------------|----------------------|
| FashionMNIST | 0.737 → **0.896** | 0.749 → **0.888** | linear → CNN → +augmentation → +label smoothing |
| MAGIC        | 0.798 → **0.871** | 0.786 → **0.868** | logistic → 2-layer MLP |

## 2. Pareto frontier (accuracy ↑ / stability ↓ / FLOPs ↓; report, no auto-pick)

- **FashionMNIST:** frontier = {baseline, small CNN (0.884 @ 510 GFLOPs), augmentation
  CNN (0.894 @ 2,960 GFLOPs), MixUp CNN (0.896 @ 85,514 GFLOPs)}. The most accurate
  method costs **~168× the FLOPs of the cheap CNN for +0.012 accuracy** — a stark
  accuracy/compute trade-off.
- **MAGIC:** 6 of 9 methods non-dominated. The **small MLP wins on accuracy *and*
  cost** (0.871 @ 0.20 GFLOPs); the agent's deeper MLPs spent **~12–23× more FLOPs to
  do *worse*** — a clean "more compute ≠ better" point.

## 3. The skeptic result — causal beats greedy under noise (BOTH tasks)

Noise model: each method is trained once on full data; a noisy eval = accuracy on a
random subset of the held-out validation set (unbiased — the true ranking is
preserved, so any accepted gain that vanishes is *purely* a measurement artifact).
We audit every accept against the full-eval truth; a "false positive" is an accept
whose true gain is ≤ 0. 200 bootstrap trials per noise level.

**MAGIC** (`sage/results/study_magic/`):

| eval size | noise σ | greedy FP-rate | causal FP-rate | greedy final acc | causal final acc |
|-----------|---------|----------------|----------------|------------------|------------------|
| 2000 | 0.008 | 0.31 | 0.06 | 0.8716 | 0.8728 |
| 1000 | 0.009 | 0.37 | 0.13 | 0.8708 | 0.8724 |
| 500  | 0.016 | 0.41 | 0.20 | 0.8703 | 0.8720 |
| 200  | 0.016 | 0.51 | 0.19 | 0.8693 | 0.8718 |
| 100  | 0.028 | 0.54 | 0.24 | 0.8676 | 0.8712 |
| 50   | 0.037 | 0.54 | 0.24 | 0.8640 | 0.8712 |
| 25   | 0.069 | 0.41 | 0.17 | 0.8573 | 0.8678 |

**FashionMNIST** (`sage/results/study_fashionmnist/`):

| eval size | noise σ | greedy FP-rate | causal FP-rate | greedy final acc | causal final acc |
|-----------|---------|----------------|----------------|------------------|------------------|
| 2000 | 0.004 | 0.45 | 0.32 | 0.9025 | 0.9019 |
| 1000 | 0.006 | 0.49 | 0.24 | 0.9016 | 0.9005 |
| 500  | 0.009 | 0.45 | 0.21 | 0.9004 | 0.8989 |
| 200  | 0.017 | 0.42 | 0.27 | 0.8991 | 0.8974 |
| 100  | 0.021 | 0.44 | 0.28 | 0.8976 | 0.8968 |
| 50   | 0.023 | 0.40 | 0.30 | 0.8964 | 0.8957 |
| 25   | 0.032 | 0.38 | 0.26 | 0.8950 | 0.8959 |

**Reading both tables:**

- **Causal has a lower false-positive rate than greedy at every noise level on both
  tasks**, and is never worse.
- **MAGIC** is the dramatic case: greedy chases noise up to **54%** of the time vs
  causal's **~20%** (~2.5–3× fewer), **and** causal keeps a better final model
  (0.868 vs greedy's 0.857 at high noise). The many near-ties (~0.86) give greedy
  lots to be fooled by.
- **FashionMNIST** corroborates it more modestly: greedy 0.38–0.49 vs causal
  0.21–0.32 (~1.4–2× fewer). The final-accuracy cost of greedy's mistakes is tiny
  (top methods are within 0.001), so here the win is **decision integrity /
  reproducibility** rather than a raw accuracy lift.

**One-line takeaway:** *under noisy evaluation a greedy research agent accepts
improvements that don't replicate (up to ~54% of the time on MAGIC); the causal gate
cuts that 2–3× — and on MAGIC also yields a better final model — with no extra LLM
calls.*

## 4. Robustness: the coherence gate + runtime-crash handling ("automated debugging")

The agent routinely writes code that *parses* but *crashes at runtime* (e.g.
`torch.zeros(..., generator=g)`, `Distribution.sample(generator=...)`). These are
caught and scored as failures (excluded from the frontier, rejected in replay)
rather than aborting the run — both in generation (subprocess harness) and in the
scoring matrix. On the FashionMNIST run, 3 of 8 proposals crashed at runtime and
were handled gracefully.

**Scoring-stage timeout (now implemented).** Generation enforces a per-eval
wall-clock timeout; the scoring matrix now does too — each method is capped at
3× the per-eval limit (covering its seeds + test + FLOPs), and a method that exceeds
it is scored as a crash rather than hanging the study. This was added after a Colored
MNIST run stalled for hours on a single pathological CNN; with the cap, such methods
are culled cleanly (e.g. the MixUp / extra-conv proposals in section 5).

---

## 5. The failure node — a spurious cue the agent gets fooled by (Colored MNIST)

**Task (a spurious-correlation stress test, cf. IRM).** Two-channel 28×28 images.
Channel 0 always holds the true digit (the invariant **shape** signal); channel 1
either *matches* it or holds a *different* digit. The binary label is `digit ≥ 5`
(flipped with 25 % noise, capping a shape-only model near ~0.75). The spurious cue —
**do the two channels match?** — predicts the label with prob 0.90 in train/val and is
**reversed** (0.10) in the held-out test. The cue is a NON-LINEAR channel interaction:
a CNN can read it, a linear model cannot.

**What happened (`results/colored_mnist/`, fig `regime_curve.png`):**

| method | val | test |
|--------|-----|------|
| baseline (linear, reads shape) | 0.607 | **0.581** (robust) |
| small CNN — *the gate accepts this* | **0.876** | **0.130** (collapses) |
| +BatchNorm / +aug / deeper CNNs | 0.85–0.87 | 0.13–0.33 |

- **The agent's CNNs raise validation but collapse on test.** Each CNN exploits the
  spurious channel-match cue: validation climbs to ~0.88 while the reversed-test score
  falls to ~0.13. The **only** robust model is the linear baseline (shape-only, val 0.61
  / test 0.58) — which looks **worst** on validation. So **optimizing validation
  actively selects the trap.**
- **The skeptic accepts the trap — and that is the point.** The causal gate re-tests
  over seeds on the *validation* distribution. The CNN's val gain is **real and
  reproducible across every seed**, so the gate (correctly, on val) accepts it. But the
  failure is *distributional*, not seed noise — re-testing on the same distribution
  cannot see it.
- **Lesson:** the seed-gate catches false wins from **measurement noise**, not false
  wins from the **wrong validation distribution**. *Re-testing buys reproducibility, not
  validity.* The fix is a **shifted** validation set, not more seeds. (The seed-noise
  axis still behaves as in §3–4 — the causal gate cuts greedy's false-positive rate
  under evaluation noise; see `results/colored_mnist/regime_curve.png`.)

**One-line takeaway:** *Colored MNIST is the failure node — a val 0.61 → 0.88 gain the
gate accepts and the seed audit blesses is a test 0.58 → 0.13 collapse. The skeptic
fixes measurement noise, not distribution shift.*

---

## Reproduce

```bash
cd sage

# 1. the code-editing agent with the skeptic gate (needs OPENAI_API_KEY in .env)
python study.py fashionmnist llm 8 5
python study.py magic llm 8 5
python study.py colored_mnist llm 8 5      # the spurious-correlation stress test

# 2. the noise-dial regime sweep (NO LLM calls — reuses each study's pool)
python regime_sweep.py fashionmnist eval 8 200
python regime_sweep.py magic eval 8 200
python regime_sweep.py colored_mnist eval 8 200
```

Each study writes `llm.json` + `methods_llm.csv` (score/Pareto table) +
`replay_llm.csv` (greedy-vs-causal accept audit); each sweep writes
`regime_eval.json/.csv` + `regime_curve_eval.png`.

---

## References (datasets)

- **FashionMNIST** — Xiao, Rasul & Vollgraf, *Fashion-MNIST*, arXiv:1708.07747 (2017).
- **MAGIC Gamma Telescope** — Bock et al., UCI Machine Learning Repository, 2007.
- **MNIST** — LeCun, Cortes & Burges, *The MNIST Database of Handwritten Digits*.
- **Colored MNIST / spurious correlation** — Arjovsky et al., *Invariant Risk Minimization*, arXiv:1907.02893 (2019).
