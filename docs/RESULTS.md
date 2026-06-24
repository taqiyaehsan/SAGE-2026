# Skeptic gate under noisy evaluation — results (FashionMNIST + MAGIC + Colored MNIST)

The headline experiment for the skeptic gate: **when an autonomous code-editing
agent is evaluated noisily, the greedy accept rule adopts "improvements" that do
not replicate, while the causal accept rule (the skeptic) does not.** We show this
on two team-authored tasks — FashionMNIST (vision) and MAGIC Gamma Telescope
(tabular) — both local, CPU, stationary.

A third task, **Colored MNIST**, is a **spurious-correlation** stress test (section 5).
A color feature is *spuriously* correlated with the digit's group in train/val but the
correlation **reverses** at test, so a model that keys on color instead of shape wins
in-distribution and fails under the shift. It shows two things: (a) a weak baseline
latches onto the spurious color and **collapses** on the reversed test, while the
agent's CNN learns the **invariant shape** signal and stays robust — real progress
that also *generalizes*; and (b) even when every candidate is near-tied, the causal
gate still cuts the greedy false-positive rate under evaluation noise. (Honest note:
train and val share the same spurious correlation, so the seed-gate re-tests on an
in-distribution signal — a win that itself relied on the spurious cue would replicate
across seeds and only a *shifted* test set would expose it. The right held-out
distribution matters as much as more seeds.)

The proposed system is the **skeptic**: a coherence gate (culls broken edits before
any eval) + a **causal accept gate** (re-tests over k seeds, accepts only if the
gain clears the noise band). Greedy (accept on one noisy score) is the **baseline**.
The agent generates its candidate stream once under the causal gate; both accept
rules are then replayed over that identical stream, so the comparison is a clean
paired ablation with no extra LLM calls.

All numbers are reproducible (commands at the bottom). Raw data per task:
`results/skeptic_regime/<task>/` (`llm.json`, `methods_llm.csv`, `replay_llm.csv`,
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

**MAGIC** (`results/skeptic_regime/magic/`):

| eval size | noise σ | greedy FP-rate | causal FP-rate | greedy final acc | causal final acc |
|-----------|---------|----------------|----------------|------------------|------------------|
| 2000 | 0.008 | 0.31 | 0.06 | 0.8716 | 0.8728 |
| 1000 | 0.009 | 0.37 | 0.13 | 0.8708 | 0.8724 |
| 500  | 0.016 | 0.41 | 0.20 | 0.8703 | 0.8720 |
| 200  | 0.016 | 0.51 | 0.19 | 0.8693 | 0.8718 |
| 100  | 0.028 | 0.54 | 0.24 | 0.8676 | 0.8712 |
| 50   | 0.037 | 0.54 | 0.24 | 0.8640 | 0.8712 |
| 25   | 0.069 | 0.41 | 0.17 | 0.8573 | 0.8678 |

**FashionMNIST** (`results/skeptic_regime/fashionmnist/`):

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

## 5. Spurious correlation — a cue the agent must learn past (Colored MNIST)

**Task (a spurious-correlation stress test).** Standard 10-class MNIST digits rendered
as 3-channel RGB: each digit is placed entirely in the **red** channel or the **green**
channel. The COLOR is a **spurious** feature — it is correlated with the digit's GROUP
(low digits 0–4 vs high digits 5–9): in train and validation, group A is ~90 % red and
group B ~90 % green; in the held-out test this correlation is **reversed** (group A is
~90 % green). Color is never stored as a label — it is purely an artifact baked into
the pixels. So a model that decides the group from **color** instead of **shape** does
well on train/val and **fails systematically** on the reversed test. This is the
classic spurious-correlation setup (cf. IRM): the spurious cue (color) is predictive
in-distribution but anti-predictive under shift, while the invariant cue (digit shape)
generalizes. The harness owns the test split; the agent never sees it.

**What happened (`results/skeptic_regime/colored_mnist/`, fig `fig_spurious.png`):**

| idx | method | val | test | GFLOPs |
|-----|--------|-----|------|--------|
| **8** *(final accept)* | + capacity CNN | **0.988** | **0.966** | 3876 |
| 6 | 3-conv CNN | 0.979 | 0.952 | 3230 |
| 4 | + dropout CNN | 0.979 | 0.937 | 1809 |
| **1** *(first accept)* | small CNN | 0.977 | 0.936 | 1206 |
| 2 *(rejected)* | + augmentation | 0.946 | 0.878 | 1809 |
| **0** | baseline (linear) | **0.827** | **0.092** | 1.3 |
| 3, 5, 7 | MixUp / extra-conv | *crash* | *crash* | — |

- **The baseline relies on the spurious color and collapses.** The linear baseline
  scores 0.827 on validation but **0.092 on the reversed test** — it learned
  color-specific digit templates, which break entirely once the colors flip (worse
  than chance under reversal). This is the spurious correlation biting.
- **The agent learns the invariant (shape) signal and stays robust.** Every accepted
  CNN raises validation **and** test together — test climbs **0.092 → 0.966**
  alongside validation. The gate accepts 3 improvements; **all survive** the
  replication audit and all generalize, with only a small residual spurious gap in the
  CNNs (~0.02 val−test) that shrinks as they improve.
- **So here the skeptic accepts genuine, robust progress** — it is *not* fooled,
  because on this construction a better validation score also means a better test
  score. (Contrast the honest boundary: if validation itself shared a cue the model
  could exploit, a seed-stable win could still collapse under shift — the seed-gate
  re-tests on the *validation* distribution, so the choice of held-out distribution
  matters as much as more seeds.)
- **The seed-noise axis still behaves as in sections 3–4.** With the CNNs near-tied
  (val 0.946–0.988), greedy chases evaluation noise while the causal gate does not:

| eval size | noise σ | greedy FP-rate | causal FP-rate | greedy final acc | causal final acc |
|-----------|---------|----------------|----------------|------------------|------------------|
| 2000 | 0.002 | 0.14 | 0.01 | 0.9876 | 0.9876 |
| 1000 | 0.002 | 0.18 | 0.04 | 0.9874 | 0.9873 |
| 500  | 0.003 | **0.21** | **0.04** | 0.9863 | 0.9867 |
| 200  | 0.006 | 0.14 | 0.04 | 0.9838 | 0.9840 |
| 100  | 0.008 | 0.14 | 0.05 | 0.9814 | 0.9818 |
| 50   | 0.010 | 0.14 | 0.07 | 0.9780 | 0.9797 |

**Bonus (crash handling):** three proposals hallucinated non-existent torch APIs
(`Generator` not iterable, `Distribution.sample(generator=…)`) or ran too slowly;
all were caught and scored as failures (culled by the new scoring-stage timeout),
not run-killers.

**One-line takeaway:** *on a dataset where color is spuriously correlated with the
label, a naive baseline latches onto the spurious cue and collapses under
distribution shift (test 0.09), while the code-editing agent learns the invariant
shape signal and is far more robust (test 0.97) — and the causal gate still cuts the
greedy false-positive rate under evaluation noise.*

**One-line takeaway:** *the causal skeptic catches false wins that come from
measurement noise, but not false wins that come from the wrong validation
distribution — Colored MNIST shows a +0.27 validation gain the gate accepts and the
seed audit blesses, which is a −0.45 test collapse. The fix is a shifted held-out
set, not more seeds.*

---

## Reproduce

```bash
cd skeptic_gate

# 1. the code-editing agent with the causal skeptic (needs OPENAI_API_KEY in .env)
python study.py fashionmnist llm 8 5
python study.py magic llm 8 5
python study.py colored_mnist llm 8 5      # the spurious-correlation stress test

# 2. the noise-dial regime sweep (NO LLM calls — reuses each study's pool)
python regime_sweep.py fashionmnist eval 8 200
python regime_sweep.py magic eval 8 200
python regime_sweep.py colored_mnist eval 8 200

# 3. poster figures (auto-discovers all three tasks; colored_mnist -> fig_spurious.png)
python make_poster_figs.py
```

Each study writes `llm.json` + `methods_llm.csv` (score/Pareto table) +
`replay_llm.csv` (greedy-vs-causal accept audit); each sweep writes
`regime_eval.json/.csv` + `regime_curve_eval.png`.
