# Results — what the agents actually wrote, and the scores

Evidence for the headline claims: the **agent-written methods** and their **held-out
scores**. Each code-editing task folder contains the agent's best method
(`best_method.py`, extracted verbatim from the run), the full score table
(`methods.csv` — every method the agent generated, with its intent + val/test/FLOPs),
and the skeptic noise-sweep figure (`regime_curve.png`).

## Code-editing tasks (the agent rewrites the model)

| Task | Baseline → agent (held-out test) | What the agent wrote |
|---|---|---|
| **FashionMNIST** | 0.749 → **0.887** | linear classifier → CNN (BatchNorm, augmentation, label smoothing) |
| **MAGIC Gamma Telescope** | 0.785 → **0.868** | logistic regression → 2-layer MLP |
| **Colored MNIST** | 0.091 → **0.965** | linear (keys on color, collapses on the flipped-color test) → CNN that learns digit **shape** and stays robust |

Open each `best_method.py` for the exact code the agent produced, and `methods.csv`
for the full pool (intent + scores) — i.e. the trajectory from baseline to the final
method.

## MLRC Machine Unlearning — CIFAR-10, named benchmark (`mlrc_unlearning/`)

The skeptic (causal) vs the credulous (greedy) agent at **equal eval budget**. The
skeptic reaches a better, more reliable unlearning score (**mean 0.104 vs greedy 0.092
vs baseline 0.054**).

- `best_method_causal.py` — the unlearning method the agent wrote that the skeptic
  accepted (run s0 reached **0.174**).
- `progressive_results.csv` — per-step incumbent/best score for every run.
- `progression.png` — greedy vs skeptic vs **eval calls spent** (equal-compute axis).
- `summary_causal_s0.json` / `summary_greedy_s0.json` — run summaries (budget, accepts,
  eval calls).

Caveat: the MLRC eval has across-window variance — see
[`../docs/OBSERVATIONS.md`](../docs/OBSERVATIONS.md). Read the trend, not a single decimal.

## Optiver — Trading at the Close (`optiver/`) — a negative result

Hyperparameter-tuning mode on a near-random financial target. `summary.json` shows the
agent made **no measurable progress** (held-out ≈ **0.522**, chance) — there was no
signal to find. Included as an honest negative result: on a task with nothing to learn,
the skeptic correctly banked nothing.

> Datasets/benchmarks are cited in the [top-level README](../README.md).
