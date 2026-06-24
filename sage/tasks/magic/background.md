# Task: MAGIC Gamma Telescope — gamma/hadron classification

## The downstream task
The MAGIC telescope records air-shower events; each is summarized by 10 numeric
features (shape, size, and orientation of the shower image). Classify each event
as a **gamma-ray signal (class 0)** or **hadron background (class 1)**. This is a
real astrophysics / AI-for-Science task. Higher held-out accuracy is better.

## What you are given
The harness hands `fit` a float tensor `X` of shape **(N, 10)** — the features are
already standardized (zero mean, unit variance, fit on train only) — and integer
labels `y` in {0, 1}. The held-out test set is owned by the harness and is never
visible to your code.

## The baseline (primary code you edit)
Logistic regression (a single linear layer), 5 epochs of plain SGD. It works but
is weak (~0.78 val). There is clear headroom — the classes are not linearly
separable.

## Approaches you might consider
- A multi-layer perceptron with nonlinearities (ReLU/tanh), dropout, weight decay.
- Better optimization: Adam, a learning-rate schedule, more epochs.
- Feature engineering: interactions / polynomial features of the 10 inputs.
- Class-imbalance handling (the dataset is ~65% gamma / 35% hadron).
You are free to combine these or do something else entirely.

## Hard rules (a violation makes the eval crash and the proposal is discarded)
- Output a COMPLETE Python file defining `class MyMethod(BaseMethod)` with exactly
  `def fit(self, X, y, seed):` and `def predict(self, X):`.
- `from base_method import BaseMethod`. Imports allowed: torch, torch.nn,
  torch.optim, torch.nn.functional, numpy, math, copy. NO file or network access.
- CPU only (`torch.device("cpu")`). Seed ALL randomness from the `seed` argument.
- Keep training cheap: one eval must finish within the harness time limit
  (a few seconds on CPU). Keep the model small and epochs modest.
- `predict` must return class indices of shape (N,).
