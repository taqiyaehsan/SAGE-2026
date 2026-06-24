# Task: Diabetes progression (REGRESSION example / template)

## The downstream task
Predict a continuous measure of diabetes disease progression one year after
baseline, from 10 standardized physiological features (age, BMI, blood pressure,
six blood-serum measurements). This is a **regression** task scored by **R²**
(coefficient of determination; higher is better, 1.0 is perfect, 0 = no better
than predicting the mean). It exists to show the pipeline handles regression — the
same loop, gates, and audit as the classification tasks.

## What you are given
`fit` gets a float tensor `X` of shape **(N, 10)** (already standardized, fit on
train only) and a float target `y` of shape **(N,)**. `predict` must return
**continuous** predictions of shape (N,). The held-out test set is owned by the
harness and never visible to your code.

## The baseline (primary code you edit)
Linear regression (a single linear layer, MSE loss, 20 epochs of SGD). It works
but is weak. Headroom: an MLP with nonlinearities, normalization, weight decay,
better optimization.

## Approaches you might consider
- A small MLP (ReLU/tanh) with dropout and weight decay.
- Adam + a learning-rate schedule + more epochs.
- Target/feature normalization, feature interactions.

## Hard rules (a violation makes the eval crash and the proposal is discarded)
- Output a COMPLETE Python file defining `class MyMethod(BaseMethod)` with exactly
  `def fit(self, X, y, seed):` and `def predict(self, X):`.
- `from base_method import BaseMethod`. Imports allowed: torch, torch.nn,
  torch.optim, torch.nn.functional, numpy, math, copy. NO file/network access.
- CPU only. Seed ALL randomness from `seed`. Keep training within the time limit.
- `predict` returns CONTINUOUS values of shape (N,) (this is regression, not
  classification — do not return class indices).
