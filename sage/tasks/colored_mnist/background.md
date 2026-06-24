# Task: Colored-digit binary classification

## The downstream task
Classify 28x28 two-channel (colored) images of handwritten digits into 2 classes.
Higher held-out accuracy is better. A convolutional network is a natural fit, but
that is your choice, not a requirement.

## What you are given
The harness hands `fit` a float tensor `X` of shape **(N, 2, 28, 28)** with pixel
values in **[0, 1]** (two color channels), and integer labels `y` in `{0, 1}`. The
labels are moderately noisy, so do not expect to reach perfect accuracy. You may
reshape, normalize, or augment as you see fit. The held-out test set is owned by
the harness and is never visible to your code.

## The baseline (primary code you edit)
A linear (softmax) classifier on the flattened pixels, 3 epochs of plain SGD. It
works but is weak. There is large headroom.

## Approaches you might consider
- A convolutional network (conv -> pool -> conv -> pool -> linear).
- A multi-layer perceptron with nonlinearities, dropout, weight decay.
- Better optimization: Adam, a learning-rate schedule, more epochs.
- Input normalization, light data augmentation.
You are free to combine these or do something else entirely.

## Hard rules (a violation makes the eval crash and the proposal is discarded)
- Output a COMPLETE Python file defining `class MyMethod(BaseMethod)` with exactly
  `def fit(self, X, y, seed):` and `def predict(self, X):`.
- `from base_method import BaseMethod`. Imports allowed: torch, torch.nn,
  torch.optim, torch.nn.functional, numpy, math, copy. NO file or network access.
- CPU only (`torch.device("cpu")`). Seed ALL randomness from the `seed` argument.
- Keep training cheap: one eval must finish within the harness time limit
  (a few seconds on CPU). Keep models small and epochs modest.
- `predict` must return class indices of shape (N,).
