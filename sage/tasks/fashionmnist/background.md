# Task: FashionMNIST image classification

## The downstream task
Classify 28x28 grayscale images of clothing into 10 categories (t-shirt, trouser,
pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot). Higher held-out
accuracy is better. This is a real vision task and a CNN is a natural fit, but
that is your choice, not a requirement.

## What you are given
The harness hands `fit` a float tensor `X` of shape **(N, 1, 28, 28)** with pixel
values already scaled to **[0, 1]**, and integer labels `y` in `[0, 9]`. You may
reshape, normalize, or augment as you see fit. The held-out test set is owned by
the harness and is never visible to your code.

## The baseline (primary code you edit)
A linear softmax classifier on flattened pixels, 3 epochs of plain SGD. It works
but is weak (~0.83 val). There is large headroom.

## Approaches you might consider
- A convolutional network (conv -> pool -> conv -> pool -> linear).
- A multi-layer perceptron with nonlinearities, dropout, weight decay.
- Better optimization: Adam, a learning-rate schedule, more epochs.
- Input normalization (per-pixel mean/std), light data augmentation.
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
