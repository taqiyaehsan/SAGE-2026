"""Data loading for the local MLRC-style tasks. Owned by the HARNESS, never by the
agent's method code. Each loader returns a fixed train/val/test split (test held
out from the whole loop) as CPU tensors.

Working sets are deliberately subsampled: the point is a real, cheap, STATIONARY
train-and-score loop, not a leaderboard model. A whole-dataset CNN would make the
replication audit (many seeds x many evals) explode for no gain to the gate story.

  fashionmnist  -- raw images (N, 1, 28, 28) in [0,1], 10 classes  (CNN possible)
  magic         -- tabular (N, 10) z-scored, 2 classes (gamma signal vs hadron bg)
  colored_mnist -- RGB images (N, 3, 28, 28) in [0,1], 10 classes (the MNIST digit),
                   with a SPURIOUS color<->group correlation (low digits 0-4 mostly
                   one color, high digits 5-9 the other) that REVERSES between
                   train/val and test. A model that reads color instead of shape wins
                   on train/val and fails on test (see loader).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_FMNIST_ROOT = Path(__file__).resolve().parent / "_data_fmnist"
_MNIST_ROOT = Path(__file__).resolve().parent / "_data_mnist"

# Cache loaded splits so repeated harness calls in one process don't re-read disk.
_CACHE: dict[str, dict] = {}


def _stratified_split(X, y, n_total, seed=0, regression=False):
    """Subsample to a fixed n_total, then a fixed 60/20/20 split. Stratified for
    classification; plain (no stratify) for regression (continuous targets)."""
    from sklearn.model_selection import train_test_split
    y = np.asarray(y)
    strat = None if regression else y
    if n_total is not None and n_total < len(y):
        X, _, y, _ = train_test_split(X, y, train_size=n_total, random_state=seed,
                                      stratify=strat)
        strat = None if regression else y
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=strat)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=seed,
        stratify=None if regression else y_tmp)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def _pack(X_tr, y_tr, X_va, y_va, X_te, y_te, regression=False):
    f = lambda A: torch.from_numpy(np.asarray(A, dtype=np.float32))
    yt = (lambda A: torch.from_numpy(np.asarray(A, dtype=np.float32))) if regression \
        else (lambda A: torch.from_numpy(np.asarray(A, dtype=np.int64)))
    return {"X_tr": f(X_tr), "y_tr": yt(y_tr), "X_va": f(X_va), "y_va": yt(y_va),
            "X_te": f(X_te), "y_te": yt(y_te)}


def load_fashionmnist(n_total: int = 12_000) -> dict:
    """FashionMNIST as raw (N,1,28,28) images in [0,1] so the agent can write a CNN."""
    from torchvision import datasets
    tr = datasets.FashionMNIST(root=str(_FMNIST_ROOT), train=True, download=True)
    X = (tr.data.numpy().astype(np.float32) / 255.0)[:, None, :, :]   # (60000,1,28,28)
    y = tr.targets.numpy()
    parts = _stratified_split(X, y, n_total)
    return _pack(*parts)


def load_magic(n_total: int = 10_000) -> dict:
    """MAGIC Gamma Telescope: 10 z-scored features; gamma 'g'->0, hadron 'h'->1.
    Scaler is fit on TRAIN only (no leakage)."""
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler
    d = fetch_openml("MagicTelescope", version=1, as_frame=False)
    X = d.data.astype(np.float32)
    y = (np.asarray(d.target) == "h").astype(np.int64)
    X_tr, y_tr, X_va, y_va, X_te, y_te = _stratified_split(X, y, n_total)
    sc = StandardScaler().fit(X_tr)
    return _pack(sc.transform(X_tr), y_tr, sc.transform(X_va), y_va,
                 sc.transform(X_te), y_te)


def load_diabetes() -> dict:
    """sklearn diabetes: REGRESSION template (10 features -> continuous target).
    Bundled, no download. The 'example_regression' task; a shape teammates copy."""
    from sklearn.datasets import load_diabetes as _ld
    from sklearn.preprocessing import StandardScaler
    d = _ld()
    X = d.data.astype(np.float32)
    y = d.target.astype(np.float32)
    X_tr, y_tr, X_va, y_va, X_te, y_te = _stratified_split(X, y, None, regression=True)
    sc = StandardScaler().fit(X_tr)
    return _pack(sc.transform(X_tr), y_tr, sc.transform(X_va), y_va,
                 sc.transform(X_te), y_te, regression=True)


def _colorize_group(images, digits, p_red_A, rng):
    """Render each grayscale digit into ONE color channel of an RGB image: red =
    channel 0, green = channel 1, the third channel always 0. Color is a SPURIOUS
    feature correlated with the digit's GROUP (A = digits 0-4, B = digits 5-9):

        P(red) = p_red_A    for group A
        P(red) = 1-p_red_A  for group B

    With p_red_A = 0.90, group A is mostly red and group B mostly green on train/val;
    passing p_red_A = 0.10 (test) reverses it. Color is NEVER stored as a label -- it
    is purely an artifact baked into the pixels. A model that reads color instead of
    shape to decide the group does well on train/val and fails on the reversed test.

      images: (N,28,28) float in [0,1]; digits: (N,) int 0-9.  Returns (N,3,28,28).
    """
    is_A = digits < 5
    p_red = np.where(is_A, p_red_A, 1.0 - p_red_A)
    red = rng.random(len(digits)) < p_red
    out = np.zeros((len(digits), 3, 28, 28), dtype=np.float32)
    out[red, 0] = images[red]            # red channel
    out[~red, 1] = images[~red]          # green channel
    return out


def load_colored_mnist(n_train_total: int = 6_000, n_test: int = 2_000,
                       p: float = 0.90) -> dict:
    """Colored MNIST (cf. Arjovsky et al., IRM) -- a SPURIOUS-CORRELATION dataset.

    The task is the standard 10-class MNIST digit (label 0-9). A spurious COLOR (which
    of two RGB channels holds the digit; see _colorize_group) is correlated with the
    digit's GROUP -- low digits 0-4 vs high digits 5-9 -- at strength `p`:

      * train + val (from MNIST train): group A mostly red, group B mostly green (p=0.90)
      * test        (from MNIST test):  the correlation is REVERSED (1-p = 0.10)

    So a classifier that leans on color to decide the group scores well on train/val
    and FAILS systematically on the held-out test, whose colors are flipped. The harness
    owns the test split; the agent's code never sees it. (Observed: a linear baseline
    overfits to color and collapses on test ~0.81 val -> ~0.09 test, while a CNN learns
    more shape and is far more robust ~0.93 val -> ~0.80 test.)
    """
    from torchvision import datasets
    from sklearn.model_selection import train_test_split
    rng = np.random.default_rng(0)
    tr = datasets.MNIST(root=str(_MNIST_ROOT), train=True, download=True)
    Xtr_all = tr.data.numpy().astype(np.float32) / 255.0      # (60000,28,28)
    dtr_all = tr.targets.numpy()
    if n_train_total is not None and n_train_total < len(dtr_all):
        sel = rng.choice(len(dtr_all), size=n_train_total, replace=False)
        Xtr_all, dtr_all = Xtr_all[sel], dtr_all[sel]
    # train + val share the p=0.90 environment; stratify the split by digit
    idx_tr, idx_va = train_test_split(np.arange(len(dtr_all)), test_size=0.25,
                                      random_state=0, stratify=dtr_all)
    te = datasets.MNIST(root=str(_MNIST_ROOT), train=False, download=True)
    Xte_all = te.data.numpy().astype(np.float32) / 255.0      # (10000,28,28)
    dte_all = te.targets.numpy()
    if n_test is not None and n_test < len(dte_all):
        sel = rng.choice(len(dte_all), size=n_test, replace=False)
        Xte_all, dte_all = Xte_all[sel], dte_all[sel]

    Xtr = _colorize_group(Xtr_all[idx_tr], dtr_all[idx_tr], p, rng)
    Xva = _colorize_group(Xtr_all[idx_va], dtr_all[idx_va], p, rng)
    Xte = _colorize_group(Xte_all, dte_all, 1.0 - p, rng)     # REVERSED environment
    return _pack(Xtr, dtr_all[idx_tr], Xva, dtr_all[idx_va], Xte, dte_all)


# Loader key MUST match the TaskSpec name in local_task.py (the harness loads data
# by task name). "example_regression" is the diabetes regression template.
LOADERS = {"fashionmnist": load_fashionmnist, "magic": load_magic,
           "colored_mnist": load_colored_mnist,
           "example_regression": load_diabetes}


def load_task(name: str) -> dict:
    if name not in LOADERS:
        raise ValueError(f"unknown task {name!r}; choose from {list(LOADERS)}")
    if name not in _CACHE:
        _CACHE[name] = LOADERS[name]()
    return _CACHE[name]
