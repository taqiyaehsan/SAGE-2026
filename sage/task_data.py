"""Data loading for the local MLRC-style tasks. Owned by the HARNESS, never by the
agent's method code. Each loader returns a fixed train/val/test split (test held
out from the whole loop) as CPU tensors.

Working sets are deliberately subsampled: the point is a real, cheap, STATIONARY
train-and-score loop, not a leaderboard model. A whole-dataset CNN would make the
replication audit (many seeds x many evals) explode for no gain to the gate story.

  fashionmnist  -- raw images (N, 1, 28, 28) in [0,1], 10 classes  (CNN possible)
  magic         -- tabular (N, 10) z-scored, 2 classes (gamma signal vs hadron bg)
  colored_mnist -- 2-channel images (N, 2, 28, 28) in [0,1], 2 classes, with a SPURIOUS
                   non-linear channel-match cue correlated with the label in train/val
                   and REVERSED at test: a CNN exploits the cue (high val) then COLLAPSES
                   on the flipped test, while a linear shape-reader stays robust -- the
                   failure-node stress test (see loader).
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


def _colorize_match(images, labels, e, rng):
    """Two-channel images whose SPURIOUS feature is a NON-LINEAR channel interaction.

    Channel 0 always holds the true digit (this carries the SHAPE signal -- the
    invariant cue). Channel 1 either MATCHES it (digits identical) or holds a
    DIFFERENT random digit, selected by the spurious bit:

        match = label XOR Bernoulli(e)      (1 -> channels match, 0 -> they differ)

    so "do the two channels match" predicts the label with probability 1-e. Crucially
    BOTH classes put a clean digit in each channel, so the per-pixel/per-channel
    MARGINALS are identical: the spurious cue is readable ONLY from the joint relation
    between the channels (an interaction), which a CNN/MLP can learn but a LINEAR model
    cannot. That capability gap is deliberate -- it leaves validation HEADROOM above
    the (linear, shape-only) baseline so a CNN can climb by exploiting the spurious
    cue, and the causal gate gets a real apparent win to (correctly, on val) accept.

      images: (N,28,28) float in [0,1]; labels: (N,) binary {0,1}; e: cue-flip prob.
    Returns (N,2,28,28) float32.
    """
    n = len(labels)
    match = np.logical_xor(labels.astype(bool), rng.random(n) < e)   # spurious bit
    other = images[rng.permutation(n)]                               # a different digit
    out = np.zeros((n, 2, 28, 28), dtype=np.float32)
    out[:, 0] = images
    out[:, 1] = np.where(match[:, None, None], images, other)
    return out


def load_colored_mnist(n_total: int = 5_000, e_train: float = 0.10,
                       e_val: float = 0.10, e_test: float = 0.90,
                       label_noise: float = 0.25) -> dict:
    """Colored MNIST (cf. Arjovsky et al., IRM) -- a SPURIOUS-CORRELATION stress test.

    The binary label is digit>=5, then flipped with prob `label_noise` (0.25), so a
    purely SHAPE-based predictor is capped around ~1-label_noise. A two-channel
    SPURIOUS cue (whether the channels match; see _colorize_match) is then correlated
    with the (noisy) label:

      * train + val: the cue matches the label with prob 1-e_train/val = 0.90
      * test:        the cue matches the label with prob 1-e_test     = 0.10  (FLIPPED)

    On train/val the spurious cue (0.90) beats the shape (~0.75), so any model that
    latches onto it scores ~0.90 in validation -- stably, across every seed -- yet
    COLLAPSES to ~0.10 on the held-out test, whose correlation is reversed. train and
    val share one distribution, so the causal gate's seed re-testing is working
    CORRECTLY (the val gains are real and reproducible); the failure is distributional,
    not seed noise -- which is exactly what makes this a clean stress test of what the
    seed-gate can and cannot catch. The cue is a NON-LINEAR channel interaction, so the
    linear baseline can only read shape (~0.64 val, and ~0.64 test -- robust), leaving
    headroom a CNN fills by exploiting the spurious cue. The harness owns the test
    split; the agent's code never sees it.
    """
    from torchvision import datasets
    from sklearn.model_selection import train_test_split
    tr = datasets.MNIST(root=str(_MNIST_ROOT), train=True, download=True)
    X = tr.data.numpy().astype(np.float32) / 255.0            # (60000,28,28)
    digit = tr.targets.numpy()
    rng = np.random.default_rng(0)
    if n_total is not None and n_total < len(digit):
        sel = rng.choice(len(digit), size=n_total, replace=False)
        X, digit = X[sel], digit[sel]
    y = (digit >= 5).astype(np.int64)                         # shape label
    y = np.logical_xor(y.astype(bool), rng.random(len(y)) < label_noise).astype(np.int64)
    idx = np.arange(len(y))
    idx_tmp, idx_te = train_test_split(idx, test_size=0.20, random_state=0, stratify=y)
    idx_tr, idx_va = train_test_split(idx_tmp, test_size=0.25, random_state=0,
                                      stratify=y[idx_tmp])
    Xtr = _colorize_match(X[idx_tr], y[idx_tr], e_train, rng)
    Xva = _colorize_match(X[idx_va], y[idx_va], e_val, rng)
    Xte = _colorize_match(X[idx_te], y[idx_te], e_test, rng)
    return _pack(Xtr, y[idx_tr], Xva, y[idx_va], Xte, y[idx_te])


# Loader key MUST match the TaskSpec name in local_task.py (the harness loads data
# by task name).
LOADERS = {"fashionmnist": load_fashionmnist, "magic": load_magic,
           "colored_mnist": load_colored_mnist}


def load_task(name: str) -> dict:
    if name not in LOADERS:
        raise ValueError(f"unknown task {name!r}; choose from {list(LOADERS)}")
    if name not in _CACHE:
        _CACHE[name] = LOADERS[name]()
    return _CACHE[name]
