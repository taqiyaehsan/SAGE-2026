"""The editable-method INTERFACE for the local MLRC-style tasks.

The division of labour mirrors MLRC-Bench (the agent owns MyMethod.py; the
harness owns evaluation.py + data):

  * The METHOD (what the agent writes / edits) owns the model and its training.
  * The HARNESS (run_method.py) owns data loading, the seed, CPU-only execution,
    a wall-clock timeout, scoring, and the HELD-OUT TEST SET -- which the method
    code never sees. This is what keeps these tasks cheap and STATIONARY (the
    reason FashionMNIST + MAGIC were chosen) no matter what the agent writes.

A valid method subclasses BaseMethod and implements fit() + predict(). The agent
is free to put any model inside (linear, MLP, CNN, ...) -- "some tasks use a CNN,
some don't" is emergent, not prescribed.
"""

from __future__ import annotations


class BaseMethod:
    """Interface every MyMethod must implement. fit() trains on the downstream
    task; predict() returns class labels. All randomness MUST be derived from the
    `seed` fit() is given, so repeated evals are genuinely independent noisy
    measurements of the SAME method (what the causal gate re-tests)."""

    def fit(self, X, y, seed: int) -> None:
        """Train a model. X: float tensor (N, *feature_shape) on CPU; y: long
        tensor (N,) of class indices. Seed every source of randomness from `seed`."""
        raise NotImplementedError

    def predict(self, X):
        """Return predictions for X as a 1-D array/tensor (N,): class indices for a
        classification task, or continuous values for a regression task."""
        raise NotImplementedError
