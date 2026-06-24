# Add your own task

SAGE is plug-and-play. A task is a tiny self-contained problem the agent edits code
for. Adding one is **four steps** — copy the shape of an existing task
(`example_regression` is the cleanest template). Everything lives under `sage/`.

## 1. Write the data loader — `task_data.py`

Add a function that returns a dict of tensors and register it in `LOADERS`:

```python
def load_mytask():
    # ... load / split your data ...
    return _pack(X_tr, y_tr, X_va, y_va, X_te, y_te)   # held-out test is owned here

LOADERS = { ..., "mytask": load_mytask }
```

- Classification: `y` are class indices, metric `accuracy`.
- Regression: `y` are continuous, metric `r2`.
- The harness owns the **test split** — the agent's code never sees it.

## 2. Write the baseline method — `tasks/mytask/baseline_method.py`

A *working but mediocre* model implementing the interface in `base_method.py`:

```python
from base_method import BaseMethod

class MyMethod(BaseMethod):
    def fit(self, X, y, seed: int) -> None:
        ...        # seed ALL randomness from `seed`
    def predict(self, X):
        ...        # class indices (classification) or values (regression)
```

This is the file the agent rewrites. Leave obvious headroom (a stronger model, more
epochs, normalization, augmentation…).

## 3. Write the brief — `tasks/mytask/background.md`

A short problem statement + the space of approaches the agent may try. This is the
prompt the LLM reads each step. Be concrete about the goal and the interface contract;
don't hand it the answer.

## 4. Register the task — `local_task.py`

Add a `TaskSpec` to `TASKS`:

```python
TASKS = { ..., "mytask": TaskSpec("mytask", time_limit=60.0,
                                  regimes=[...], metric="accuracy") }
```

`time_limit` is the per-eval wall-clock cap (seconds); `regimes` are the noise levels
for `regime_sweep.py`; `metric` is `accuracy` or `r2`.

## Run it

```bash
python study.py mytask 4 3          # mock proposer, free
python study.py mytask llm 8 5      # real agent (needs OPENAI_API_KEY)
python regime_sweep.py mytask eval 8 200
```

## What "good" looks like

- **Real progress:** validation *and* held-out test both improve, and track together.
- **The skeptic earns its keep** when the eval is noisy: greedy's false-discovery rate
  (and `n_vanished` in the replication audit) rises above the skeptic's. If your eval is
  too clean (greedy == skeptic), shrink the evaluation set to add noise.
- **A trap** (like `colored_mnist`): validation rises but test collapses — useful for
  showing the *limit* of seed-based re-testing (it buys reproducibility, not validity).
