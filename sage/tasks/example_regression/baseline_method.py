"""PRIMARY CODE for the regression EXAMPLE task (sklearn diabetes) -- the working
baseline the agent edits. This is a TEMPLATE showing the pipeline handles
regression: predict() returns continuous values and the harness scores with R^2.

A deliberately mediocre linear-regression baseline. Headroom: an MLP,
nonlinearities, normalization, regularization, more epochs.

Harness contract:
  * fit(X, y, seed): X is float (N, 10); y is float (N,) continuous target.
    Seed ALL randomness.
  * predict(X): return CONTINUOUS predictions, shape (N,).
  * CPU only. No file/network access. Harness enforces a wall-clock limit.
"""

import torch
import torch.nn as nn

from base_method import BaseMethod

DEVICE = torch.device("cpu")


class MyMethod(BaseMethod):
    def fit(self, X, y, seed: int) -> None:
        torch.manual_seed(seed)
        n, d = X.shape
        self.model = nn.Linear(d, 1).to(DEVICE)          # linear regression
        opt = torch.optim.SGD(self.model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        g = torch.Generator().manual_seed(seed)
        self.model.train()
        for _ in range(20):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, 64):
                idx = perm[i:i + 64]
                opt.zero_grad()
                loss_fn(self.model(X[idx]).reshape(-1), y[idx]).backward()
                opt.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X).reshape(-1)             # continuous predictions
