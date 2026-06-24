"""PRIMARY CODE for the MAGIC Gamma Telescope task -- the working baseline the
agent edits.

A deliberately mediocre logistic-regression baseline (a single linear layer) on
the 10 standardized air-shower features: it really trains and scores, but leaves
obvious headroom (an MLP, nonlinearities, regularization, more epochs, feature
interactions). The agent's job is to EDIT this file to raise held-out accuracy.

Harness contract (do not rely on anything outside it):
  * fit(X, y, seed): X is a float tensor (N, 10) of standardized features; y is
    long (N,) with 0 = gamma signal, 1 = hadron background. Seed ALL randomness.
  * predict(X): return predicted class indices, shape (N,).
  * CPU only. No file or network access. The harness enforces a wall-clock limit.
"""

import torch
import torch.nn as nn

from base_method import BaseMethod

DEVICE = torch.device("cpu")


class MyMethod(BaseMethod):
    def fit(self, X, y, seed: int) -> None:
        torch.manual_seed(seed)
        n, d = X.shape
        self.model = nn.Linear(d, 2).to(DEVICE)        # logistic regression
        opt = torch.optim.SGD(self.model.parameters(), lr=0.05)
        loss_fn = nn.CrossEntropyLoss()
        g = torch.Generator().manual_seed(seed)
        self.model.train()
        for _ in range(5):                              # only 5 epochs -> headroom
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, 128):
                idx = perm[i:i + 128]
                opt.zero_grad()
                loss_fn(self.model(X[idx]), y[idx]).backward()
                opt.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X).argmax(1)
