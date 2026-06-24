"""PRIMARY CODE for the Colored-MNIST task -- the working baseline the agent edits.

A deliberately mediocre linear (softmax) classifier on flattened two-channel pixels:
it really trains and scores, but leaves obvious headroom (a CNN, an MLP, more epochs,
normalization, ...). The agent's job is to EDIT this file to raise held-out accuracy.

Harness contract (do not rely on anything outside it):
  * fit(X, y, seed): X is a float tensor (N, 2, 28, 28) in [0,1]; y is long (N,) in
    {0,1}. Train a model. Seed ALL randomness from `seed`.
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
        n, c, h, w = X.shape
        self.model = nn.Linear(c * h * w, 2).to(DEVICE)
        opt = torch.optim.SGD(self.model.parameters(), lr=0.05)
        loss_fn = nn.CrossEntropyLoss()
        Xf = X.reshape(n, -1)
        g = torch.Generator().manual_seed(seed)
        self.model.train()
        for _ in range(3):                       # only 3 epochs -> headroom
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, 128):
                idx = perm[i:i + 128]
                opt.zero_grad()
                loss = loss_fn(self.model(Xf[idx]), y[idx])
                loss.backward()
                opt.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X.reshape(X.shape[0], -1)).argmax(1)
