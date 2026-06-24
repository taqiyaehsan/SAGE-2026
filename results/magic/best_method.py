"""Agent-written BEST method for magic.

intent : Upgrade to a two-layer MLP with ReLU, weight decay, Adam optimizer, batch training for 20 epochs, and proper seeding to improve classification accuracy.
scores : val=0.8707  test=0.8675  GFLOPs=0.20
vs baseline: test 0.7855 -> 0.8675
(full pool + every method the agent wrote: methods.csv)
"""

"""PRIMARY CODE for the MAGIC Gamma Telescope task -- improved MLP with ReLU, Adam optimizer, weight decay, and longer training."""

import torch
import torch.nn as nn

from base_method import BaseMethod

DEVICE = torch.device("cpu")

class MyMethod(BaseMethod):
    def fit(self, X, y, seed: int) -> None:
        torch.manual_seed(seed)
        n, d = X.shape

        # A simple 2-layer MLP with ReLU nonlinearities
        self.model = nn.Sequential(
            nn.Linear(d, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        ).to(DEVICE)

        # Adam optimizer with weight decay (L2 regularization)
        opt = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        g = torch.Generator().manual_seed(seed)
        batch_size = 128

        self.model.train()
        for epoch in range(20):  # longer training
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                loss = loss_fn(self.model(X[idx]), y[idx])
                loss.backward()
                opt.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X).argmax(1)
