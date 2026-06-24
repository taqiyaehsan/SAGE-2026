"""Colored MNIST — the method the gate ACCEPTED (the failure node).

intent : Replace the linear model with a small convolutional neural network, add input normalization, use Adam optimizer, increase training to 10 epochs, and apply batch training with shuffling to improve accu
scores : val=0.876  test=0.130
The skeptic accepts this on validation (the val gain is real & seed-stable),
but it leans on the spurious channel-match cue and COLLAPSES on the flipped
test: val 0.88 -> test 0.13. The only
robust model is the linear baseline (val 0.61, test 0.58),
which looks WORST on val — so optimizing validation SELECTS the trap.
"""

"""PRIMARY CODE for the Colored-MNIST task -- improved CNN classifier with normalization and Adam optimizer."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_method import BaseMethod

DEVICE = torch.device("cpu")

class MyMethod(BaseMethod):
    def fit(self, X, y, seed: int) -> None:
        torch.manual_seed(seed)
        n, c, h, w = X.shape

        # Normalize input per channel
        mean = X.mean(dim=(0, 2, 3), keepdim=True)  # shape (2,1,1)
        std = X.std(dim=(0, 2, 3), keepdim=True) + 1e-6
        X = (X - mean) / std

        # Define a small CNN
        class CNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2)
                self.fc1 = nn.Linear(32 * 7 * 7, 64)
                self.fc2 = nn.Linear(64, 2)
                self.dropout = nn.Dropout(0.3)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = self.pool(x)
                x = F.relu(self.conv2(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x

        self.model = CNN().to(DEVICE)

        opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        batch_size = 128
        g = torch.Generator().manual_seed(seed)

        self.model.train()
        for epoch in range(10):  # more epochs
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb = X[idx].to(DEVICE)
                yb = y[idx].to(DEVICE)
                opt.zero_grad()
                out = self.model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            # Normalize using stored mean/std from training
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            std = X.std(dim=(0, 2, 3), keepdim=True) + 1e-6
            Xn = (X - mean) / std
            out = self.model(Xn.to(DEVICE))
            return out.argmax(1).cpu()
