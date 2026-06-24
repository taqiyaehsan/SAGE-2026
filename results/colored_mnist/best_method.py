"""Agent-written BEST method for colored_mnist.

intent : Improve generalization and held-out accuracy by adding label smoothing in the loss, replacing LeakyReLU with GELU activations for smoother gradients, and slightly increasing training epochs with the e
scores : val=0.9879  test=0.9655  GFLOPs=3875.56
vs baseline: test 0.0915 -> 0.9655
(full pool + every method the agent wrote: methods.csv)
"""

"""PRIMARY CODE for the Colored-MNIST task -- enhanced CNN with 3 conv blocks, dropout, GELU activations, label smoothing, cosine annealing LR, and mild random affine augmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_method import BaseMethod

DEVICE = torch.device("cpu")

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, preds, target):
        # preds: (N, C), target: (N,)
        log_probs = F.log_softmax(preds, dim=-1)
        n_classes = preds.size(1)
        with torch.no_grad():
            # Smooth targets
            true_dist = torch.zeros_like(preds)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

class MyMethod(BaseMethod):
    def __init__(self):
        super().__init__()
        # Define a conv net with 3 conv blocks, batchnorm, dropout and GELU activations
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),           # 28->14
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),           # 14->7
            nn.Dropout(0.15),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),           # 7->3
            nn.Dropout(0.2),

            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        ).to(DEVICE)

        # Identity augmentation placeholder (augmentation done manually in fit)
        self.augment = nn.Sequential(nn.Identity())

    def fit(self, X, y, seed: int) -> None:
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        n = X.shape[0]

        # Normalize input to zero mean 0.5 and std 0.5 (simple normalization)
        X = (X - 0.5) / 0.5

        max_rotate = 10  # degrees
        max_translate = 2  # pixels

        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)

        batch_size = 128
        epochs = 18  # Slightly increased epochs

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        for _ in range(epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]

                batch = X[idx]

                angles = (torch.rand(batch.shape[0], generator=g) * 2 - 1) * max_rotate
                translations = (torch.rand(batch.shape[0], 2, generator=g) * 2 - 1) * max_translate

                angles_rad = angles * 3.14159265 / 180.0

                cos = torch.cos(angles_rad)
                sin = torch.sin(angles_rad)

                tx = translations[:,0] * 2 / 28
                ty = translations[:,1] * 2 / 28

                affine_matrices = torch.zeros(batch.shape[0], 2, 3, device=batch.device)
                affine_matrices[:,0,0] = cos
                affine_matrices[:,0,1] = -sin
                affine_matrices[:,0,2] = tx
                affine_matrices[:,1,0] = sin
                affine_matrices[:,1,1] = cos
                affine_matrices[:,1,2] = ty

                grid = F.affine_grid(affine_matrices, batch.size(), align_corners=False)
                batch = F.grid_sample(batch, grid, padding_mode='border', align_corners=False)

                opt.zero_grad()
                out = self.model(batch)
                loss = loss_fn(out, y[idx])
                loss.backward()
                opt.step()
            scheduler.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = (X - 0.5) / 0.5
            out = self.model(X)
            return out.argmax(1)
