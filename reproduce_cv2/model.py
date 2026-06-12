"""Model and loss definitions for the CV2 DeepSLP MLP.

These are kept byte-for-byte equivalent to the architecture recovered from the
saved checkpoints in
    data/interim/ReLU128_f_a075_g15_10folds_pt10/CV2_811_GIV_NN_LR1e2_50e_p10_d01_*.pth

A 4-layer fully-connected network:
    256 -> 128 -> 64 -> 32 -> 1
with BatchNorm1d + ReLU + Dropout(p=0.3) after each of the first three layers
and a single logit output (sigmoid is applied outside the model).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    """Feed-forward binary classifier for genetic-interaction prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size1: int,
        hidden_size2: int,
        hidden_size3: int,
        output_size: int = 1,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(dropout_p)

        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(dropout_p)

        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        self.dropout3 = nn.Dropout(dropout_p)

        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(torch.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(torch.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x).squeeze(-1)
        return x


class FocalLoss(nn.Module):
    """Binary focal loss operating on raw logits.

    Recovered training run used alpha=0.75, gamma=1.5.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
