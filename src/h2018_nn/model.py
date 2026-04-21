"""Neural network architecture and loss identical to the H2018 notebook."""
from __future__ import annotations

import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    """3-hidden-layer MLP with BatchNorm + Dropout, single-logit output."""

    def __init__(self, input_size: int,
                 hidden_size1: int = 128,
                 hidden_size2: int = 64,
                 hidden_size3: int = 32,
                 output_size: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        self.dropout3 = nn.Dropout(dropout)

        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(torch.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(torch.relu(self.bn3(self.fc3(x))))
        return self.fc4(x).squeeze(-1)


class FocalLoss(nn.Module):
    """Binary focal loss on BCE-with-logits."""

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
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
