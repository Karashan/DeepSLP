"""Neural network model and custom loss for synthetic-lethality prediction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced binary classification.

    Down-weights easy examples so the model focuses on hard negatives.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    ``alpha`` is applied **per-class**: positives receive weight ``alpha``,
    negatives receive ``1 - alpha``.  Set ``alpha > 0.5`` to up-weight the
    minority (positive) class.

    Expects raw logits (applies sigmoid internally via BCEWithLogitsLoss).
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 1.5, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        # Per-class alpha: alpha for positives, (1-alpha) for negatives
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ---------------------------------------------------------------------------
# Residual block used when consecutive layers share the same width
# ---------------------------------------------------------------------------

class _ResidualBlock(nn.Module):
    """Linear → BN → ReLU → Dropout with a skip connection."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SLPredictorMLP(nn.Module):
    """Multi-layer perceptron for binary synthetic-lethality prediction.

    Each hidden layer applies: Linear -> BatchNorm -> ReLU -> Dropout.
    When two consecutive layers share the **same width**, a residual (skip)
    connection is used instead, which improves gradient flow in deeper nets.

    The output head bias is initialised to ``log(pos_prior / (1 - pos_prior))``
    so the model starts near the dataset base-rate and converges faster on
    imbalanced targets.

    Parameters
    ----------
    input_size : int
        Number of input features.
    hidden_sizes : list[int]
        Widths of hidden layers (e.g. ``[128, 64, 32]``).
    dropout : float
        Dropout probability after each hidden layer.
    pos_prior : float or None
        Estimated fraction of positive labels.  Used to bias-initialise
        the output head.  ``None`` skips the initialisation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        dropout: float = 0.3,
        pos_prior: float | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_size
        for h in hidden_sizes:
            if h == in_dim:
                # Same width → residual block (skip connection)
                layers.append(_ResidualBlock(h, dropout))
            else:
                layers.extend([
                    nn.Linear(in_dim, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)

        # Bias-init the output head to the log-prior of the positive class
        if pos_prior is not None and 0 < pos_prior < 1:
            nn.init.constant_(self.head.bias, math.log(pos_prior / (1 - pos_prior)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x).squeeze(-1)
