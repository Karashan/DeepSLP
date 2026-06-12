"""Evaluation metrics shared by training and prediction."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
)


def recall_at_k(y_true, y_pred, k: int) -> float:
    top_k = np.argsort(y_pred)[::-1][:k]
    relevant = np.sum(np.array(y_true)[top_k])
    total_relevant = np.sum(y_true)
    return float(relevant / total_relevant) if total_relevant > 0 else 0.0


def precision_at_k(y_true, y_pred, k: int) -> float:
    top_k = np.argsort(y_pred)[::-1][:k]
    relevant = np.sum(np.array(y_true)[top_k])
    return float(relevant / k) if k > 0 else 0.0


def compute_metrics(y_true, y_prob, k: int = 100) -> Dict[str, float]:
    """ROC-AUC, PR-AUC, average precision and Precision/Recall@K."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return {
        "roc_auc": float(auc(fpr, tpr)),
        "pr_auc": float(auc(recall, precision)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        f"recall_at_{k}": recall_at_k(y_true, y_prob, k),
        f"precision_at_{k}": precision_at_k(y_true, y_prob, k),
    }


def safe_auroc(y_true, y_prob) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")
