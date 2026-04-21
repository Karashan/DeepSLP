"""Evaluation metrics and plotting helpers."""
from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def log_message(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def recall_at_k(y_true, y_pred, K: int) -> float:
    top = np.argsort(y_pred)[::-1][:K]
    rel = float(np.sum(np.array(y_true)[top]))
    total = float(np.sum(y_true))
    return rel / total if total > 0 else 0.0


def precision_at_k(y_true, y_pred, K: int) -> float:
    top = np.argsort(y_pred)[::-1][:K]
    rel = float(np.sum(np.array(y_true)[top]))
    return rel / K if K > 0 else 0.0


def evaluate_metrics(model, data_loader, device: str):
    """Return (AUPR, AUROC) on a dataloader (single-logit model with sigmoid)."""
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            logits = model(inputs).squeeze(-1)
            probs = torch.sigmoid(logits)
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
    y = torch.cat(all_labels).numpy()
    p = torch.cat(all_probs).numpy()
    aupr = average_precision_score(y, p)
    try:
        auroc = roc_auc_score(y, p)
    except ValueError:
        auroc = float("nan")
    return aupr, auroc


def plot_roc_pr(all_labels, all_probs, plt_path: Optional[str] = None, K: int = 100):
    """Plot ROC + PR and return [AUROC, AUPR, AP, Recall@K, Precision@K]."""
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    rk = recall_at_k(all_labels, all_probs, K)
    pk = precision_at_k(all_labels, all_probs, K)
    print(f"ROC AUC: {roc_auc:.4f} | PRC AUC: {pr_auc:.4f} | AP: {ap:.4f}")
    print(f"Recall@{K}: {rk:.4f} | Precision@{K}: {pk:.4f}")

    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color="blue", label=f"ROC (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color="green", label=f"PR (AUC = {pr_auc:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve"); plt.legend()
    plt.tight_layout()
    if plt_path is not None:
        ensure_dir(os.path.dirname(plt_path) or ".")
        fig.savefig(plt_path, dpi=200)
    plt.close(fig)
    return [roc_auc, pr_auc, ap, rk, pk]


PERF_COLUMNS = ["AUROC", "AUPR", "AP", "Recall@K", "Precision@K"]


def perf_columns(K: int) -> List[str]:
    return ["AUROC", "AUPR", "AP", f"Recall@{K}", f"Precision@{K}"]


def plot_curves(train_vals: Sequence[float], val_vals: Sequence[float],
                train_label: str = "Train", val_label: str = "Val",
                plt_path: Optional[str] = None) -> None:
    fig = plt.figure(figsize=(9, 5))
    epochs = list(range(1, len(train_vals) + 1))
    plt.plot(epochs, train_vals, label=train_label, marker="o")
    plt.plot(epochs, val_vals,   label=val_label,   marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Value"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    if plt_path is not None:
        ensure_dir(os.path.dirname(plt_path) or ".")
        fig.savefig(plt_path, dpi=200)
    plt.close(fig)
