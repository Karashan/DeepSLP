"""Training loop, evaluation metrics, and plotting utilities."""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .models import FocalLoss, SLPredictorMLP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    top_k = np.argsort(y_score)[::-1][:k]
    return float(np.sum(y_true[top_k]) / k) if k > 0 else 0.0


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    top_k = np.argsort(y_score)[::-1][:k]
    total = np.sum(y_true)
    return float(np.sum(y_true[top_k]) / total) if total > 0 else 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Return (AUPR, AUROC) on *loader*."""
    model.eval()
    all_labels, all_probs = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        probs = torch.sigmoid(model(X))
        all_labels.append(y.cpu())
        all_probs.append(probs.cpu())
    labels = torch.cat(all_labels).numpy()
    probs = torch.cat(all_probs).numpy()
    aupr = average_precision_score(labels, probs)
    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = float("nan")
    return aupr, auroc


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    top_k: int = 100,
) -> dict[str, float]:
    """Compute a full suite of classification metrics."""
    pred = (y_score >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    from sklearn.metrics import auc as sk_auc
    return {
        "AUROC": float(sk_auc(fpr, tpr)),
        "AUPR": float(sk_auc(rec, prec)),
        "AP": float(average_precision_score(y_true, y_score)),
        f"Recall@{top_k}": recall_at_k(y_true, y_score, top_k),
        f"Precision@{top_k}": precision_at_k(y_true, y_score, top_k),
        "Accuracy": float(accuracy_score(y_true, pred)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_roc_pr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """Plot ROC and Precision-Recall curves side by side."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    from sklearn.metrics import auc as sk_auc
    roc_auc = sk_auc(fpr, tpr)
    pr_auc = sk_auc(rec, prec)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax1.set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
    ax1.legend()

    ax2.plot(rec, prec, label=f"AP = {pr_auc:.4f}")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    ax2.legend()

    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_losses(
    train_losses: list[float],
    val_losses: list[float],
    save_path: str | Path | None = None,
) -> None:
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, marker="o", markersize=3, label="Train Loss")
    ax.plot(epochs, val_losses, marker="o", markersize=3, label="Val Loss")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_dashboard(
    history: dict,
    save_path: str | Path | None = None,
) -> None:
    """4-panel training dashboard: loss, AUPR, AUROC, learning rate.

    A vertical dashed line marks the best epoch (by val AUPR).
    """
    epochs = range(1, len(history["train_losses"]) + 1)
    best_ep = history.get("best_epoch", None)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Training Dashboard", fontsize=14, fontweight="bold")

    # --- Panel 1: Loss ---
    ax = axes[0, 0]
    ax.plot(epochs, history["train_losses"], label="Train")
    ax.plot(epochs, history["val_losses"], label="Val")
    if best_ep:
        ax.axvline(best_ep, ls="--", color="grey", alpha=0.6, label=f"Best (ep {best_ep})")
    ax.set(xlabel="Epoch", ylabel="Focal Loss", title="Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 2: AUPR ---
    ax = axes[0, 1]
    ax.plot(epochs, history["train_auprs"], label="Train")
    ax.plot(epochs, history["val_auprs"], label="Val")
    if best_ep:
        ax.axvline(best_ep, ls="--", color="grey", alpha=0.6)
    ax.set(xlabel="Epoch", ylabel="AUPR", title="Average Precision (AUPR)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 3: AUROC ---
    ax = axes[1, 0]
    ax.plot(epochs, history["train_aurocs"], label="Train")
    ax.plot(epochs, history["val_aurocs"], label="Val")
    if best_ep:
        ax.axvline(best_ep, ls="--", color="grey", alpha=0.6)
    ax.set(xlabel="Epoch", ylabel="AUROC", title="ROC AUC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Learning rate ---
    ax = axes[1, 1]
    ax.plot(epochs, history["lrs"], color="tab:orange")
    if best_ep:
        ax.axvline(best_ep, ls="--", color="grey", alpha=0.6)
    ax.set(xlabel="Epoch", ylabel="LR", title="Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(history: dict, save_path: str | Path) -> None:
    """Export epoch-level metrics to CSV for post-hoc analysis."""
    n = len(history["train_losses"])
    df = pd.DataFrame({
        "epoch": range(1, n + 1),
        "train_loss": history["train_losses"],
        "val_loss": history["val_losses"],
        "train_aupr": history["train_auprs"],
        "val_aupr": history["val_auprs"],
        "train_auroc": history["train_aurocs"],
        "val_auroc": history["val_aurocs"],
        "lr": history["lrs"],
    })
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    df.to_csv(save_path, index=False)


# ---------------------------------------------------------------------------
# DataLoader construction
# ---------------------------------------------------------------------------

def make_loaders(
    split: dict,
    batch_size: int = 64,
    balanced_sampling: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from a split dict.

    Expects split values for X to be numpy arrays and y to be Series or arrays.
    When *balanced_sampling* is True, a WeightedRandomSampler is used for the
    training loader so each mini-batch sees both classes.
    """
    def _to_tensor(x, dtype):
        if isinstance(x, pd.Series):
            x = x.values
        return torch.tensor(np.asarray(x), dtype=dtype)

    y_train_t = _to_tensor(split["y_train"], torch.float32)
    train_ds = TensorDataset(_to_tensor(split["X_train"], torch.float32), y_train_t)
    val_ds = TensorDataset(_to_tensor(split["X_val"], torch.float32),
                           _to_tensor(split["y_val"], torch.float32))
    test_ds = TensorDataset(_to_tensor(split["X_test"], torch.float32),
                            _to_tensor(split["y_test"], torch.float32))

    if balanced_sampling:
        n_pos = y_train_t.sum().item()
        n_neg = len(y_train_t) - n_pos
        weight_pos = len(y_train_t) / (2 * max(n_pos, 1))
        weight_neg = len(y_train_t) / (2 * max(n_neg, 1))
        sample_weights = torch.where(y_train_t == 1, weight_pos, weight_neg)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    return (
        train_loader,
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model: SLPredictorMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    criterion: nn.Module | None = None,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    epochs: int = 50,
    patience: int = 15,
    scheduler_type: str = "plateau",
    lr_decay_factor: float = 0.5,
    lr_patience: int = 5,
    device: torch.device | None = None,
) -> dict:
    """Train *model* and return training history and best state dict.

    Parameters
    ----------
    weight_decay : float
        L2 regularisation coefficient for AdamW.
    max_grad_norm : float
        Maximum gradient norm for clipping (0 to disable).
    scheduler_type : str
        ``"plateau"`` for ReduceLROnPlateau, ``"cosine"`` for
        CosineAnnealingWarmRestarts.
    patience : int
        Early stopping patience — stop if val AUPR does not improve for
        this many epochs.  Must be **larger** than ``lr_patience`` so the
        scheduler has time to reduce the LR and let the model recover
        before training is terminated.
    lr_decay_factor : float
        Multiplicative factor for ReduceLROnPlateau (default 0.5).
    lr_patience : int
        Number of epochs with no val-loss improvement before the LR is
        reduced.  Should be **smaller** than ``patience``.

    Returns
    -------
    dict with keys:
        best_state    – state dict of best model (by val AUPR)
        train_losses  – per-epoch training losses
        val_losses    – per-epoch validation losses
        train_auprs   – per-epoch training AUPR
        val_auprs     – per-epoch validation AUPR
        train_aurocs  – per-epoch training AUROC
        val_aurocs    – per-epoch validation AUROC
        lrs           – per-epoch learning rate
        best_epoch    – 1-indexed epoch of best val AUPR
        stopped_epoch – epoch at which training stopped (1-indexed)
    """
    if device is None:
        device = get_device()
    if criterion is None:
        criterion = FocalLoss(alpha=0.75, gamma=1.5)

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2,
        )
    else:  # "plateau"
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_decay_factor, patience=lr_patience,
        )

    best_aupr = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    best_state = None

    train_losses, val_losses = [], []
    train_auprs, val_auprs = [], []
    train_aurocs, val_aurocs = [], []
    lrs: list[float] = []

    _log("Training started")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # -- train --
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            running_loss += loss.item() * X.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # -- validate --
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                vl += criterion(model(X), y).item() * X.size(0)
        val_loss = vl / len(val_loader.dataset)
        val_losses.append(val_loss)

        if scheduler_type == "cosine":
            scheduler.step(epoch)
        else:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)

        t_aupr, t_auroc = evaluate(model, train_loader, device)
        v_aupr, v_auroc = evaluate(model, val_loader, device)
        train_auprs.append(t_aupr)
        val_auprs.append(v_aupr)
        train_aurocs.append(t_auroc)
        val_aurocs.append(v_auroc)

        print(
            f"  Epoch {epoch:3d}/{epochs}  "
            f"loss {train_loss:.4f}/{val_loss:.4f}  "
            f"AUPR {t_aupr:.4f}/{v_aupr:.4f}  "
            f"AUROC {t_auroc:.4f}/{v_auroc:.4f}  "
            f"LR {current_lr:.2e}  "
            f"Best: ep {best_epoch}"
        )

        # early stopping on validation AUPR
        if v_aupr > best_aupr:
            best_aupr = v_aupr
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            _log(f"Early stopping at epoch {epoch} (best val AUPR {best_aupr:.4f} at ep {best_epoch})")
            break

    _log(f"Training finished in {time.time() - t0:.1f}s")
    return dict(
        best_state=best_state,
        train_losses=train_losses,
        val_losses=val_losses,
        train_auprs=train_auprs,
        val_auprs=val_auprs,
        train_aurocs=train_aurocs,
        val_aurocs=val_aurocs,
        lrs=lrs,
        best_epoch=best_epoch,
        stopped_epoch=epoch,
    )


# ---------------------------------------------------------------------------
# Full pipeline: train + evaluate on test set
# ---------------------------------------------------------------------------

def run_pipeline(
    split: dict,
    df: pd.DataFrame,
    non_feature_cols: list[str],
    *,
    hidden_sizes: list[int] = [128, 64, 32],
    dropout: float = 0.2,
    batch_size: int = 64,
    balanced_sampling: bool = False,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    epochs: int = 50,
    patience: int = 15,
    scheduler_type: str = "plateau",
    lr_decay_factor: float = 0.5,
    lr_patience: int = 5,
    focal_alpha: float = 0.75,
    focal_gamma: float = 1.5,
    top_k: int = 100,
    output_dir: str | Path = "outputs",
    tag: str = "run",
    save_model: bool = True,
) -> dict:
    """End-to-end: scale, build loaders, train, evaluate, save artefacts."""
    from .data import scale_splits

    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()

    # scale
    split, scaler = scale_splits(split, scaler_path=output_dir / f"{tag}_scaler.pkl")

    # loaders (with optional balanced sampling for imbalanced targets)
    train_loader, val_loader, test_loader = make_loaders(
        split, batch_size, balanced_sampling=balanced_sampling,
    )

    # compute positive-class prior from training labels for output bias init
    y_train = split["y_train"]
    if hasattr(y_train, "values"):
        y_train = y_train.values
    pos_prior = float(np.mean(y_train))

    # model
    input_size = split["X_train"].shape[1]
    model = SLPredictorMLP(input_size, hidden_sizes, dropout, pos_prior=pos_prior)
    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    # train
    history = train_model(
        model, train_loader, val_loader,
        criterion=criterion, lr=lr, weight_decay=weight_decay,
        max_grad_norm=max_grad_norm, epochs=epochs, patience=patience,
        scheduler_type=scheduler_type,
        lr_decay_factor=lr_decay_factor, lr_patience=lr_patience, device=device,
    )
    # restore best
    model.load_state_dict(history["best_state"])
    model.to(device)

    # plots & CSV export
    plot_training_dashboard(history, save_path=output_dir / f"{tag}_dashboard.png")
    plot_losses(history["train_losses"], history["val_losses"],
                save_path=output_dir / f"{tag}_loss.png")
    save_training_curves(history, save_path=output_dir / f"{tag}_training_curves.csv")

    # test evaluation
    model.eval()
    all_labels, all_scores = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            probs = torch.sigmoid(model(X))
            all_labels.append(y)
            all_scores.append(probs.cpu())
    y_true = torch.cat(all_labels).numpy()
    y_score = torch.cat(all_scores).numpy()

    metrics = compute_all_metrics(y_true, y_score, top_k=top_k)
    _log(f"Test metrics: {metrics}")
    plot_roc_pr(y_true, y_score, save_path=output_dir / f"{tag}_roc_pr.png")

    # save predictions
    test_idx = split["y_test"].index if hasattr(split["y_test"], "index") else None
    if test_idx is not None:
        df_test = df.loc[test_idx, non_feature_cols].copy()
        df_test["predict_proba"] = y_score
        df_test.to_csv(output_dir / f"{tag}_predictions.tsv", sep="\t", index=False)

    # save model
    if save_model:
        torch.save(model.state_dict(), output_dir / f"{tag}_model.pt")

    return {**metrics, "history": history, "model": model}
