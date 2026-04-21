"""Training + evaluation loop (FocalLoss, ReduceLROnPlateau, early-stop on val AUPR)."""
from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .config import HyperParams
from .metrics import (
    ensure_dir,
    evaluate_metrics,
    log_message,
    perf_columns,
    plot_curves,
    plot_roc_pr,
)
from .model import FocalLoss, NeuralNetwork, pick_device


def train_validate_test_nn(
    splits: Dict[str, pd.DataFrame],
    df: pd.DataFrame,
    non_feature_cols: List[str],
    output_dir: str,
    plt_name: str,
    table_name: str,
    hp: HyperParams = None,
    model_name: str = "model.pth",
    scaler_name: str = "scaler.joblib",
    save_model: bool = True,
    plot_losses: bool = True,
    topk: int = 100,
) -> Tuple[torch.nn.Module, pd.DataFrame, List[float]]:
    """Train, validate, and test one NN model. Returns (model, df_test, perf_metrics)."""
    hp = hp or HyperParams()
    ensure_dir(output_dir)
    device = pick_device()
    print(f"Using {device} device")

    X_train, X_val, X_test = splits["X_train"], splits["X_val"], splits["X_test"]
    y_train, y_val, y_test = splits["y_train"], splits["y_val"], splits["y_test"]
    test_idx = y_test.index

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    if scaler_name:
        joblib.dump(scaler, os.path.join(output_dir, scaler_name))

    def _loader(X, y, shuffle, drop_last=False):
        return DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32),
                          torch.tensor(y.values, dtype=torch.long)),
            batch_size=hp.batch_size, shuffle=shuffle, drop_last=drop_last)

    # drop_last=True on training loader to avoid BatchNorm failure when the
    # final batch has size 1 (raises "Expected more than 1 value per channel").
    train_loader = _loader(X_train_s, y_train, shuffle=True, drop_last=True)
    val_loader   = _loader(X_val_s,   y_val,   shuffle=False)
    test_loader  = _loader(X_test_s,  y_test,  shuffle=False)

    model = NeuralNetwork(
        input_size=X_train_s.shape[1],
        hidden_size1=hp.hidden_size1, hidden_size2=hp.hidden_size2,
        hidden_size3=hp.hidden_size3, output_size=1, dropout=hp.dropout,
    ).to(device)
    criterion = FocalLoss(alpha=hp.focal_alpha, gamma=hp.focal_gamma)
    optimizer = optim.Adam(model.parameters(), lr=hp.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=hp.decay_factor,
        patience=hp.scheduler_patience, verbose=False)

    best_aupr = 0.0
    epochs_no_improve = 0
    best_state = None
    train_losses, val_losses, train_auprs, val_auprs = [], [], [], []

    log_message("Training started")
    t0 = time.time()
    for epoch in range(hp.num_epochs):
        model.train()
        running = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device); labels = labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            running += loss.item() * inputs.size(0)
        train_loss = running / len(train_loader.dataset)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device); labels = labels.to(device).float()
                outputs = model(inputs).squeeze(-1)
                vl += criterion(outputs, labels).item() * inputs.size(0)
        val_loss = vl / len(val_loader.dataset)
        scheduler.step(val_loss)

        tr_aupr, tr_auroc = evaluate_metrics(model, train_loader, device)
        va_aupr, va_auroc = evaluate_metrics(model, val_loader,   device)
        train_losses.append(train_loss); val_losses.append(val_loss)
        train_auprs.append(tr_aupr);     val_auprs.append(va_aupr)
        print(f"Ep {epoch+1}/{hp.num_epochs}: TrL={train_loss:.4f} VaL={val_loss:.4f} "
              f"| TrAUPR={tr_aupr:.4f} VaAUPR={va_aupr:.4f}")

        if va_aupr > best_aupr:
            best_aupr = va_aupr
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= hp.patience:
            print("Early stopping.")
            break

    log_message(f"Total training time: {time.time() - t0:.2f}s")

    if plot_losses:
        plot_curves(train_losses, val_losses, "Training Loss", "Validation Loss",
                    plt_path=os.path.join(output_dir, plt_name.replace(".", "_train_val_loss.")))
        plot_curves(train_auprs, val_auprs, "Training AUPR", "Validation AUPR",
                    plt_path=os.path.join(output_dir, plt_name.replace(".", "_train_val_aupr.")))

    if best_state is not None:
        model.load_state_dict(best_state)
    if save_model:
        torch.save(model, os.path.join(output_dir, model_name))

    model.eval()
    all_labels, all_scores = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device); labels = labels.to(device).float()
            outputs = model(inputs).squeeze(-1)
            probs = torch.sigmoid(outputs)
            all_scores.append(probs.cpu()); all_labels.append(labels.cpu())
    all_scores = torch.cat(all_scores).numpy()
    all_labels = torch.cat(all_labels).numpy()

    perf = plot_roc_pr(all_labels, all_scores,
                       plt_path=os.path.join(output_dir, plt_name), K=topk)
    df_test = df.loc[test_idx, non_feature_cols].copy()
    df_test["predict_proba"] = all_scores
    df_test.to_csv(os.path.join(output_dir, table_name), sep="\t")
    # Tag output columns for easier downstream reading
    _ = perf_columns(topk)  # just to assert name consistency
    return model, df_test, perf
