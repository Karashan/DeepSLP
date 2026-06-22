"""Reproduce the 10-fold CV2 DeepSLP training run.

This is a standalone re-implementation of the procedure recovered from the
saved checkpoints in
    data/interim/ReLU128_f_a075_g15_10folds_pt10/
and the original notebook notebooks/cv2_tuning.ipynb. It does not import from
the project's ``src/`` package.

For each fold i in [1, n_folds]:
  * split data by holding out whole query genes (CV2), seed = i + seed_offset
  * standardise features with StandardScaler (fit on train)
  * train NeuralNetwork(256->128->64->32->1) with FocalLoss(a=0.75, g=1.5)
  * Adam(lr=1e-2), ReduceLROnPlateau(factor=0.1, patience=5)
  * early stopping (patience=10) on best validation AUPR; restore best weights
  * save the whole model (.pth), the scaler (.joblib), test predictions (.tsv)

Example:
    python reproduce_cv2/train.py \
        --input-dir data/input/GIV_24Q4/ReLU128_5L/ \
        --output-dir reproduce_cv2/output/

Run ``python reproduce_cv2/train.py --help`` for all options.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config import TrainConfig
from data import load_input, split_cv2_query_holdout
from metrics import compute_metrics, safe_auroc
from model import FocalLoss, NeuralNetwork


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _eval_aupr(model, loader, device) -> float:
    model.eval()
    labels_all, probs_all = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs.to(device))
            probs_all.append(torch.sigmoid(logits).cpu())
            labels_all.append(labels.float().cpu())
    y = torch.cat(labels_all).numpy()
    p = torch.cat(probs_all).numpy()
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, p))


def train_one_fold(split, cfg: TrainConfig, device: str):
    X_train = split["X_train"]
    X_val = split["X_val"]
    X_test = split["X_test"]
    y_train = split["y_train"]
    y_val = split["y_val"]
    y_test = split["y_test"]

    input_size = X_train.shape[1]
    model = NeuralNetwork(
        input_size,
        cfg.hidden_size1,
        cfg.hidden_size2,
        cfg.hidden_size3,
        cfg.output_size,
        dropout_p=cfg.dropout_p,
    ).to(device)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    def loaders(Xs, y, shuffle, drop_last=False):
        ds = TensorDataset(
            torch.tensor(Xs, dtype=torch.float32),
            torch.tensor(y.values, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, drop_last=drop_last)

    # drop_last on train avoids a size-1 final batch breaking BatchNorm1d
    train_loader = loaders(X_train_s, y_train, True, drop_last=True)
    val_loader = loaders(X_val_s, y_val, False)
    test_loader = loaders(X_test_s, y_test, False)

    criterion = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience
    )

    best_aupr = 0.0
    best_state = None
    epochs_no_improve = 0

    log("Training started")
    start = time.time()
    for epoch in range(cfg.num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float()
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        val_aupr = _eval_aupr(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}/{cfg.num_epochs}: "
            f"Train Loss={train_loss:.4f} Val Loss={val_loss:.4f} Val AUPR={val_aupr:.4f}"
        )

        if val_aupr > best_aupr:
            best_aupr = val_aupr
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= cfg.early_stop_patience:
            print("Early stopping triggered.")
            break

    log(f"Total training time: {time.time() - start:.2f}s")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test predictions
    model.eval()
    scores, labels_all = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            logits = model(inputs.to(device))
            scores.append(torch.sigmoid(logits).cpu())
            labels_all.append(labels.float().cpu())
    test_scores = torch.cat(scores).numpy()
    test_labels = torch.cat(labels_all).numpy()

    return model, scaler, test_scores, test_labels, y_test.index


def main():
    cfg = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default=cfg.input_dir)
    p.add_argument("--output-dir", default=cfg.output_dir)
    p.add_argument("--prefix", default=cfg.prefix)
    p.add_argument("--n-folds", type=int, default=cfg.n_folds)
    p.add_argument("--start-fold", type=int, default=1)
    p.add_argument("--iter-num", type=int, default=cfg.input_glob_iter)
    p.add_argument("--label-col", default=cfg.label_col)
    p.add_argument("--epochs", type=int, default=cfg.num_epochs)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--topk", type=int, default=cfg.topk)
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto-detect if unset)")
    args = p.parse_args()

    cfg.input_dir = args.input_dir
    cfg.output_dir = args.output_dir
    cfg.prefix = args.prefix
    cfg.n_folds = args.n_folds
    cfg.input_glob_iter = args.iter_num
    cfg.label_col = args.label_col
    cfg.num_epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.lr
    cfg.topk = args.topk

    device = args.device or pick_device()
    log(f"Using {device} device")
    os.makedirs(cfg.output_dir, exist_ok=True)

    log(f"Loading input from {cfg.input_dir}")
    df = load_input(
        cfg.input_dir,
        iter_num=cfg.input_glob_iter,
        tail=cfg.file_tail,
        query_col=cfg.query_col,
        lib_col=cfg.lib_col,
        fdr_col=cfg.fdr_col,
    )
    log(f"Input prepared: {df.shape[0]} pairs, {df[cfg.query_col].nunique()} unique queries")

    perf_rows = []
    for fold in range(args.start_fold, args.start_fold + cfg.n_folds):
        rand_seed = fold + cfg.seed_offset
        log(f"===== Fold {fold} (seed={rand_seed}) =====")
        split = split_cv2_query_holdout(
            df,
            label_col=cfg.label_col,
            non_feature_cols=cfg.non_feature_cols,
            query_col=cfg.query_col,
            test_ratio=cfg.test_ratio,
            val_ratio=cfg.val_ratio,
            rand_seed=rand_seed,
        )
        print(
            f"Train={split['X_train'].shape[0]} "
            f"Val={split['X_val'].shape[0]} "
            f"Test={split['X_test'].shape[0]} (features={split['X_train'].shape[1]})"
        )

        model, scaler, scores, labels, test_idx = train_one_fold(split, cfg, device)

        # Save whole model + scaler (matches original torch.save(model, ...))
        model_path = os.path.join(cfg.output_dir, f"{cfg.prefix}_{fold}.pth")
        scaler_path = os.path.join(cfg.output_dir, f"CV2_811_seed{fold}.joblib")
        torch.save(model, model_path)
        joblib.dump(scaler, scaler_path)

        # Save test predictions table
        df_test = df.loc[test_idx, cfg.non_feature_cols].copy()
        df_test["predict_proba"] = scores
        tsv_path = os.path.join(cfg.output_dir, f"{cfg.prefix}_{fold}.tsv")
        df_test.to_csv(tsv_path, sep="\t")

        m = compute_metrics(labels, scores, k=cfg.topk)
        m["auroc_sklearn"] = safe_auroc(labels, scores)
        perf_rows.append(m)
        print(
            f"Fold {fold}: ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} "
            f"AP={m['average_precision']:.4f} "
            f"P@{cfg.topk}={m[f'precision_at_{cfg.topk}']:.4f} "
            f"R@{cfg.topk}={m[f'recall_at_{cfg.topk}']:.4f}"
        )
        print(f"Saved: {model_path}\n       {scaler_path}\n       {tsv_path}")

    perf = pd.DataFrame(perf_rows)
    perf_path = os.path.join(cfg.output_dir, f"{cfg.prefix}_performance_stats_{cfg.n_folds}folds.tsv")
    perf.to_csv(perf_path, sep="\t", index=False)
    log(f"Saved performance summary: {perf_path}")
    print("\nMean across folds:\n", perf.mean(numeric_only=True))
    print("\nStd across folds:\n", perf.std(numeric_only=True))


if __name__ == "__main__":
    main()
