"""One-time, single-split reproduction of the CV2 DeepSLP training run.

Goal
----
Train ONE model from scratch using the same procedure / hyperparameters as the
original notebook (notebooks/cv2_tuning.ipynb) and a single CV2 query-holdout
split, then:
  1. report held-out *test* performance (from the internal CV2 split), and
  2. evaluate the trained model on the *external* held-out dataset
     (data/input/GIV_24Q4_heldout/) and report AUROC / PR-AUC / P@100 / R@100.

This uses the same recovered configuration as the saved checkpoints in
    data/interim/ReLU128_f_a075_g15_10folds_pt10/
i.e. fold-1 split (rand_seed = 1 + 1842 = 1843), NeuralNetwork 256->128->64->32->1,
FocalLoss(alpha=0.75, gamma=1.5), Adam(lr=1e-2), ReduceLROnPlateau(0.1, 5),
early stopping (patience=10) on best validation AUPR.

Nothing existing is overwritten: every artifact is written into a fresh,
timestamped run directory under data/interim/.

Usage:
    python reproduce_cv2/run_single_training.py            # defaults below
    python reproduce_cv2/run_single_training.py --epochs 50 --seed 1843
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch

# Make the package importable regardless of the current working directory.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from config import TrainConfig, REPO_ROOT  # noqa: E402
from data import load_input, split_cv2_query_holdout, feature_columns  # noqa: E402
from metrics import compute_metrics, safe_auroc  # noqa: E402
from train import train_one_fold, pick_device, log  # noqa: E402


def evaluate_external(
    model,
    scaler,
    cfg: TrainConfig,
    heldout_dir: str,
    heldout_iter: int,
    device: str,
    label_cols,
    topk: int,
    batch_size: int = 8192,
):
    """Score the external held-out dataset and compute metrics per label column."""
    log(f"Loading external held-out data from {heldout_dir}")
    df_ext = load_input(
        heldout_dir,
        iter_num=heldout_iter,
        tail=cfg.file_tail,
        query_col=cfg.query_col,
        lib_col=cfg.lib_col,
        fdr_col=cfg.fdr_col,
    )
    log(f"External prepared: {df_ext.shape[0]} pairs, "
        f"{df_ext[cfg.query_col].nunique()} unique queries")

    feat_cols = feature_columns(df_ext, cfg.non_feature_cols)
    X = df_ext[feat_cols]
    valid = X.notna().all(axis=1)
    n_drop = int((~valid).sum())
    if n_drop:
        log(f"Dropping {n_drop} external rows with missing features.")
    df_ext = df_ext.loc[valid].reset_index(drop=True)
    X = scaler.transform(df_ext[feat_cols].values)
    X_t = torch.tensor(X, dtype=torch.float32)

    probs = np.empty(X_t.shape[0], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(0, X_t.shape[0], batch_size):
            batch = X_t[s : s + batch_size].to(device)
            logits = model(batch).squeeze(-1)
            probs[s : s + batch.shape[0]] = torch.sigmoid(logits).cpu().numpy()

    df_ext["predict_proba"] = probs

    ext_metrics = {"n_pairs": int(df_ext.shape[0])}
    for lc in label_cols:
        if lc not in df_ext.columns:
            continue
        y = df_ext[lc].values
        if pd.isna(y).any() or len(np.unique(y)) < 2:
            ext_metrics[lc] = {"note": "label missing/constant; skipped"}
            continue
        m = compute_metrics(y, probs, k=topk)
        m["auroc_sklearn"] = safe_auroc(y, probs)
        m["n_positives"] = int(np.sum(y))
        ext_metrics[lc] = m

    return df_ext, ext_metrics


def main():
    cfg = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default=cfg.input_dir)
    p.add_argument("--iter-num", type=int, default=cfg.input_glob_iter)
    p.add_argument("--heldout-dir", default=os.path.join(REPO_ROOT, "data", "input", "GIV_24Q4_heldout") + os.sep)
    p.add_argument("--heldout-iter", type=int, default=11, help="number of heldout shards (0..N-1)")
    p.add_argument("--seed", type=int, default=1843, help="CV2 split seed (original fold-1 = 1843)")
    p.add_argument("--epochs", type=int, default=cfg.num_epochs)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--label-col", default=cfg.label_col)
    p.add_argument("--topk", type=int, default=cfg.topk)
    p.add_argument("--device", default=None)
    p.add_argument("--run-dir", default=None, help="output dir (default: auto-timestamped under data/interim)")
    args = p.parse_args()

    # Apply CLI overrides to the config
    cfg.input_dir = args.input_dir
    cfg.input_glob_iter = args.iter_num
    cfg.num_epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.lr
    cfg.label_col = args.label_col
    cfg.topk = args.topk

    # Fresh, non-destructive run directory
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or os.path.join(
        REPO_ROOT, "data", "interim", f"repro_single_seed{args.seed}_{stamp}"
    )
    os.makedirs(run_dir, exist_ok=True)
    cfg.output_dir = run_dir

    device = args.device or pick_device()
    log(f"=== One-time CV2 reproduction run ===")
    log(f"Run dir : {run_dir}")
    log(f"Device  : {device}")
    log(f"Seed    : {args.seed} | epochs: {cfg.num_epochs} | batch: {cfg.batch_size} | lr: {cfg.learning_rate}")
    log(f"Label   : {cfg.label_col}")

    # Persist the resolved configuration
    with open(os.path.join(run_dir, "run_config.json"), "w") as fh:
        json.dump(
            {
                "seed": args.seed,
                "input_dir": cfg.input_dir,
                "iter_num": cfg.input_glob_iter,
                "heldout_dir": args.heldout_dir,
                "heldout_iter": args.heldout_iter,
                "label_col": cfg.label_col,
                "epochs": cfg.num_epochs,
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.learning_rate,
                "focal_alpha": cfg.focal_alpha,
                "focal_gamma": cfg.focal_gamma,
                "scheduler_factor": cfg.scheduler_factor,
                "scheduler_patience": cfg.scheduler_patience,
                "early_stop_patience": cfg.early_stop_patience,
                "dropout_p": cfg.dropout_p,
                "hidden_sizes": [cfg.hidden_size1, cfg.hidden_size2, cfg.hidden_size3],
                "topk": cfg.topk,
            },
            fh,
            indent=2,
        )

    # ---- 1. Load + process training data (saved as interim artifact) ----
    log(f"Loading training input from {cfg.input_dir}")
    df = load_input(
        cfg.input_dir,
        iter_num=cfg.input_glob_iter,
        tail=cfg.file_tail,
        query_col=cfg.query_col,
        lib_col=cfg.lib_col,
        fdr_col=cfg.fdr_col,
    )
    log(f"Processed training data: {df.shape[0]} pairs, "
        f"{df[cfg.query_col].nunique()} unique queries")
    joblib.dump(df, os.path.join(run_dir, "processed_train_data.joblib"), compress=3)

    # ---- 2. Single CV2 query-holdout split ----
    split = split_cv2_query_holdout(
        df,
        label_col=cfg.label_col,
        non_feature_cols=cfg.non_feature_cols,
        query_col=cfg.query_col,
        test_ratio=cfg.test_ratio,
        val_ratio=cfg.val_ratio,
        rand_seed=args.seed,
    )
    log(
        f"Split -> Train={split['X_train'].shape[0]} "
        f"Val={split['X_val'].shape[0]} Test={split['X_test'].shape[0]} "
        f"(features={split['X_train'].shape[1]})"
    )
    log(
        f"Positives -> Train={int(split['y_train'].sum())} "
        f"Val={int(split['y_val'].sum())} Test={int(split['y_test'].sum())}"
    )
    joblib.dump(
        {
            "train_idx": np.asarray(split["X_train"].index),
            "val_idx": np.asarray(split["X_val"].index),
            "test_idx": np.asarray(split["X_test"].index),
            "seed": args.seed,
        },
        os.path.join(run_dir, "split_indices.joblib"),
        compress=3,
    )

    # ---- 3. Train ----
    model, scaler, test_scores, test_labels, test_idx = train_one_fold(split, cfg, device)

    # ---- 4. Save model + scaler + test predictions ----
    model_path = os.path.join(run_dir, "model.pth")
    scaler_path = os.path.join(run_dir, "scaler.joblib")
    torch.save(model, model_path)
    joblib.dump(scaler, scaler_path)

    df_test = df.loc[test_idx, cfg.non_feature_cols].copy()
    df_test["predict_proba"] = test_scores
    df_test.to_csv(os.path.join(run_dir, "test_predictions.tsv"), sep="\t")

    test_metrics = compute_metrics(test_labels, test_scores, k=cfg.topk)
    test_metrics["auroc_sklearn"] = safe_auroc(test_labels, test_scores)
    test_metrics["n_test_pairs"] = int(len(test_labels))
    test_metrics["n_test_positives"] = int(np.sum(test_labels))
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as fh:
        json.dump(test_metrics, fh, indent=2)

    log("===== INTERNAL TEST (CV2 held-out queries) =====")
    log(
        f"ROC-AUC={test_metrics['roc_auc']:.4f} | PR-AUC={test_metrics['pr_auc']:.4f} | "
        f"AP={test_metrics['average_precision']:.4f} | "
        f"P@{cfg.topk}={test_metrics[f'precision_at_{cfg.topk}']:.4f} | "
        f"R@{cfg.topk}={test_metrics[f'recall_at_{cfg.topk}']:.4f} | "
        f"pos={test_metrics['n_test_positives']}/{test_metrics['n_test_pairs']}"
    )

    # ---- 5. External held-out validation ----
    label_cols = [cfg.label_col]
    if "GI_standard_Type2" not in label_cols:
        label_cols.append("GI_standard_Type2")

    df_ext, ext_metrics = evaluate_external(
        model, scaler, cfg, args.heldout_dir, args.heldout_iter, device, label_cols, cfg.topk
    )
    keep_cols = [c for c in cfg.non_feature_cols if c in df_ext.columns] + ["predict_proba"]
    df_ext[keep_cols].to_csv(os.path.join(run_dir, "heldout_predictions.tsv"), sep="\t", index=False)
    with open(os.path.join(run_dir, "heldout_metrics.json"), "w") as fh:
        json.dump(ext_metrics, fh, indent=2)

    log("===== EXTERNAL HELD-OUT VALIDATION =====")
    log(f"n_pairs={ext_metrics['n_pairs']}")
    for lc in label_cols:
        m = ext_metrics.get(lc, {})
        if "roc_auc" in m:
            log(
                f"[{lc}] ROC-AUC={m['roc_auc']:.4f} | PR-AUC={m['pr_auc']:.4f} | "
                f"AP={m['average_precision']:.4f} | "
                f"P@{cfg.topk}={m[f'precision_at_{cfg.topk}']:.4f} | "
                f"R@{cfg.topk}={m[f'recall_at_{cfg.topk}']:.4f} | "
                f"pos={m['n_positives']}"
            )
        else:
            log(f"[{lc}] {m.get('note', 'skipped')}")

    log("===== DONE =====")
    log(f"All artifacts saved under: {run_dir}")


if __name__ == "__main__":
    main()
