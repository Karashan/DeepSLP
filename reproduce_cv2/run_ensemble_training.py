"""Full 10-fold CV2 training + ENSEMBLE external validation.

Trains 10 independent models (one per CV2 query-holdout split, seeds 1843..1852,
matching the original fold seeds = i + 1842), then evaluates the mean-probability
ENSEMBLE on the external held-out dataset (data/input/GIV_24Q4_heldout/).

This mirrors how the original notebook produced its (much better) external
numbers: the held-out predictions there were the average of all 10 fold models.

All artifacts go into a fresh, timestamped run directory under data/interim/ so
nothing existing is overwritten:
    fold{1..10}_model.pth, fold{1..10}_scaler.joblib,
    fold{1..10}_test_predictions.tsv
    internal_test_metrics_10folds.tsv
    heldout_ensemble_predictions.tsv
    heldout_ensemble_metrics.json
    heldout_perfold_metrics.json
    run_config.json
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

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from config import TrainConfig, REPO_ROOT  # noqa: E402
from data import load_input, split_cv2_query_holdout, feature_columns  # noqa: E402
from metrics import compute_metrics, safe_auroc  # noqa: E402
from train import train_one_fold, pick_device, log  # noqa: E402
import model as _model_module  # noqa: E402

# Allow un-pickling whole-model checkpoints regardless of how they were saved.
sys.modules["__main__"].NeuralNetwork = _model_module.NeuralNetwork
sys.modules["__main__"].FocalLoss = _model_module.FocalLoss


def _internal_metrics_from_predictions(pred_path, label_col, topk):
    """Recompute internal test metrics from a saved fold predictions TSV."""
    t = pd.read_csv(pred_path, sep="\t", low_memory=False)
    y = t[label_col].values
    p = t["predict_proba"].values
    mask = ~pd.isna(y) & ~pd.isna(p)
    y = y[mask].astype(int)
    p = p[mask]
    m = compute_metrics(y, p, k=topk)
    m["auroc_sklearn"] = safe_auroc(y, p)
    m["n_test_pairs"] = int(len(y))
    m["n_test_positives"] = int(y.sum())
    return m


def score(model, scaler, X_values, device, batch_size=8192):
    Xt = torch.tensor(scaler.transform(X_values), dtype=torch.float32)
    model.eval().to(device)
    out = np.empty(Xt.shape[0], dtype=np.float32)
    with torch.no_grad():
        for s in range(0, Xt.shape[0], batch_size):
            b = Xt[s : s + batch_size].to(device)
            out[s : s + b.shape[0]] = torch.sigmoid(model(b).squeeze(-1)).cpu().numpy()
    return out


def main():
    cfg = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default=cfg.input_dir)
    p.add_argument("--iter-num", type=int, default=cfg.input_glob_iter)
    p.add_argument("--heldout-dir", default=os.path.join(REPO_ROOT, "data", "input", "GIV_24Q4_heldout") + os.sep)
    p.add_argument("--heldout-iter", type=int, default=11)
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--seed-offset", type=int, default=cfg.seed_offset)
    p.add_argument("--epochs", type=int, default=cfg.num_epochs)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--label-col", default=cfg.label_col)
    p.add_argument("--topk", type=int, default=cfg.topk)
    p.add_argument("--device", default=None)
    p.add_argument("--run-dir", default=None)
    args = p.parse_args()

    cfg.input_dir = args.input_dir
    cfg.input_glob_iter = args.iter_num
    cfg.num_epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.lr
    cfg.label_col = args.label_col
    cfg.topk = args.topk
    cfg.seed_offset = args.seed_offset

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or os.path.join(REPO_ROOT, "data", "interim", f"repro_ensemble_{args.n_folds}folds_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    cfg.output_dir = run_dir

    device = args.device or pick_device()
    log("=== 10-fold CV2 ensemble reproduction run ===")
    log(f"Run dir : {run_dir}")
    log(f"Device  : {device} | folds: {args.n_folds} | epochs: {cfg.num_epochs} | "
        f"batch: {cfg.batch_size} | lr: {cfg.learning_rate} | label: {cfg.label_col}")

    with open(os.path.join(run_dir, "run_config.json"), "w") as fh:
        json.dump(
            {
                "n_folds": args.n_folds,
                "seeds": [f + cfg.seed_offset for f in range(1, args.n_folds + 1)],
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

    # ---- Load + process training data once ----
    log(f"Loading training input from {cfg.input_dir}")
    df = load_input(
        cfg.input_dir, iter_num=cfg.input_glob_iter, tail=cfg.file_tail,
        query_col=cfg.query_col, lib_col=cfg.lib_col, fdr_col=cfg.fdr_col,
    )
    log(f"Processed training data: {df.shape[0]} pairs, {df[cfg.query_col].nunique()} unique queries")

    # ---- Train each fold (resume completed folds if artifacts exist) ----
    models = []   # list of (model, scaler)
    internal_rows = []
    for fold in range(1, args.n_folds + 1):
        seed = fold + cfg.seed_offset
        model_path = os.path.join(run_dir, f"fold{fold}_model.pth")
        scaler_path = os.path.join(run_dir, f"fold{fold}_scaler.joblib")
        pred_path = os.path.join(run_dir, f"fold{fold}_test_predictions.tsv")

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            log(f"===== Fold {fold}/{args.n_folds} (seed={seed}) — RESUMED from disk =====")
            model = torch.load(model_path, map_location=device, weights_only=False)
            scaler = joblib.load(scaler_path)
            if os.path.exists(pred_path):
                m = _internal_metrics_from_predictions(pred_path, cfg.label_col, cfg.topk)
                m["fold"] = fold
                m["seed"] = seed
                internal_rows.append(m)
                log(f"Fold {fold} internal test (loaded): ROC={m['roc_auc']:.4f} "
                    f"AP={m['average_precision']:.4f} P@{cfg.topk}={m[f'precision_at_{cfg.topk}']:.4f}")
            models.append((model, scaler))
            continue

        log(f"===== Fold {fold}/{args.n_folds} (seed={seed}) =====")
        split = split_cv2_query_holdout(
            df, label_col=cfg.label_col, non_feature_cols=cfg.non_feature_cols,
            query_col=cfg.query_col, test_ratio=cfg.test_ratio, val_ratio=cfg.val_ratio,
            rand_seed=seed,
        )
        log(f"Split -> Train={split['X_train'].shape[0]} Val={split['X_val'].shape[0]} "
            f"Test={split['X_test'].shape[0]} | test_pos={int(split['y_test'].sum())}")

        model, scaler, test_scores, test_labels, test_idx = train_one_fold(split, cfg, device)

        torch.save(model, model_path)
        joblib.dump(scaler, scaler_path)
        df_test = df.loc[test_idx, cfg.non_feature_cols].copy()
        df_test["predict_proba"] = test_scores
        df_test.to_csv(pred_path, sep="\t")

        m = compute_metrics(test_labels, test_scores, k=cfg.topk)
        m["auroc_sklearn"] = safe_auroc(test_labels, test_scores)
        m["fold"] = fold
        m["seed"] = seed
        m["n_test_pairs"] = int(len(test_labels))
        m["n_test_positives"] = int(np.sum(test_labels))
        internal_rows.append(m)
        log(f"Fold {fold} internal test: ROC={m['roc_auc']:.4f} AP={m['average_precision']:.4f} "
            f"P@{cfg.topk}={m[f'precision_at_{cfg.topk}']:.4f}")

        models.append((model, scaler))

    perf = pd.DataFrame(internal_rows)
    perf.to_csv(os.path.join(run_dir, f"internal_test_metrics_{args.n_folds}folds.tsv"), sep="\t", index=False)
    log("===== INTERNAL TEST (per-fold, CV2 held-out queries) =====")
    log(f"Mean ROC={perf['roc_auc'].mean():.4f}±{perf['roc_auc'].std():.4f} | "
        f"Mean AP={perf['average_precision'].mean():.4f} | "
        f"Mean P@{cfg.topk}={perf[f'precision_at_{cfg.topk}'].mean():.4f}")

    # ---- External held-out ENSEMBLE evaluation ----
    log(f"Loading external held-out data from {args.heldout_dir}")
    df_ext = load_input(
        args.heldout_dir, iter_num=args.heldout_iter, tail=cfg.file_tail,
        query_col=cfg.query_col, lib_col=cfg.lib_col, fdr_col=cfg.fdr_col,
    )
    feat_cols = feature_columns(df_ext, cfg.non_feature_cols)
    valid = df_ext[feat_cols].notna().all(axis=1)
    n_drop = int((~valid).sum())
    if n_drop:
        log(f"Dropping {n_drop} external rows with missing features.")
    df_ext = df_ext.loc[valid].reset_index(drop=True)
    Xv = df_ext[feat_cols].values
    log(f"External prepared: {df_ext.shape[0]} pairs, {df_ext[cfg.query_col].nunique()} unique queries")

    label_cols = [cfg.label_col]
    if "GI_standard_Type2" not in label_cols:
        label_cols.append("GI_standard_Type2")

    # Per-fold predictions + accumulate ensemble
    ens = np.zeros(df_ext.shape[0], dtype=np.float64)
    perfold_ext = {}
    for fold, (model, scaler) in enumerate(models, start=1):
        p_fold = score(model, scaler, Xv, device)
        ens += p_fold
        fold_metrics = {}
        for lc in label_cols:
            y = df_ext[lc].values
            if pd.isna(y).any() or len(np.unique(y)) < 2:
                continue
            mm = compute_metrics(y, p_fold, k=cfg.topk)
            mm["auroc_sklearn"] = safe_auroc(y, p_fold)
            fold_metrics[lc] = mm
        perfold_ext[f"fold{fold}"] = fold_metrics
    ens /= len(models)
    df_ext["predict_proba"] = ens

    ext_metrics = {"n_pairs": int(df_ext.shape[0]), "n_models": len(models)}
    for lc in label_cols:
        y = df_ext[lc].values
        if pd.isna(y).any() or len(np.unique(y)) < 2:
            ext_metrics[lc] = {"note": "label missing/constant; skipped"}
            continue
        mm = compute_metrics(y, ens, k=cfg.topk)
        mm["auroc_sklearn"] = safe_auroc(y, ens)
        mm["n_positives"] = int(np.sum(y))
        ext_metrics[lc] = mm

    keep = [c for c in cfg.non_feature_cols if c in df_ext.columns] + ["predict_proba"]
    df_ext[keep].to_csv(os.path.join(run_dir, "heldout_ensemble_predictions.tsv"), sep="\t", index=False)
    with open(os.path.join(run_dir, "heldout_ensemble_metrics.json"), "w") as fh:
        json.dump(ext_metrics, fh, indent=2)
    with open(os.path.join(run_dir, "heldout_perfold_metrics.json"), "w") as fh:
        json.dump(perfold_ext, fh, indent=2)

    log("===== EXTERNAL HELD-OUT (10-model ENSEMBLE) =====")
    log(f"n_pairs={ext_metrics['n_pairs']} | n_models={ext_metrics['n_models']}")
    for lc in label_cols:
        m = ext_metrics.get(lc, {})
        if "roc_auc" in m:
            log(f"[{lc}] ROC-AUC={m['roc_auc']:.4f} | PR-AUC={m['pr_auc']:.4f} | "
                f"AP={m['average_precision']:.4f} | P@{cfg.topk}={m[f'precision_at_{cfg.topk}']:.4f} | "
                f"R@{cfg.topk}={m[f'recall_at_{cfg.topk}']:.4f} | pos={m['n_positives']}")
        else:
            log(f"[{lc}] {m.get('note', 'skipped')}")

    log("===== DONE =====")
    log(f"All artifacts saved under: {run_dir}")


if __name__ == "__main__":
    main()
