"""Full 10-seed CV2 DeepSLP assessment (single train/val/test split per seed).

Efficient multi-seed version of run_single_training.py: the (large) training
and external held-out datasets are loaded and preprocessed ONCE, then we loop
over the 10 original seeds (1843..1852 = fold + 1842). For each seed we:
  * make a CV2 query-holdout split,
  * train NeuralNetwork(256->128->64->32->1) with the recovered config,
  * record internal CV2 *test* metrics,
  * score the external held-out set with that fold's model+scaler and record
    external metrics (GI_stringent_Type2 and GI_standard_Type2).

Finally we aggregate mean +/- std across seeds and write a summary.

Nothing is overwritten: all artifacts go to a fresh timestamped run directory
under data/interim/, with one subfolder per seed.

Usage:
    python reproduce_cv2/run_10seeds.py            # seeds 1843..1852, 50 epochs
    python reproduce_cv2/run_10seeds.py --n-seeds 10 --start-seed 1843 --epochs 50
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


def score_external(model, scaler, X_ext_raw, device, batch_size=8192):
    """Scale + predict probabilities for a preloaded external feature matrix."""
    X = scaler.transform(X_ext_raw)
    X_t = torch.tensor(X, dtype=torch.float32)
    probs = np.empty(X_t.shape[0], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(0, X_t.shape[0], batch_size):
            batch = X_t[s : s + batch_size].to(device)
            logits = model(batch).squeeze(-1)
            probs[s : s + batch.shape[0]] = torch.sigmoid(logits).cpu().numpy()
    return probs


def main():
    cfg = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default=cfg.input_dir)
    p.add_argument("--iter-num", type=int, default=cfg.input_glob_iter)
    p.add_argument("--heldout-dir", default=os.path.join(REPO_ROOT, "data", "input", "GIV_24Q4_heldout") + os.sep)
    p.add_argument("--heldout-iter", type=int, default=11)
    p.add_argument("--start-seed", type=int, default=1843, help="first CV2 seed (fold 1 = 1843)")
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=cfg.num_epochs)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--label-col", default=cfg.label_col)
    p.add_argument("--topk", type=int, default=cfg.topk)
    p.add_argument("--device", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--save-heldout-preds", action="store_true",
                   help="also save the (large) per-seed external prediction tables")
    args = p.parse_args()

    cfg.input_dir = args.input_dir
    cfg.input_glob_iter = args.iter_num
    cfg.num_epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.lr
    cfg.label_col = args.label_col
    cfg.topk = args.topk

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or os.path.join(REPO_ROOT, "data", "interim", f"repro_10seeds_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    device = args.device or pick_device()
    ext_label_cols = [cfg.label_col]
    if "GI_standard_Type2" not in ext_label_cols:
        ext_label_cols.append("GI_standard_Type2")

    log("=== Full 10-seed CV2 reproduction assessment ===")
    log(f"Run dir : {run_dir}")
    log(f"Device  : {device}")
    log(f"Seeds   : {args.start_seed}..{args.start_seed + args.n_seeds - 1}")
    log(f"Config  : epochs={cfg.num_epochs} batch={cfg.batch_size} lr={cfg.learning_rate} label={cfg.label_col}")
    log(f"Data    : {cfg.input_dir}")

    with open(os.path.join(run_dir, "run_config.json"), "w") as fh:
        json.dump(
            {
                "seeds": list(range(args.start_seed, args.start_seed + args.n_seeds)),
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

    # ---- Load training data ONCE ----
    log(f"Loading training input from {cfg.input_dir}")
    df = load_input(
        cfg.input_dir, iter_num=cfg.input_glob_iter, tail=cfg.file_tail,
        query_col=cfg.query_col, lib_col=cfg.lib_col, fdr_col=cfg.fdr_col,
    )
    log(f"Processed training data: {df.shape[0]} pairs, {df[cfg.query_col].nunique()} unique queries")
    joblib.dump(df, os.path.join(run_dir, "processed_train_data.joblib"), compress=3)

    # ---- Load external held-out data ONCE ----
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
    X_ext_raw = df_ext[feat_cols].values.astype(np.float32)
    ext_labels = {lc: df_ext[lc].values for lc in ext_label_cols if lc in df_ext.columns}
    ext_meta = df_ext[[c for c in cfg.non_feature_cols if c in df_ext.columns]].copy()
    log(f"External prepared: {df_ext.shape[0]} pairs, {df_ext[cfg.query_col].nunique()} unique queries")

    internal_rows = []
    external_rows = []

    for i in range(args.n_seeds):
        seed = args.start_seed + i
        fold = i + 1
        seed_dir = os.path.join(run_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        log(f"================= Seed {seed} (fold {fold}/{args.n_seeds}) =================")

        split = split_cv2_query_holdout(
            df, label_col=cfg.label_col, non_feature_cols=cfg.non_feature_cols,
            query_col=cfg.query_col, test_ratio=cfg.test_ratio, val_ratio=cfg.val_ratio,
            rand_seed=seed,
        )
        log(f"Split -> Train={split['X_train'].shape[0]} Val={split['X_val'].shape[0]} "
            f"Test={split['X_test'].shape[0]} | Test pos={int(split['y_test'].sum())}")

        model, scaler, test_scores, test_labels, test_idx = train_one_fold(split, cfg, device)

        torch.save(model, os.path.join(seed_dir, "model.pth"))
        joblib.dump(scaler, os.path.join(seed_dir, "scaler.joblib"))

        df_test = df.loc[test_idx, cfg.non_feature_cols].copy()
        df_test["predict_proba"] = test_scores
        df_test.to_csv(os.path.join(seed_dir, "test_predictions.tsv"), sep="\t")

        tm = compute_metrics(test_labels, test_scores, k=cfg.topk)
        tm["auroc_sklearn"] = safe_auroc(test_labels, test_scores)
        tm["n_test_pairs"] = int(len(test_labels))
        tm["n_test_positives"] = int(np.sum(test_labels))
        with open(os.path.join(seed_dir, "test_metrics.json"), "w") as fh:
            json.dump(tm, fh, indent=2)
        internal_rows.append({"seed": seed, **tm})
        log(f"[seed {seed}] INTERNAL TEST ROC-AUC={tm['roc_auc']:.4f} PR-AUC={tm['pr_auc']:.4f} "
            f"AP={tm['average_precision']:.4f} P@{cfg.topk}={tm[f'precision_at_{cfg.topk}']:.4f}")

        # External scoring with this fold's model + scaler
        ext_probs = score_external(model, scaler, X_ext_raw, device)
        seed_ext = {}
        for lc, y in ext_labels.items():
            if pd.isna(y).any() or len(np.unique(y)) < 2:
                seed_ext[lc] = {"note": "label missing/constant; skipped"}
                continue
            em = compute_metrics(y, ext_probs, k=cfg.topk)
            em["auroc_sklearn"] = safe_auroc(y, ext_probs)
            em["n_positives"] = int(np.sum(y))
            seed_ext[lc] = em
            external_rows.append({"seed": seed, "label": lc, **em})
            log(f"[seed {seed}] EXTERNAL [{lc}] ROC-AUC={em['roc_auc']:.4f} "
                f"AP={em['average_precision']:.4f} P@{cfg.topk}={em[f'precision_at_{cfg.topk}']:.4f}")
        seed_ext["n_pairs"] = int(X_ext_raw.shape[0])
        with open(os.path.join(seed_dir, "heldout_metrics.json"), "w") as fh:
            json.dump(seed_ext, fh, indent=2)

        if args.save_heldout_preds:
            out = ext_meta.copy()
            out["predict_proba"] = ext_probs
            out.to_csv(os.path.join(seed_dir, "heldout_predictions.tsv"), sep="\t", index=False)

    # ---- Aggregate ----
    df_int = pd.DataFrame(internal_rows)
    df_ext_perf = pd.DataFrame(external_rows)
    df_int.to_csv(os.path.join(run_dir, "internal_test_per_seed.tsv"), sep="\t", index=False)
    df_ext_perf.to_csv(os.path.join(run_dir, "external_per_seed.tsv"), sep="\t", index=False)

    metric_cols = ["roc_auc", "pr_auc", "average_precision",
                   f"precision_at_{cfg.topk}", f"recall_at_{cfg.topk}"]

    summary = {"n_seeds": int(args.n_seeds), "seeds": list(range(args.start_seed, args.start_seed + args.n_seeds))}
    summary["internal_test"] = {
        m: {"mean": float(df_int[m].mean()), "std": float(df_int[m].std())} for m in metric_cols
    }
    summary["external"] = {}
    for lc in ext_label_cols:
        sub = df_ext_perf[df_ext_perf["label"] == lc]
        if len(sub):
            summary["external"][lc] = {
                m: {"mean": float(sub[m].mean()), "std": float(sub[m].std())} for m in metric_cols
            }
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    log("================= AGGREGATE SUMMARY (mean +/- std) =================")
    log("--- INTERNAL CV2 TEST ---")
    for m in metric_cols:
        log(f"  {m:<20} {df_int[m].mean():.4f} +/- {df_int[m].std():.4f}")
    for lc in ext_label_cols:
        sub = df_ext_perf[df_ext_perf["label"] == lc]
        if len(sub):
            log(f"--- EXTERNAL [{lc}] ---")
            for m in metric_cols:
                log(f"  {m:<20} {sub[m].mean():.4f} +/- {sub[m].std():.4f}")
    log("===== DONE =====")
    log(f"All artifacts saved under: {run_dir}")


if __name__ == "__main__":
    main()
