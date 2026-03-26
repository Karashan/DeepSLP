"""10-fold cross-validation with default (or CLI) hyperparameters.

Runs a single 10-fold CV (cv=4: query-gene k-fold) so that every sample
appears in the test set exactly once.  After all folds, prints mean ± SE
of test metrics across the 10 folds.

Usage
-----
    # Defaults: 10 folds, same architecture as train.py
    python -m src.cross_validate --data-dir data/input/GIV_24Q4/ReLU128_5L

    # Override hyperparameters
    python -m src.cross_validate --data-dir data/input/GIV_24Q4/ReLU128_5L \\
        --hidden-sizes 256 128 64 --lr 5e-3 --n-folds 5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    NON_FEATURE_COLS,
    filter_unique_pairs,
    load_tsv_iterations,
    split_query_kfold,
)
from .training import run_pipeline


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="k-fold cross-validation with default hyperparameters",
    )
    # Data
    p.add_argument("--data-dir", required=True,
                   help="Directory with numbered TSV files")
    p.add_argument("--n-iters", type=int, default=20)
    p.add_argument("--label-col", default="GI_stringent_Type2")

    # CV design
    p.add_argument("--n-folds", type=int, default=10,
                   help="Number of folds")
    p.add_argument("--seed", type=int, default=42)

    # Model architecture
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 64, 32])
    p.add_argument("--dropout", type=float, default=0.3)

    # Training
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--scheduler", default="plateau", choices=["plateau", "cosine"])
    p.add_argument("--lr-decay-factor", type=float, default=0.5)
    p.add_argument("--lr-patience", type=int, default=5)
    p.add_argument("--focal-alpha", type=float, default=0.75)
    p.add_argument("--focal-gamma", type=float, default=1.5)
    p.add_argument("--balanced-sampling", action="store_true")
    p.add_argument("--top-k", type=int, default=100)

    # Output
    p.add_argument("--output-dir", default="outputs/cross_validate")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] Starting {args.n_folds}-fold cross-validation")

    # ------------------------------------------------------------------
    # 1. Load and clean data (once)
    # ------------------------------------------------------------------
    print(f"Loading data from {args.data_dir} ...")
    df = load_tsv_iterations(args.data_dir, n_iters=args.n_iters)
    df = filter_unique_pairs(df)
    print(f"Loaded {len(df)} unique gene pairs\n")

    # ------------------------------------------------------------------
    # 2. k-fold CV
    # ------------------------------------------------------------------
    all_fold_rows: list[dict] = []

    wall_t0 = time.time()

    for fold in range(1, args.n_folds + 1):
        print(f"\n{'='*60}")
        print(f"  Fold {fold}/{args.n_folds}")
        print(f"{'='*60}")

        split = split_query_kfold(
            df, args.label_col,
            fold=fold, seed=args.seed, n_folds=args.n_folds,
            non_feature_cols=NON_FEATURE_COLS,
        )
        print(f"  train={len(split['y_train'])}  "
              f"val={len(split['y_val'])}  "
              f"test={len(split['y_test'])}")

        result = run_pipeline(
            split, df, NON_FEATURE_COLS,
            hidden_sizes=args.hidden_sizes,
            dropout=args.dropout,
            batch_size=args.batch_size,
            balanced_sampling=args.balanced_sampling,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            epochs=args.epochs,
            patience=args.patience,
            scheduler_type=args.scheduler,
            lr_decay_factor=args.lr_decay_factor,
            lr_patience=args.lr_patience,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            top_k=args.top_k,
            output_dir=output_dir,
            tag=f"fold_{fold}",
            save_model=False,
        )

        fold_metrics = {k: v for k, v in result.items()
                       if k not in ("history", "model")}
        fold_metrics["fold"] = fold
        all_fold_rows.append(fold_metrics)

    wall_elapsed = time.time() - wall_t0

    # ------------------------------------------------------------------
    # 3. Save per-fold table
    # ------------------------------------------------------------------
    df_all = pd.DataFrame(all_fold_rows)
    front = ["fold"]
    rest = [c for c in df_all.columns if c != "fold"]
    df_all = df_all[front + rest]
    fold_path = output_dir / "fold_metrics.tsv"
    df_all.to_csv(fold_path, sep="\t", index=False)
    print(f"\nPer-fold metrics saved to {fold_path}")

    # ------------------------------------------------------------------
    # 4. Summary: mean ± SE across folds
    # ------------------------------------------------------------------
    metric_cols = [c for c in df_all.columns if c != "fold"]

    print(f"\n{'='*60}")
    print(f"  Summary: {args.n_folds}-fold CV  (mean ± SE)")
    print(f"{'='*60}")

    summary_rows = {}
    for col in metric_cols:
        vals = df_all[col].values
        mean = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        summary_rows[col] = {"mean": mean, "se": se}
        print(f"  {col:>20s}:  {mean:.4f} ± {se:.4f}")

    print(f"\n  Wall time: {wall_elapsed / 60:.1f} min")

    # ------------------------------------------------------------------
    # 5. Save config and summary JSON
    # ------------------------------------------------------------------
    config = {
        "data_dir": args.data_dir,
        "label_col": args.label_col,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "hidden_sizes": args.hidden_sizes,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "epochs": args.epochs,
        "patience": args.patience,
        "scheduler": args.scheduler,
        "lr_decay_factor": args.lr_decay_factor,
        "lr_patience": args.lr_patience,
        "focal_alpha": args.focal_alpha,
        "focal_gamma": args.focal_gamma,
        "balanced_sampling": args.balanced_sampling,
        "top_k": args.top_k,
        "wall_time_min": round(wall_elapsed / 60, 2),
        "summary": {
            col: {"mean": row["mean"], "se": row["se"]}
            for col, row in summary_rows.items()
        },
    }
    config_path = output_dir / "cv_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config + summary saved to {config_path}")


if __name__ == "__main__":
    main()
