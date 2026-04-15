"""Quick single-run training with default (or manually specified) hyperparameters.

Use this for a fast sanity check before committing to full Optuna tuning or
10-fold evaluation.

Usage
-----
    # Defaults (128→64→32, lr=1e-2, 50 epochs, query-holdout split)
    python -m src.train --data-dir data/input/GIV_24Q4/ReLU128_5L

    # Override some params
    python -m src.train --data-dir data/input/GIV_24Q4/ReLU128_5L \\
        --hidden-sizes 256 128 64 --lr 5e-3 --epochs 30 --batch-size 128

    # Run a specific fold of 10-fold CV
    python -m src.train --data-dir data/input/GIV_24Q4/ReLU128_5L \\
        --cv 4 --fold 1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .data import (
    NON_FEATURE_COLS,
    filter_unique_pairs,
    load_tsv_iterations,
    split_query_holdout,
    split_query_kfold,
    split_random_stratified,
)
from .training import run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single training run with default hyperparameters",
    )
    # Data
    p.add_argument("--data-dir", required=True, help="Directory with numbered TSV files")
    p.add_argument("--n-iters", type=int, default=20)
    p.add_argument("--label-col", default="GI_stringent_Type2")

    # Split strategy
    p.add_argument("--cv", type=int, default=2, choices=[1, 2, 4],
                   help="1=random-stratified, 2=query-holdout, 4=k-fold")
    p.add_argument("--fold", type=int, default=1, help="Fold number (for cv=4)")
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    # Model architecture
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 64, 32])
    p.add_argument("--dropout", type=float, default=0.3)

    # Training
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--scheduler", default="plateau", choices=["plateau", "cosine"])
    p.add_argument("--focal-alpha", type=float, default=0.75)
    p.add_argument("--focal-gamma", type=float, default=1.5)
    p.add_argument("--balanced-sampling", action="store_true")
    p.add_argument("--top-k", type=int, default=100)

    # Output
    p.add_argument("--output-dir", default="outputs/test_run")
    p.add_argument("--tag", default="test")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data_dir} ...")
    df = load_tsv_iterations(args.data_dir, n_iters=args.n_iters)
    df = filter_unique_pairs(df)
    print(f"Loaded {len(df)} unique gene pairs")

    # Split
    if args.cv == 1:
        split = split_random_stratified(df, args.label_col, seed=args.seed,
                                        non_feature_cols=NON_FEATURE_COLS)
    elif args.cv == 2:
        split = split_query_holdout(df, args.label_col, seed=args.seed,
                                    non_feature_cols=NON_FEATURE_COLS)
    else:
        split = split_query_kfold(df, args.label_col, fold=args.fold,
                                  seed=args.seed, n_folds=args.n_folds,
                                  non_feature_cols=NON_FEATURE_COLS)

    print(f"Split sizes — train: {len(split['y_train'])}, "
          f"val: {len(split['y_val'])}, test: {len(split['y_test'])}")

    # Train + evaluate
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
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        top_k=args.top_k,
        output_dir=output_dir,
        tag=args.tag,
    )

    # Print summary
    print("\n=== Test Results ===")
    for k, v in result.items():
        if k not in ("history", "model"):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nOutputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
