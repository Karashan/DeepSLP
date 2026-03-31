"""Final 10-fold evaluation using best hyperparameters from Optuna tuning.

Usage
-----
    python -m src.evaluate --data-dir data/input/GIV_24Q4/ReLU128_5L \\
                           --tuning-json outputs/tuning/tuning_results.json \\
                           --output-dir outputs/final

Reads the best hyperparameters from the tuning results JSON, runs a full
10-fold cross-validation (CV4: query-gene holdout), saves per-fold artefacts
(model, predictions, dashboard, curves CSV), and produces a summary table
with mean ± std across folds.
"""

from __future__ import annotations

import argparse
import json
import os
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
        description="Final evaluation with best hyperparameters from Optuna tuning",
    )
    p.add_argument("--data-dir", required=True,
                   help="Directory with numbered TSV files")
    p.add_argument("--n-iters", type=int, default=20,
                   help="Number of TSV iterations to load")
    p.add_argument("--label-col", default="GI_stringent_Type2",
                   help="Binary label column")
    p.add_argument("--tuning-json", required=True,
                   help="Path to tuning_results.json from src.tune")
    p.add_argument("--cv", type=int, default=4, choices=[1, 2, 4],
                   help="CV strategy for final eval (default: 4 = 10-fold)")
    p.add_argument("--n-folds", type=int, default=10,
                   help="Number of folds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50,
                   help="Max training epochs per fold")
    p.add_argument("--top-k", type=int, default=100,
                   help="K for Recall@K and Precision@K")
    p.add_argument("--output-dir", default="outputs/final")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load best hyperparameters
    # ------------------------------------------------------------------
    with open(args.tuning_json) as f:
        tuning = json.load(f)
    best = tuning["best_params"]
    print(f"Best tuning AUPR: {tuning['best_value']:.4f}")
    print(f"Best params: {json.dumps(best, indent=2)}")

    # Reconstruct hidden sizes (supports both preset and per-layer formats)
    if "arch" in best:
        ARCH_PRESETS = {
            "64-32": [64, 32], "128-64": [128, 64],
            "128-64-32": [128, 64, 32], "256-128-64": [256, 128, 64],
        }
        hidden_sizes = ARCH_PRESETS[best["arch"]]
    else:
        n_layers = best["n_layers"]
        hidden_sizes = [best[f"hidden_{i}"] for i in range(n_layers)]

    # ------------------------------------------------------------------
    # 2. Load and clean data
    # ------------------------------------------------------------------
    print(f"\nLoading data from {args.data_dir} ...")
    df = load_tsv_iterations(args.data_dir, n_iters=args.n_iters)
    df = filter_unique_pairs(df)
    print(f"Loaded {len(df)} unique gene pairs")

    # ------------------------------------------------------------------
    # 3. Run k-fold evaluation
    # ------------------------------------------------------------------
    all_metrics: list[dict] = []

    for fold in range(1, args.n_folds + 1):
        print(f"\n{'='*60}")
        print(f"  Fold {fold}/{args.n_folds}")
        print(f"{'='*60}")

        split = split_query_kfold(
            df, args.label_col, fold=fold, seed=args.seed,
            n_folds=args.n_folds,
            non_feature_cols=NON_FEATURE_COLS,
        )

        result = run_pipeline(
            split, df, NON_FEATURE_COLS,
            hidden_sizes=hidden_sizes,
            dropout=best["dropout"],
            batch_size=best["batch_size"],
            balanced_sampling=best.get("balanced_sampling", False),
            lr=best["lr"],
            weight_decay=best["weight_decay"],
            max_grad_norm=best.get("max_grad_norm", 1.0),
            epochs=args.epochs,
            focal_alpha=best["focal_alpha"],
            focal_gamma=best["focal_gamma"],
            scheduler_type=best.get("scheduler_type", "plateau"),
            top_k=args.top_k,
            output_dir=output_dir,
            tag=f"fold_{fold}",
            save_model=True,
        )

        # Collect metrics (exclude non-serialisable objects)
        fold_metrics = {k: v for k, v in result.items()
                        if k not in ("history", "model")}
        fold_metrics["fold"] = fold
        all_metrics.append(fold_metrics)

    # ------------------------------------------------------------------
    # 4. Aggregate and save summary
    # ------------------------------------------------------------------
    df_perf = pd.DataFrame(all_metrics)

    # Move fold column to front
    cols = ["fold"] + [c for c in df_perf.columns if c != "fold"]
    df_perf = df_perf[cols]

    summary_path = output_dir / "performance_summary.tsv"
    df_perf.to_csv(summary_path, sep="\t", index=False)

    print(f"\n{'='*60}")
    print("  Final Results (mean ± std across folds)")
    print(f"{'='*60}")
    metric_cols = [c for c in df_perf.columns if c != "fold"]
    for col in metric_cols:
        mean = df_perf[col].mean()
        std = df_perf[col].std()
        print(f"  {col:>20s}:  {mean:.4f} ± {std:.4f}")

    print(f"\nPer-fold table saved to {summary_path}")

    # Save config used for reproducibility
    config = {
        "data_dir": args.data_dir,
        "label_col": args.label_col,
        "cv": args.cv,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "epochs": args.epochs,
        "top_k": args.top_k,
        "tuning_json": str(args.tuning_json),
        "best_tuning_aupr": tuning["best_value"],
        "best_params": best,
        "mean_metrics": {col: float(df_perf[col].mean()) for col in metric_cols},
        "std_metrics": {col: float(df_perf[col].std()) for col in metric_cols},
    }
    config_path = output_dir / "eval_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")


if __name__ == "__main__":
    main()
