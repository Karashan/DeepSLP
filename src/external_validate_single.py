"""External validation with a single trained model.

Loads one model + scaler pair and evaluates on a held-out dataset.
Reports overall metrics, per-query metrics, and saves predictions.

Usage
-----
    python -m src.external_validate_single \\
        --data-dir data/input/GIV_24Q4_heldout \\
        --model-pt outputs/test_run/test_model.pt \\
        --scaler-pkl outputs/test_run/test_scaler.pkl \\
        --hidden-sizes 128 64 32 \\
        --output-dir outputs/external_val_single
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .data import (
    NON_FEATURE_COLS,
    filter_unique_pairs,
    load_tsv_iterations,
)
from .external_validate import _predict, compute_query_metrics
from .models import SLPredictorMLP
from .training import compute_all_metrics, get_device, plot_roc_pr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="External validation with a single trained model",
    )
    p.add_argument("--data-dir", required=True,
                   help="Directory with external validation TSV files")
    p.add_argument("--n-iters", type=int, default=11)
    p.add_argument("--label-col", default="GI_stringent_Type2")
    p.add_argument("--model-pt", required=True,
                   help="Path to saved model weights (.pt)")
    p.add_argument("--scaler-pkl", required=True,
                   help="Path to saved StandardScaler (.pkl)")
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 64, 32],
                   help="Hidden layer sizes matching the saved model")
    p.add_argument("--dropout", type=float, default=0.3,
                   help="Dropout rate matching the saved model")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--output-dir", default="outputs/external_val_single")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()

    # ------------------------------------------------------------------
    # 1. Load external validation data
    # ------------------------------------------------------------------
    print(f"Loading external data from {args.data_dir} ...")
    df = load_tsv_iterations(args.data_dir, n_iters=args.n_iters)
    df = filter_unique_pairs(df)
    print(f"Loaded {len(df)} unique gene pairs")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].dropna(how="any")
    y = df.loc[X.index, args.label_col]
    print(f"Features: {X.shape[1]}, samples: {len(X)}, "
          f"positives: {int(y.sum())} ({y.mean()*100:.2f}%)")

    # ------------------------------------------------------------------
    # 2. Load scaler and transform
    # ------------------------------------------------------------------
    scaler = joblib.load(args.scaler_pkl)
    n_scaler_features = scaler.n_features_in_
    if X.shape[1] != n_scaler_features:
        if hasattr(scaler, "feature_names_in_"):
            keep_cols = [c for c in scaler.feature_names_in_ if c in X.columns]
            X = X[keep_cols]
        else:
            X = X.iloc[:, :n_scaler_features]
        print(f"  Aligned features to scaler: {n_scaler_features}")

    X_scaled = scaler.transform(X)

    # ------------------------------------------------------------------
    # 3. Load model and predict
    # ------------------------------------------------------------------
    input_size = X_scaled.shape[1]
    model = SLPredictorMLP(input_size, args.hidden_sizes, args.dropout)
    state = torch.load(args.model_pt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Model loaded: {args.model_pt}")

    y_score = _predict(model, X_scaled, device)
    y_true = y.values

    # ------------------------------------------------------------------
    # 4. Overall metrics
    # ------------------------------------------------------------------
    metrics = compute_all_metrics(y_true, y_score, top_k=args.top_k)

    print(f"\n{'='*60}")
    print("  External Validation Metrics")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k:>20s}:  {v:.4f}")

    plot_roc_pr(y_true, y_score, save_path=output_dir / "roc_pr.png")

    # ------------------------------------------------------------------
    # 5. Save predictions
    # ------------------------------------------------------------------
    df_pred = df.loc[X.index, [c for c in NON_FEATURE_COLS if c in df.columns]].copy()
    df_pred["predict_proba"] = y_score
    df_pred.to_csv(output_dir / "predictions.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------
    # 6. Per-query metrics
    # ------------------------------------------------------------------
    df_q_input = df.loc[X.index, ["Query"]].copy()
    df_q_input["y_true"] = y_true
    df_q_input["y_score"] = y_score
    query_df = compute_query_metrics(df_q_input, top_k=args.top_k)
    query_df = query_df.sort_values("AUROC", ascending=False)
    query_df.to_csv(output_dir / "query_metrics.tsv", sep="\t", index=False)

    n_queries = len(query_df)
    n_good = len(query_df[query_df["AUROC"] >= 0.7])
    n_poor = len(query_df[query_df["AUROC"] < 0.5])
    print(f"\n  Per-query summary: {n_queries} queries total")
    print(f"    AUROC >= 0.7: {n_good} queries")
    print(f"    AUROC <  0.5: {n_poor} queries")

    print(f"\n  Top 5 easiest queries:")
    for _, row in query_df.head(5).iterrows():
        print(f"    {row['Query']:>12s}  AUROC={row['AUROC']:.4f}  "
              f"AUPR={row['AUPR']:.4f}  n_pos={int(row['n_pos'])}/{int(row['n_pairs'])}")
    print(f"\n  Top 5 hardest queries:")
    for _, row in query_df.tail(5).iterrows():
        print(f"    {row['Query']:>12s}  AUROC={row['AUROC']:.4f}  "
              f"AUPR={row['AUPR']:.4f}  n_pos={int(row['n_pos'])}/{int(row['n_pairs'])}")

    # ------------------------------------------------------------------
    # 7. Save config
    # ------------------------------------------------------------------
    out_config = {
        "data_dir": args.data_dir,
        "n_iters": args.n_iters,
        "label_col": args.label_col,
        "model_pt": str(args.model_pt),
        "scaler_pkl": str(args.scaler_pkl),
        "hidden_sizes": args.hidden_sizes,
        "dropout": args.dropout,
        "top_k": args.top_k,
        "n_samples": len(X),
        "n_features": X_scaled.shape[1],
        "n_positives": int(y.sum()),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "n_queries": n_queries,
    }
    with open(output_dir / "external_val_config.json", "w") as f:
        json.dump(out_config, f, indent=2)
    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
