"""Predict genetic-interaction probabilities on new data with a saved model.

Loads one of the trained CV2 checkpoints (saved with ``torch.save(model, ...)``)
together with its matching StandardScaler (.joblib), applies the model to a new
feature table, and writes the predictions to a TSV. If the input table also
contains the ground-truth label column, evaluation metrics are reported.

The new data must contain the same 256 feature columns the model was trained on
(``ko_0..ko_127`` and ``exp_0..exp_127``). Metadata columns are passed through
to the output unchanged.

Examples:
    # Predict and save probabilities
    python reproduce_cv2/predict.py \
        --model data/interim/ReLU128_f_a075_g15_10folds_pt10/CV2_811_GIV_NN_LR1e2_50e_p10_d01_1.pth \
        --scaler data/interim/ReLU128_f_a075_g15_10folds_pt10/CV2_811_seed1.joblib \
        --input new_pairs.tsv \
        --output predictions.tsv

    # Also compute metrics against a known label column
    python reproduce_cv2/predict.py ... --label-col GI_stringent_Type2 --topk 100

Run ``python reproduce_cv2/predict.py --help`` for all options.
"""

from __future__ import annotations

import argparse
import sys

import joblib
import numpy as np
import pandas as pd
import torch

from config import NON_FEATURE_COLS
from metrics import compute_metrics, safe_auroc

# The original checkpoints were pickled from a notebook whose __main__ module
# defined NeuralNetwork / FocalLoss. Register them in __main__ so torch.load can
# un-pickle the whole-model objects regardless of how this script is launched.
import model as _model_module  # noqa: E402

sys.modules["__main__"].NeuralNetwork = _model_module.NeuralNetwork
sys.modules["__main__"].FocalLoss = _model_module.FocalLoss


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_path: str, device: str) -> torch.nn.Module:
    model = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(model, dict):
        raise ValueError(
            f"{model_path} appears to be a state_dict, not a whole model. "
            "Re-instantiate NeuralNetwork and load_state_dict() instead."
        )
    model.to(device)
    model.eval()
    return model


def resolve_feature_columns(df: pd.DataFrame, scaler, feature_cols_arg: str | None):
    """Determine which columns are model features."""
    if feature_cols_arg:
        cols = [c.strip() for c in feature_cols_arg.split(",") if c.strip()]
    else:
        cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    expected = getattr(scaler, "n_features_in_", None)
    if expected is not None and len(cols) != expected:
        raise ValueError(
            f"Found {len(cols)} feature columns but the scaler expects {expected}. "
            "Use --feature-cols to specify them explicitly."
        )

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required feature columns: {missing[:10]} ...")
    return cols


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Path to a saved *.pth model")
    p.add_argument("--scaler", required=True, help="Path to the matching StandardScaler *.joblib")
    p.add_argument("--input", required=True, help="New data table (TSV by default)")
    p.add_argument("--output", required=True, help="Where to write predictions (TSV)")
    p.add_argument("--sep", default="\t", help="Field separator for input/output (default: tab)")
    p.add_argument("--index-col", default=None, help="Column to use as the input index (optional)")
    p.add_argument(
        "--feature-cols",
        default=None,
        help="Comma-separated feature column names. If omitted, all columns not in "
        "the standard metadata list are used.",
    )
    p.add_argument(
        "--label-col",
        default=None,
        help="If present in the input, compute evaluation metrics against it.",
    )
    p.add_argument("--topk", type=int, default=100, help="K for Precision@K / Recall@K")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto-detect if unset)")
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"Using {device} device")

    model = load_model(args.model, device)
    scaler = joblib.load(args.scaler)

    df = pd.read_csv(args.input, sep=args.sep, index_col=args.index_col)
    print(f"Loaded {df.shape[0]} rows from {args.input}")

    feature_cols = resolve_feature_columns(df, scaler, args.feature_cols)

    # Drop rows with missing features (cannot be scored) and remember which remain
    X_df = df[feature_cols]
    valid_mask = X_df.notna().all(axis=1)
    n_dropped = int((~valid_mask).sum())
    if n_dropped:
        print(f"Dropping {n_dropped} rows with missing feature values.")
    X_df = X_df.loc[valid_mask]

    X = scaler.transform(X_df.values)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Batched inference
    probs = np.empty(X_tensor.shape[0], dtype=np.float32)
    with torch.no_grad():
        for start in range(0, X_tensor.shape[0], args.batch_size):
            batch = X_tensor[start : start + args.batch_size].to(device)
            logits = model(batch).squeeze(-1)
            probs[start : start + batch.shape[0]] = torch.sigmoid(logits).cpu().numpy()

    out = df.loc[valid_mask].copy()
    out["predict_proba"] = probs
    out = out.sort_values("predict_proba", ascending=False)
    out.to_csv(args.output, sep=args.sep)
    print(f"Wrote {out.shape[0]} predictions to {args.output}")

    if args.label_col and args.label_col in out.columns:
        y_true = out[args.label_col].values
        if pd.notna(y_true).all() and len(np.unique(y_true)) > 1:
            m = compute_metrics(y_true, out["predict_proba"].values, k=args.topk)
            m["auroc_sklearn"] = safe_auroc(y_true, out["predict_proba"].values)
            print("\n--- Evaluation metrics ---")
            for key, val in m.items():
                print(f"  {key}: {val:.4f}")
        else:
            print(f"\nLabel column '{args.label_col}' is missing values or single-class; "
                  "skipping metrics.")


if __name__ == "__main__":
    main()
