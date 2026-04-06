"""External validation using the OLD 10 NeuralNetwork models (torch.save'd).

These models live in ``data/interim/ReLU128_f_a075_g15_10folds_pt10/`` and
were saved as full ``torch.save(model)`` objects using the 3-hidden-layer
``NeuralNetwork`` class (128→64→32, dropout=0.3).  The corresponding
scalers are ``CV2_811_seed{1..10}.joblib``.

This script mirrors ``src/external_validate.py`` but loads the old-format
models and produces the same metrics:

    model_metrics.tsv, model_summary.tsv, query_metrics.tsv,
    query_summary_per_model.tsv, ensemble_predictions.tsv,
    external_val_config.json

Usage
-----
    python -m src.external_validate_old_models              # defaults below
    python -m src.external_validate_old_models --top-k 50

Outputs go to ``outputs/external_val/external_val_old_models/`` by default.
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
import torch.nn as nn

from .data import NON_FEATURE_COLS, filter_unique_pairs, load_tsv_iterations
from .training import compute_all_metrics, plot_roc_pr
from .external_validate import compute_query_metrics


# ---------------------------------------------------------------------------
# Old NeuralNetwork class (needed for torch.load)
# ---------------------------------------------------------------------------

class NeuralNetwork(nn.Module):
    """The legacy 3-hidden-layer architecture used during training."""

    def __init__(
        self,
        input_size: int,
        hidden_size1: int,
        hidden_size2: int,
        hidden_size3: int,
        output_size: int = 1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(torch.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(torch.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x).squeeze(-1)
        return x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _register_legacy_class() -> None:
    """Make ``NeuralNetwork`` importable as ``__main__.NeuralNetwork``.

    The old models were saved with ``torch.save(model)`` from a script where
    the class lived in ``__main__``.  ``torch.load`` tries to resolve that
    path, so we inject our copy into ``sys.modules['__main__']``.
    """
    import sys
    main_mod = sys.modules["__main__"]
    if not hasattr(main_mod, "NeuralNetwork"):
        main_mod.NeuralNetwork = NeuralNetwork  # type: ignore[attr-defined]


def _discover_old_models(
    model_dir: Path,
    model_prefix: str = "CV2_811_GIV_NN_LR1e2_50e_p10_d01_",
    scaler_prefix: str = "CV2_811_seed",
    n_folds: int = 10,
    start_idx: int = 1,
) -> list[tuple[Path, Path]]:
    """Return sorted (model.pth, scaler.joblib) pairs."""
    pairs = []
    for i in range(start_idx, start_idx + n_folds):
        mp = model_dir / f"{model_prefix}{i}.pth"
        sp = model_dir / f"{scaler_prefix}{i}.joblib"
        if not mp.exists():
            print(f"  Warning: model not found: {mp}")
            continue
        if not sp.exists():
            print(f"  Warning: scaler not found: {sp}")
            continue
        pairs.append((mp, sp))
    if not pairs:
        raise FileNotFoundError(f"No model/scaler pairs found in {model_dir}")
    return pairs


def _prepare_features(
    df: pd.DataFrame,
    label_col: str,
    non_feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Extract features and labels, dropping NaN rows."""
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    X = df[feature_cols].dropna(how="any")
    y = df.loc[X.index, label_col]
    return X, y, feature_cols


@torch.no_grad()
def _predict(
    model: nn.Module,
    X_scaled: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Run inference and return predicted probabilities."""
    model.eval()
    model.to(device)
    preds = []
    n = len(X_scaled)
    for i in range(0, n, batch_size):
        batch = torch.tensor(
            X_scaled[i : i + batch_size], dtype=torch.float32
        ).to(device)
        logits = model(batch)
        preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="External validation with old NeuralNetwork models",
    )
    p.add_argument(
        "--data-dir",
        default="data/input/GIV_24Q4_heldout",
        help="Directory with heldout TSV files",
    )
    p.add_argument("--n-iters", type=int, default=11,
                   help="Number of TSV iterations to load")
    p.add_argument("--label-col", default="GI_stringent_Type2",
                   help="Binary label column")
    p.add_argument(
        "--model-dir",
        default="data/interim/ReLU128_f_a075_g15_10folds_pt10",
        help="Directory containing old .pth models and .joblib scalers",
    )
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--top-k", type=int, default=100,
                   help="K for Recall@K and Precision@K")
    p.add_argument("--output-dir", default="outputs/external_val/external_val_old_models")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    device = _get_device()

    # 0. Register legacy class so torch.load can unpickle __main__.NeuralNetwork
    _register_legacy_class()

    # 1. Discover old model/scaler pairs
    model_pairs = _discover_old_models(model_dir, n_folds=args.n_folds)
    print(f"Found {len(model_pairs)} model(s) in {model_dir}")

    # 2. Load and clean external validation data
    print(f"\nLoading external data from {args.data_dir} ...")
    df = load_tsv_iterations(args.data_dir, n_iters=args.n_iters)
    df = filter_unique_pairs(df)
    print(f"Loaded {len(df)} unique gene pairs")

    X, y, feature_cols = _prepare_features(df, args.label_col, NON_FEATURE_COLS)
    print(
        f"Features: {X.shape[1]}, samples: {len(X)}, "
        f"positives: {int(y.sum())} ({y.mean()*100:.2f}%)"
    )

    # 3. Run each model on the external set
    all_model_metrics: list[dict] = []
    all_probs: list[np.ndarray] = []
    all_query_rows: list[pd.DataFrame] = []

    for model_path, scaler_path in model_pairs:
        tag = model_path.stem  # e.g. CV2_811_GIV_NN_LR1e2_50e_p10_d01_1
        print(f"\n--- {tag} ---")

        # Load scaler and align features
        scaler = joblib.load(scaler_path)
        n_scaler_features = scaler.n_features_in_
        if X.shape[1] != n_scaler_features:
            if hasattr(scaler, "feature_names_in_"):
                keep_cols = [c for c in scaler.feature_names_in_ if c in X.columns]
                X_subset = X[keep_cols]
            else:
                X_subset = X.iloc[:, :n_scaler_features]
            print(f"  Aligned features: {X.shape[1]} -> {X_subset.shape[1]}")
        else:
            X_subset = X
        X_scaled = scaler.transform(X_subset)

        # Load old full-model (torch.save'd entire model object)
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()

        # Predict
        y_score = _predict(model, X_scaled, device)
        all_probs.append(y_score)

        # Compute metrics (same as external_validate.py)
        y_true = y.values
        metrics = compute_all_metrics(y_true, y_score, top_k=args.top_k)
        metrics["model"] = tag
        all_model_metrics.append(metrics)
        print(
            f"  AUROC={metrics['AUROC']:.4f}  AUPR={metrics['AUPR']:.4f}  "
            f"AP={metrics['AP']:.4f}"
        )

        # Plot ROC/PR per model
        plot_roc_pr(y_true, y_score, save_path=output_dir / f"{tag}_roc_pr.png")

        # Per-query metrics for this model
        df_pred = df.loc[X.index, ["Query"]].copy()
        df_pred["y_true"] = y_true
        df_pred["y_score"] = y_score
        q_metrics = compute_query_metrics(df_pred, top_k=args.top_k)
        q_metrics["model"] = tag
        all_query_rows.append(q_metrics)

    # 4. Per-model metrics table
    df_models = pd.DataFrame(all_model_metrics)
    front = ["model"]
    rest = [c for c in df_models.columns if c != "model"]
    df_models = df_models[front + rest]
    df_models.to_csv(output_dir / "model_metrics.tsv", sep="\t", index=False)

    # Model-level summary (mean ± SE)
    metric_cols = [c for c in df_models.columns if c != "model"]
    summary: dict[str, dict[str, float]] = {}
    print(f"\n{'='*60}")
    print(f"  External Validation: {len(model_pairs)} models  (mean ± SE)")
    print(f"{'='*60}")
    for col in metric_cols:
        vals = df_models[col].values
        mean = float(np.mean(vals))
        se = (
            float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            if len(vals) > 1
            else 0.0
        )
        summary[col] = {"mean": mean, "se": se}
        print(f"  {col:>20s}:  {mean:.4f} ± {se:.4f}")

    summary_df = pd.DataFrame(
        [{"metric": k, "mean": v["mean"], "se": v["se"]} for k, v in summary.items()]
    )
    summary_df.to_csv(output_dir / "model_summary.tsv", sep="\t", index=False)

    # 5. Ensemble predictions (mean probability across models)
    ensemble_probs = np.mean(np.column_stack(all_probs), axis=1)
    df_ensemble = df.loc[
        X.index, [c for c in NON_FEATURE_COLS if c in df.columns]
    ].copy()
    df_ensemble["predict_proba"] = ensemble_probs
    df_ensemble.to_csv(
        output_dir / "ensemble_predictions.tsv", sep="\t", index=False
    )

    ensemble_metrics = compute_all_metrics(
        y.values, ensemble_probs, top_k=args.top_k
    )
    print(
        f"\n  Ensemble AUROC={ensemble_metrics['AUROC']:.4f}  "
        f"AUPR={ensemble_metrics['AUPR']:.4f}  "
        f"AP={ensemble_metrics['AP']:.4f}"
    )
    plot_roc_pr(
        y.values, ensemble_probs, save_path=output_dir / "ensemble_roc_pr.png"
    )

    # 6. Per-query metrics
    df_query_all = pd.concat(all_query_rows, ignore_index=True)
    df_query_all.to_csv(
        output_dir / "query_summary_per_model.tsv", sep="\t", index=False
    )

    query_avg = (
        df_query_all.groupby("Query")
        .agg(
            n_pairs=("n_pairs", "first"),
            n_pos=("n_pos", "first"),
            pos_rate=("pos_rate", "first"),
            AUROC_mean=("AUROC", "mean"),
            AUROC_se=(
                "AUROC",
                lambda x: x.std(ddof=1) / np.sqrt(x.notna().sum())
                if x.notna().sum() > 1
                else 0.0,
            ),
            AUPR_mean=("AUPR", "mean"),
            AUPR_se=(
                "AUPR",
                lambda x: x.std(ddof=1) / np.sqrt(x.notna().sum())
                if x.notna().sum() > 1
                else 0.0,
            ),
            AP_mean=("AP", "mean"),
            AP_se=(
                "AP",
                lambda x: x.std(ddof=1) / np.sqrt(x.notna().sum())
                if x.notna().sum() > 1
                else 0.0,
            ),
        )
        .reset_index()
        .sort_values("AUROC_mean", ascending=False)
    )
    query_avg.to_csv(output_dir / "query_metrics.tsv", sep="\t", index=False)

    n_queries = len(query_avg)
    n_good = len(query_avg[query_avg["AUROC_mean"] >= 0.7])
    n_poor = len(query_avg[query_avg["AUROC_mean"] < 0.5])
    print(f"\n  Per-query summary: {n_queries} queries total")
    print(f"    AUROC >= 0.7: {n_good} queries")
    print(f"    AUROC <  0.5: {n_poor} queries")
    print(f"\n  Top 10 easiest queries:")
    for _, row in query_avg.head(10).iterrows():
        print(
            f"    {row['Query']:>12s}  AUROC={row['AUROC_mean']:.4f}  "
            f"AUPR={row['AUPR_mean']:.4f}  "
            f"n_pos={int(row['n_pos'])}/{int(row['n_pairs'])}"
        )
    print(f"\n  Top 10 hardest queries:")
    for _, row in query_avg.tail(10).iterrows():
        print(
            f"    {row['Query']:>12s}  AUROC={row['AUROC_mean']:.4f}  "
            f"AUPR={row['AUPR_mean']:.4f}  "
            f"n_pos={int(row['n_pos'])}/{int(row['n_pairs'])}"
        )

    # 7. Save config
    out_config = {
        "data_dir": args.data_dir,
        "n_iters": args.n_iters,
        "label_col": args.label_col,
        "model_dir": str(model_dir),
        "n_models": len(model_pairs),
        "architecture": "NeuralNetwork(256→128→64→32→1, dropout=0.3)",
        "top_k": args.top_k,
        "n_samples": len(X),
        "n_features": X_scaled.shape[1],
        "n_positives": int(y.sum()),
        "model_summary": summary,
        "ensemble_metrics": {k: float(v) for k, v in ensemble_metrics.items()},
        "n_queries": n_queries,
    }
    with open(output_dir / "external_val_config.json", "w") as f:
        json.dump(out_config, f, indent=2)
    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
