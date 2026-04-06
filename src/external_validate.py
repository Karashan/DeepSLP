"""External validation: test saved models on a held-out dataset.

Loads all ``*_model.pt`` / ``*_scaler.pkl`` pairs from a model directory,
applies each to an external validation set, and reports per-model and
per-query metrics.

Usage
-----
    python -m src.external_validate \\
        --data-dir data/input/GIV_24Q4_heldout \\
        --model-dir outputs/final \\
        --output-dir outputs/external_val

Outputs
-------
    model_metrics.tsv          – one row per model (AUROC, AUPR, AP, P@K, R@K)
    model_summary.tsv          – mean ± SE across models
    query_metrics.tsv          – per-query metrics averaged over all models
    query_summary_per_model.tsv – per (model, query) granularity
    ensemble_predictions.tsv   – mean predicted probability from all models
    external_val_config.json   – full config and summary
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

from .data import (
    NON_FEATURE_COLS,
    filter_unique_pairs,
    load_tsv_iterations,
)
from .models import SLPredictorMLP
from .training import compute_all_metrics, get_device, plot_roc_pr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ARCH_PRESETS = {
    "64-32": [64, 32],
    "128-64": [128, 64],
    "128-64-32": [128, 64, 32],
    "256-128-64": [256, 128, 64],
}


def _load_config(model_dir: Path) -> dict:
    """Load eval_config.json from the model directory."""
    config_path = model_dir / "eval_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No eval_config.json in {model_dir}. "
            "Run src.evaluate first to generate models."
        )
    with open(config_path) as f:
        return json.load(f)


def _get_hidden_sizes(config: dict) -> list[int]:
    """Extract hidden_sizes from eval config."""
    best = config["best_params"]
    if "arch" in best:
        return ARCH_PRESETS[best["arch"]]
    n_layers = best["n_layers"]
    return [best[f"hidden_{i}"] for i in range(n_layers)]


def _discover_models(model_dir: Path) -> list[tuple[Path, Path]]:
    """Find all (model.pt, scaler.pkl) pairs, sorted by name."""
    models = sorted(model_dir.glob("*_model.pt"))
    pairs = []
    for mp in models:
        tag = mp.name.replace("_model.pt", "")
        sp = model_dir / f"{tag}_scaler.pkl"
        if sp.exists():
            pairs.append((mp, sp))
        else:
            print(f"  Warning: no scaler found for {mp.name}, skipping")
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
    """Run model inference and return predicted probabilities."""
    model.eval()
    model.to(device)
    preds = []
    n = len(X_scaled)
    for i in range(0, n, batch_size):
        batch = torch.tensor(X_scaled[i : i + batch_size], dtype=torch.float32).to(device)
        logits = model(batch)
        preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(preds)


def compute_query_metrics(
    df_pred: pd.DataFrame,
    query_col: str = "Query",
    label_col: str = "y_true",
    score_col: str = "y_score",
    top_k: int = 100,
) -> pd.DataFrame:
    """Compute per-query metrics.

    For each unique query, calculates AUROC, AUPR, AP, and the number of
    positives/total pairs.  Queries with fewer than 2 classes are marked NaN
    (metrics are undefined).
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = []
    for query, grp in df_pred.groupby(query_col):
        y_true = grp[label_col].values
        y_score = grp[score_col].values
        n_pos = int(y_true.sum())
        n_total = len(y_true)

        if n_pos == 0 or n_pos == n_total:
            # Metrics undefined with a single class
            rows.append({
                query_col: query,
                "n_pairs": n_total,
                "n_pos": n_pos,
                "pos_rate": n_pos / n_total if n_total > 0 else 0.0,
                "AUROC": float("nan"),
                "AUPR": float("nan"),
                "AP": float("nan"),
            })
            continue

        try:
            auroc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            auroc = float("nan")
        aupr = float(average_precision_score(y_true, y_score))

        rows.append({
            query_col: query,
            "n_pairs": n_total,
            "n_pos": n_pos,
            "pos_rate": n_pos / n_total,
            "AUROC": auroc,
            "AUPR": aupr,
            "AP": aupr,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="External validation of saved models on a held-out dataset",
    )
    p.add_argument("--data-dir", required=True,
                   help="Directory with external validation TSV files")
    p.add_argument("--n-iters", type=int, default=11,
                   help="Number of TSV iterations to load")
    p.add_argument("--label-col", default="GI_stringent_Type2",
                   help="Binary label column")
    p.add_argument("--model-dir", default="outputs/final",
                   help="Directory containing *_model.pt and *_scaler.pkl")
    p.add_argument("--top-k", type=int, default=100,
                   help="K for Recall@K and Precision@K")
    p.add_argument("--output-dir", default="outputs/external_val")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()

    # ------------------------------------------------------------------
    # 1. Load model config and discover model/scaler pairs
    # ------------------------------------------------------------------
    config = _load_config(model_dir)
    hidden_sizes = _get_hidden_sizes(config)
    dropout = config["best_params"]["dropout"]

    model_pairs = _discover_models(model_dir)
    print(f"Found {len(model_pairs)} model(s) in {model_dir}")
    print(f"Architecture: hidden_sizes={hidden_sizes}, dropout={dropout}")

    # ------------------------------------------------------------------
    # 2. Load and clean external validation data
    # ------------------------------------------------------------------
    print(f"\nLoading external data from {args.data_dir} ...")
    df = load_tsv_iterations(args.data_dir, n_iters=args.n_iters)
    df = filter_unique_pairs(df)
    print(f"Loaded {len(df)} unique gene pairs")

    X, y, feature_cols = _prepare_features(df, args.label_col, NON_FEATURE_COLS)
    print(f"Features: {X.shape[1]}, samples: {len(X)}, "
          f"positives: {int(y.sum())} ({y.mean()*100:.2f}%)")

    # ------------------------------------------------------------------
    # 3. Run each model on the external set
    # ------------------------------------------------------------------
    all_model_metrics: list[dict] = []
    all_probs: list[np.ndarray] = []
    all_query_rows: list[pd.DataFrame] = []

    for model_path, scaler_path in model_pairs:
        tag = model_path.stem.replace("_model", "")
        print(f"\n--- {tag} ---")

        # Load scaler and transform features
        scaler = joblib.load(scaler_path)
        # Align feature columns to what the scaler expects
        n_scaler_features = scaler.n_features_in_
        if X.shape[1] != n_scaler_features:
            # The scaler may have been fit after dropping constant features;
            # we need to select the same subset.
            # Try to use feature names if available
            if hasattr(scaler, "feature_names_in_"):
                keep_cols = [c for c in scaler.feature_names_in_ if c in X.columns]
                X_subset = X[keep_cols]
            else:
                # Fall back: take first n_scaler_features columns
                X_subset = X.iloc[:, :n_scaler_features]
            print(f"  Aligned features: {X.shape[1]} -> {X_subset.shape[1]}")
        else:
            X_subset = X

        X_scaled = scaler.transform(X_subset)

        # Load model
        input_size = X_scaled.shape[1]
        model = SLPredictorMLP(input_size, hidden_sizes, dropout)
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)

        # Predict
        y_score = _predict(model, X_scaled, device)
        all_probs.append(y_score)

        # Compute overall metrics
        y_true = y.values
        metrics = compute_all_metrics(y_true, y_score, top_k=args.top_k)
        metrics["model"] = tag
        all_model_metrics.append(metrics)
        print(f"  AUROC={metrics['AUROC']:.4f}  AUPR={metrics['AUPR']:.4f}  "
              f"AP={metrics['AP']:.4f}")

        # Plot ROC/PR per model
        plot_roc_pr(y_true, y_score,
                    save_path=output_dir / f"{tag}_roc_pr.png")

        # Per-query metrics for this model
        df_pred = df.loc[X.index, ["Query"]].copy()
        df_pred["y_true"] = y_true
        df_pred["y_score"] = y_score
        q_metrics = compute_query_metrics(df_pred, top_k=args.top_k)
        q_metrics["model"] = tag
        all_query_rows.append(q_metrics)

    # ------------------------------------------------------------------
    # 4. Save per-model metrics
    # ------------------------------------------------------------------
    df_models = pd.DataFrame(all_model_metrics)
    front = ["model"]
    rest = [c for c in df_models.columns if c != "model"]
    df_models = df_models[front + rest]
    df_models.to_csv(output_dir / "model_metrics.tsv", sep="\t", index=False)

    # Model-level summary (mean ± SE)
    metric_cols = [c for c in df_models.columns if c != "model"]
    summary = {}
    print(f"\n{'='*60}")
    print(f"  External Validation: {len(model_pairs)} models  (mean ± SE)")
    print(f"{'='*60}")
    for col in metric_cols:
        vals = df_models[col].values
        mean = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        summary[col] = {"mean": mean, "se": se}
        print(f"  {col:>20s}:  {mean:.4f} ± {se:.4f}")

    summary_df = pd.DataFrame([
        {"metric": k, "mean": v["mean"], "se": v["se"]}
        for k, v in summary.items()
    ])
    summary_df.to_csv(output_dir / "model_summary.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------
    # 5. Ensemble predictions (mean probability across models)
    # ------------------------------------------------------------------
    ensemble_probs = np.mean(np.column_stack(all_probs), axis=1)
    df_ensemble = df.loc[X.index, [c for c in NON_FEATURE_COLS if c in df.columns]].copy()
    df_ensemble["predict_proba"] = ensemble_probs
    df_ensemble.to_csv(output_dir / "ensemble_predictions.tsv", sep="\t", index=False)

    ensemble_metrics = compute_all_metrics(y.values, ensemble_probs, top_k=args.top_k)
    print(f"\n  Ensemble AUROC={ensemble_metrics['AUROC']:.4f}  "
          f"AUPR={ensemble_metrics['AUPR']:.4f}  "
          f"AP={ensemble_metrics['AP']:.4f}")
    plot_roc_pr(y.values, ensemble_probs,
                save_path=output_dir / "ensemble_roc_pr.png")

    # ------------------------------------------------------------------
    # 6. Per-query metrics
    # ------------------------------------------------------------------
    # Full granularity: (model, query) table
    df_query_all = pd.concat(all_query_rows, ignore_index=True)
    df_query_all.to_csv(
        output_dir / "query_summary_per_model.tsv", sep="\t", index=False,
    )

    # Averaged across models per query
    query_avg = (
        df_query_all
        .groupby("Query")
        .agg(
            n_pairs=("n_pairs", "first"),
            n_pos=("n_pos", "first"),
            pos_rate=("pos_rate", "first"),
            AUROC_mean=("AUROC", "mean"),
            AUROC_se=("AUROC", lambda x: x.std(ddof=1) / np.sqrt(x.notna().sum()) if x.notna().sum() > 1 else 0.0),
            AUPR_mean=("AUPR", "mean"),
            AUPR_se=("AUPR", lambda x: x.std(ddof=1) / np.sqrt(x.notna().sum()) if x.notna().sum() > 1 else 0.0),
            AP_mean=("AP", "mean"),
            AP_se=("AP", lambda x: x.std(ddof=1) / np.sqrt(x.notna().sum()) if x.notna().sum() > 1 else 0.0),
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
        print(f"    {row['Query']:>12s}  AUROC={row['AUROC_mean']:.4f}  "
              f"AUPR={row['AUPR_mean']:.4f}  n_pos={int(row['n_pos'])}/{int(row['n_pairs'])}")
    print(f"\n  Top 10 hardest queries:")
    for _, row in query_avg.tail(10).iterrows():
        print(f"    {row['Query']:>12s}  AUROC={row['AUROC_mean']:.4f}  "
              f"AUPR={row['AUPR_mean']:.4f}  n_pos={int(row['n_pos'])}/{int(row['n_pairs'])}")

    # ------------------------------------------------------------------
    # 7. Save config
    # ------------------------------------------------------------------
    out_config = {
        "data_dir": args.data_dir,
        "n_iters": args.n_iters,
        "label_col": args.label_col,
        "model_dir": str(model_dir),
        "n_models": len(model_pairs),
        "hidden_sizes": hidden_sizes,
        "dropout": dropout,
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
