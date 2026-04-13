"""Evaluate whether model predictions can guide smarter query selection.

This script tests two hypotheses using the external validation dataset:

1. **SL Discovery**: Selecting queries guided by model predictions finds more
   synthetic lethal (SL) gene pairs than random query selection.
2. **Library Gene Coverage**: Model-guided query selection covers more unique
   library genes with at least one newly identified SL partner.

Simulation
----------
We have a pool of Q query genes (from the external validation set).
An experimenter has budget to screen K queries (K = 1, 2, ..., Q).

- **Random**: pick K queries at random; average over many permutations.
- **Model-guided**: rank queries by a score derived from model predictions
  (e.g. sum of predicted probabilities, or count above a threshold),
  then pick the top K.
- **Oracle**: rank queries by the actual number of SL hits (upper bound).

After selecting K queries, we count:
  (a) total true SL pairs found,
  (b) unique library genes with at least one SL partner found.

Usage
-----
    python -m src.evaluate_query_prioritization \\
        --predictions outputs/external_val/external_val_old_models/ensemble_predictions.tsv \\
        --output-dir outputs/external_val/external_val_old_models/query_prioritization

Outputs
-------
    sl_discovery_curves.tsv          – cumulative SL pairs at each K
    library_coverage_curves.tsv      – cumulative library gene coverage at each K
    sl_discovery.png                 – plot of SL pairs vs queries screened
    library_coverage.png             – plot of gene coverage vs queries screened
    combined_panel.png               – side-by-side panel figure
    query_ranking.tsv                – per-query priority scores and true stats
    prioritization_summary.json      – key statistics and AUC comparisons
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Query scoring strategies
# ---------------------------------------------------------------------------

def score_queries_by_sum(
    df: pd.DataFrame,
    query_col: str = "Query",
    score_col: str = "predict_proba",
) -> pd.Series:
    """Rank queries by the sum of predicted probabilities (expected SL count)."""
    return df.groupby(query_col)[score_col].sum().sort_values(ascending=False)


def score_queries_by_mean(
    df: pd.DataFrame,
    query_col: str = "Query",
    score_col: str = "predict_proba",
) -> pd.Series:
    """Rank queries by the mean predicted probability."""
    return df.groupby(query_col)[score_col].mean().sort_values(ascending=False)


def score_queries_by_top_hits(
    df: pd.DataFrame,
    threshold: float = 0.5,
    query_col: str = "Query",
    score_col: str = "predict_proba",
) -> pd.Series:
    """Rank queries by the number of pairs predicted above *threshold*."""
    return (
        df.assign(_hit=(df[score_col] >= threshold).astype(int))
        .groupby(query_col)["_hit"]
        .sum()
        .sort_values(ascending=False)
    )


def score_queries_by_top_percentile(
    df: pd.DataFrame,
    top_n: int = 100,
    query_col: str = "Query",
    score_col: str = "predict_proba",
) -> pd.Series:
    """Rank queries by the sum of their top-*top_n* predicted probabilities."""
    def _top_sum(grp):
        return grp.nlargest(min(top_n, len(grp))).sum()

    return (
        df.groupby(query_col)[score_col]
        .apply(_top_sum)
        .sort_values(ascending=False)
    )


# ---------------------------------------------------------------------------
# Oracle ranking (ground truth upper bound)
# ---------------------------------------------------------------------------

def oracle_query_order(
    df: pd.DataFrame,
    label_col: str,
    query_col: str = "Query",
    gene_col: str = "Gene",
    metric: str = "sl_count",
) -> list[str]:
    """Return queries ordered by true SL count (greedy oracle).

    Parameters
    ----------
    metric : str
        ``"sl_count"`` – sort by number of true SL pairs.
        ``"gene_coverage_greedy"`` – greedy set-cover: at each step,
        pick the query that adds the most new library genes.
    """
    assert metric in ("sl_count", "gene_coverage_greedy")
    sl = df[df[label_col] == 1]

    if metric == "sl_count":
        counts = sl.groupby(query_col).size().sort_values(ascending=False)
        # Include queries with 0 SL hits at the end
        all_queries = df[query_col].unique()
        order = list(counts.index)
        remaining = [q for q in all_queries if q not in order]
        np.random.shuffle(remaining)
        return order + remaining

    # Greedy set-cover for gene_coverage
    sl_genes_per_query = sl.groupby(query_col)[gene_col].apply(set).to_dict()
    all_queries = list(df[query_col].unique())
    covered: set[str] = set()
    order: list[str] = []
    remaining_q = set(all_queries)

    while remaining_q:
        best_q = None
        best_gain = -1
        for q in remaining_q:
            new_genes = sl_genes_per_query.get(q, set()) - covered
            if len(new_genes) > best_gain:
                best_gain = len(new_genes)
                best_q = q
        assert best_q is not None
        order.append(best_q)
        covered |= sl_genes_per_query.get(best_q, set())
        remaining_q.remove(best_q)

    return order


# ---------------------------------------------------------------------------
# Cumulative statistics given an order of queries
# ---------------------------------------------------------------------------

def cumulative_sl_pairs(
    query_order: list[str],
    sl_per_query: dict[str, int],
) -> np.ndarray:
    """Return array of length len(query_order)+1: cumulative SL pairs at k=0..Q."""
    cum = np.zeros(len(query_order) + 1, dtype=int)
    for i, q in enumerate(query_order):
        cum[i + 1] = cum[i] + sl_per_query.get(q, 0)
    return cum


def cumulative_gene_coverage(
    query_order: list[str],
    sl_genes_per_query: dict[str, set],
) -> np.ndarray:
    """Return array of cumulative unique library genes covered at k=0..Q."""
    cum = np.zeros(len(query_order) + 1, dtype=int)
    covered: set[str] = set()
    for i, q in enumerate(query_order):
        covered |= sl_genes_per_query.get(q, set())
        cum[i + 1] = len(covered)
    return cum


# ---------------------------------------------------------------------------
# Random baseline (permutation)
# ---------------------------------------------------------------------------

def random_baseline(
    all_queries: list[str],
    sl_per_query: dict[str, int],
    sl_genes_per_query: dict[str, set],
    n_permutations: int = 10_000,
    seed: int = 42,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute random baseline with mean and confidence intervals.

    Returns
    -------
    dict with keys "sl_pairs" and "gene_coverage", each containing:
        "mean", "ci_lo", "ci_hi" – arrays of length Q+1.
    """
    rng = np.random.default_rng(seed)
    Q = len(all_queries)
    sl_matrix = np.zeros((n_permutations, Q + 1), dtype=int)
    cov_matrix = np.zeros((n_permutations, Q + 1), dtype=int)

    for p in range(n_permutations):
        perm = rng.permutation(all_queries)
        sl_matrix[p] = cumulative_sl_pairs(list(perm), sl_per_query)
        cov_matrix[p] = cumulative_gene_coverage(list(perm), sl_genes_per_query)

    def _summarize(mat):
        return {
            "mean": mat.mean(axis=0),
            "ci_lo": np.percentile(mat, 2.5, axis=0),
            "ci_hi": np.percentile(mat, 97.5, axis=0),
        }

    return {
        "sl_pairs": _summarize(sl_matrix),
        "gene_coverage": _summarize(cov_matrix),
    }


# ---------------------------------------------------------------------------
# Area-under-curve comparison metric
# ---------------------------------------------------------------------------

def normalized_auc(curve: np.ndarray) -> float:
    """Compute AUC of a curve normalized to [0, 1] on both axes.

    The curve has Q+1 points (at k=0, 1, ..., Q). We normalize x by Q
    and y by y[-1] (the maximum achievable value at full screening).
    Returns a value in [0, 1].  A perfect oracle scores close to 1;
    random is ~0.5.
    """
    Q = len(curve) - 1
    y_max = curve[-1]
    if y_max == 0 or Q == 0:
        return 0.0
    x_norm = np.arange(Q + 1) / Q
    y_norm = curve / y_max
    return float(np.trapezoid(y_norm, x_norm))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_prioritization_curve(
    ax: plt.Axes,
    ks: np.ndarray,
    curves: dict[str, np.ndarray],
    random_stats: dict[str, np.ndarray],
    ylabel: str,
    title: str,
    total_queries: int,
    y_total: int | None = None,
) -> None:
    """Plot one panel of the prioritization figure."""
    # Random baseline
    ax.fill_between(
        ks,
        random_stats["ci_lo"],
        random_stats["ci_hi"],
        alpha=0.15,
        color="grey",
        label="Random 95% CI",
    )
    ax.plot(ks, random_stats["mean"], color="grey", ls="--", lw=1.5, label="Random")

    # Model-guided strategies
    colors = {"sum_proba": "#1f77b4", "mean_proba": "#ff7f0e",
              "top100_sum": "#2ca02c",
              "hits_baserate": "#9467bd"}
    labels = {"sum_proba": "Guided (Σ pred_prob)",
              "mean_proba": "Guided (mean pred_prob)",
              "top100_sum": "Guided (Σ top-100 pred_prob)",
              "hits_baserate": "Guided (# pred > base-rate cutoff)"}
    for name, curve in curves.items():
        if name.startswith("oracle"):
            continue
        c = colors.get(name, "blue")
        ax.plot(ks, curve, color=c, lw=2, label=labels.get(name, name))

    # Oracle
    for name, curve in curves.items():
        if name.startswith("oracle"):
            label = "Oracle (SL count)" if "sl" in name else "Oracle (greedy coverage)"
            ax.plot(ks, curve, color="black", ls=":", lw=1.5, label=label)

    ax.set_xlabel("Number of queries screened")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if y_total is not None:
        ax.axhline(y=y_total, color="black", ls="-", lw=0.5, alpha=0.3)
    ax.set_xlim(0, total_queries)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate model-guided query prioritization vs random",
    )
    p.add_argument(
        "--predictions",
        default="outputs/external_val/external_val_old_models/ensemble_predictions.tsv",
        help="Ensemble predictions TSV (with Gene, Query, label, predict_proba)",
    )
    p.add_argument("--label-col", default="GI_stringent_Type2")
    p.add_argument("--gene-col", default="Gene")
    p.add_argument("--query-col", default="Query")
    p.add_argument("--score-col", default="predict_proba")
    p.add_argument("--n-permutations", type=int, default=10_000,
                   help="Number of random permutations for baseline")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        default="outputs/external_val/external_val_old_models/query_prioritization",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load ensemble predictions
    # ------------------------------------------------------------------
    print(f"Loading predictions from {args.predictions} ...")
    df = pd.read_csv(args.predictions, sep="\t")
    print(f"  {len(df):,} gene pairs, {df[args.query_col].nunique()} queries, "
          f"{df[args.gene_col].nunique():,} library genes")

    # Positive (SL) pairs
    sl = df[df[args.label_col] == 1]
    total_sl = len(sl)
    print(f"  {total_sl:,} true SL pairs")

    all_queries = sorted(df[args.query_col].unique())
    Q = len(all_queries)

    # Per-query ground truth
    sl_per_query: dict[str, int] = sl.groupby(args.query_col).size().to_dict()
    sl_genes_per_query: dict[str, set] = (
        sl.groupby(args.query_col)[args.gene_col].apply(set).to_dict()
    )
    total_sl_genes = len(sl[args.gene_col].unique())
    print(f"  {total_sl_genes:,} unique library genes with ≥1 SL partner")

    # ------------------------------------------------------------------
    # 2. Compute model-guided query rankings
    # ------------------------------------------------------------------
    print("\nComputing query priority scores ...")
    rankings: dict[str, list[str]] = {}

    # Strategy 1: sum of predicted probabilities
    s = score_queries_by_sum(df, args.query_col, args.score_col)
    rankings["sum_proba"] = list(s.index)

    # Strategy 2: mean predicted probability
    s = score_queries_by_mean(df, args.query_col, args.score_col)
    rankings["mean_proba"] = list(s.index)

    # Strategy 3: sum of top-100 predicted probabilities
    s = score_queries_by_top_percentile(df, top_n=100,
                                        query_col=args.query_col,
                                        score_col=args.score_col)
    rankings["top100_sum"] = list(s.index)

    # Strategy 4: count predicted above base-rate-calibrated threshold
    #   Choose the cutoff so that the fraction of predicted positives
    #   matches the true SL prevalence in the external validation set.
    true_pos_rate = total_sl / len(df)
    calibrated_cutoff = float(np.quantile(df[args.score_col], 1 - true_pos_rate))
    print(f"  True SL rate = {true_pos_rate*100:.4f}%  "
          f"-> calibrated cutoff = {calibrated_cutoff:.6f}")
    n_pred_pos = int((df[args.score_col] >= calibrated_cutoff).sum())
    print(f"  # predicted positives at cutoff = {n_pred_pos:,} "
          f"(vs {total_sl:,} true SL)")
    s = score_queries_by_top_hits(df, threshold=calibrated_cutoff,
                                  query_col=args.query_col,
                                  score_col=args.score_col)
    rankings["hits_baserate"] = list(s.index)

    # Oracle rankings
    rankings["oracle_sl"] = oracle_query_order(
        df, args.label_col, args.query_col, args.gene_col, metric="sl_count"
    )
    rankings["oracle_cov"] = oracle_query_order(
        df, args.label_col, args.query_col, args.gene_col,
        metric="gene_coverage_greedy",
    )

    # ------------------------------------------------------------------
    # 3. Compute cumulative curves
    # ------------------------------------------------------------------
    print("Computing cumulative curves ...")
    ks = np.arange(Q + 1)

    sl_curves: dict[str, np.ndarray] = {}
    cov_curves: dict[str, np.ndarray] = {}

    for name, order in rankings.items():
        sl_curves[name] = cumulative_sl_pairs(order, sl_per_query)
        cov_curves[name] = cumulative_gene_coverage(order, sl_genes_per_query)

    # ------------------------------------------------------------------
    # 4. Random baseline
    # ------------------------------------------------------------------
    print(f"Running {args.n_permutations:,} random permutations ...")
    rand = random_baseline(
        all_queries, sl_per_query, sl_genes_per_query,
        n_permutations=args.n_permutations, seed=args.seed,
    )

    # ------------------------------------------------------------------
    # 5. Save curves to TSV
    # ------------------------------------------------------------------
    # SL discovery
    sl_df = pd.DataFrame({"k": ks})
    sl_df["random_mean"] = rand["sl_pairs"]["mean"]
    sl_df["random_ci_lo"] = rand["sl_pairs"]["ci_lo"]
    sl_df["random_ci_hi"] = rand["sl_pairs"]["ci_hi"]
    for name in rankings:
        sl_df[name] = sl_curves[name]
    sl_df.to_csv(output_dir / "sl_discovery_curves.tsv", sep="\t", index=False)

    # Library coverage
    cov_df = pd.DataFrame({"k": ks})
    cov_df["random_mean"] = rand["gene_coverage"]["mean"]
    cov_df["random_ci_lo"] = rand["gene_coverage"]["ci_lo"]
    cov_df["random_ci_hi"] = rand["gene_coverage"]["ci_hi"]
    for name in rankings:
        cov_df[name] = cov_curves[name]
    cov_df.to_csv(output_dir / "library_coverage_curves.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------
    # 6. Per-query ranking table
    # ------------------------------------------------------------------
    sum_scores = score_queries_by_sum(df, args.query_col, args.score_col)
    mean_scores = score_queries_by_mean(df, args.query_col, args.score_col)
    top100_scores = score_queries_by_top_percentile(
        df, top_n=100, query_col=args.query_col, score_col=args.score_col,
    )

    query_df = pd.DataFrame({
        args.query_col: all_queries,
    }).set_index(args.query_col)
    query_df["n_true_sl"] = pd.Series(sl_per_query)
    query_df["n_true_sl_genes"] = pd.Series({
        q: len(genes) for q, genes in sl_genes_per_query.items()
    })
    query_df["score_sum_proba"] = sum_scores
    query_df["score_mean_proba"] = mean_scores
    query_df["score_top100_sum"] = top100_scores
    query_df = query_df.fillna(0).sort_values("score_sum_proba", ascending=False)
    query_df.to_csv(output_dir / "query_ranking.tsv", sep="\t")

    # Spearman correlation between model score and true SL count
    from scipy.stats import spearmanr
    rho, pval = spearmanr(query_df["score_sum_proba"], query_df["n_true_sl"])
    print(f"\n  Spearman(sum_proba, n_true_sl) = {rho:.4f}  (p = {pval:.2e})")

    # ------------------------------------------------------------------
    # 7. Compute normalized AUC for each strategy
    # ------------------------------------------------------------------
    print("\nNormalized AUC (higher = better prioritization):")
    auc_results: dict[str, dict[str, float]] = {}
    for name in rankings:
        sl_auc = normalized_auc(sl_curves[name])
        cov_auc = normalized_auc(cov_curves[name])
        auc_results[name] = {"sl_discovery_nAUC": sl_auc,
                             "gene_coverage_nAUC": cov_auc}
        print(f"  {name:>20s}:  SL nAUC={sl_auc:.4f}  Coverage nAUC={cov_auc:.4f}")

    rand_sl_auc = normalized_auc(rand["sl_pairs"]["mean"])
    rand_cov_auc = normalized_auc(rand["gene_coverage"]["mean"])
    auc_results["random"] = {"sl_discovery_nAUC": rand_sl_auc,
                             "gene_coverage_nAUC": rand_cov_auc}
    print(f"  {'random':>20s}:  SL nAUC={rand_sl_auc:.4f}  Coverage nAUC={rand_cov_auc:.4f}")

    # ------------------------------------------------------------------
    # 8. Key milestone comparisons
    # ------------------------------------------------------------------
    milestones = [10, 20, 30, 50]
    print("\n--- SL pairs found at key milestones ---")
    milestone_results: dict[int, dict[str, float]] = {}
    for k in milestones:
        if k > Q:
            continue
        row: dict[str, float] = {"random": float(rand["sl_pairs"]["mean"][k])}
        for name in rankings:
            if not name.startswith("oracle"):
                row[name] = int(sl_curves[name][k])
        row["oracle_sl"] = int(sl_curves["oracle_sl"][k])
        milestone_results[k] = row
        parts = [f"  K={k:3d}: "]
        for strategy, val in row.items():
            parts.append(f"{strategy}={val:.0f}")
        print("  ".join(parts))

    # ------------------------------------------------------------------
    # 9. Plots
    # ------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid")

    # Individual SL discovery plot
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_prioritization_curve(
        ax, ks,
        {k: v for k, v in sl_curves.items()},
        rand["sl_pairs"],
        ylabel="Cumulative SL pairs found",
        title="Model-Guided vs Random Query Selection:\nSynthetic Lethal Discovery",
        total_queries=Q,
        y_total=total_sl,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "sl_discovery.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Individual library coverage plot
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_prioritization_curve(
        ax, ks,
        {k: v for k, v in cov_curves.items()},
        rand["gene_coverage"],
        ylabel="Unique library genes covered",
        title="Model-Guided vs Random Query Selection:\nLibrary Gene Coverage",
        total_queries=Q,
        y_total=total_sl_genes,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "library_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Combined panel
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_prioritization_curve(
        axes[0], ks,
        {k: v for k, v in sl_curves.items()},
        rand["sl_pairs"],
        ylabel="Cumulative SL pairs found",
        title="SL Pair Discovery",
        total_queries=Q,
        y_total=total_sl,
    )
    plot_prioritization_curve(
        axes[1], ks,
        {k: v for k, v in cov_curves.items()},
        rand["gene_coverage"],
        ylabel="Unique library genes covered",
        title="Library Gene Coverage",
        total_queries=Q,
        y_total=total_sl_genes,
    )
    fig.suptitle("Query Prioritization: Model-Guided vs Random Selection",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "combined_panel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlots saved to {output_dir}/")

    # ------------------------------------------------------------------
    # 10. Save summary JSON
    # ------------------------------------------------------------------
    summary = {
        "predictions_file": args.predictions,
        "label_col": args.label_col,
        "n_queries": Q,
        "n_gene_pairs": len(df),
        "n_sl_pairs": total_sl,
        "n_sl_library_genes": total_sl_genes,
        "true_sl_rate": true_pos_rate,
        "calibrated_cutoff": calibrated_cutoff,
        "n_permutations": args.n_permutations,
        "seed": args.seed,
        "spearman_sum_proba_vs_true_sl": {"rho": float(rho), "p_value": float(pval)},
        "normalized_auc": auc_results,
        "milestones_sl_pairs": {str(k): v for k, v in milestone_results.items()},
    }
    with open(output_dir / "prioritization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs saved to {output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
