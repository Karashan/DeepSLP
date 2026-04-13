"""Extended evaluations for model-guided query prioritization.

Six additional analyses beyond the basic prioritization curves:

1. **Pair-level precision at budget** – Among the top-K model-ranked gene
   pairs (globally), what fraction are true SL?  Compared to the random
   base rate.
2. **Discovery of novel SL hubs** – Does model-guided selection preferentially
   recover hub library genes (SL with ≥N queries)?
3. **Efficiency under a fixed discovery budget** – How many queries are needed
   to reach various SL-pair or coverage milestones?
4. **Enrichment in rare SL genes** – Does model guidance
   disproportionately help recover library genes that are SL with only
   1–2 queries (easy to miss by random screening)?
5. **Per-model variability** – Run query prioritization separately with each
   of the 10 individual model predictions.  Reports the distribution of
   nAUC across models.
6. **Stratified analysis by query difficulty** – Split queries into easy
   (high AUROC) vs hard (low AUROC) and measure prioritization benefit
   within each stratum.

Usage
-----
    python -m src.evaluate_extended \\
        --predictions outputs/external_val/external_val_old_models/ensemble_predictions.tsv \\
        --query-metrics outputs/external_val/external_val_old_models/query_metrics.tsv \\
        --output-dir outputs/external_val/external_val_old_models/extended_eval

Outputs go into per-test subdirectories under ``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .evaluate_query_prioritization import (
    cumulative_gene_coverage,
    cumulative_sl_pairs,
    normalized_auc,
    score_queries_by_sum,
)


# ===================================================================
# Shared helpers
# ===================================================================

def _load_ensemble(path: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    return df


def _sl_stats(df, label_col, query_col, gene_col):
    """Compute frequently used SL statistics."""
    sl = df[df[label_col] == 1]
    sl_per_query = sl.groupby(query_col).size().to_dict()
    sl_genes_per_query = sl.groupby(query_col)[gene_col].apply(set).to_dict()
    # How many queries each library gene is SL with
    gene_query_counts = sl.groupby(gene_col)[query_col].nunique().to_dict()
    return sl, sl_per_query, sl_genes_per_query, gene_query_counts


# ===================================================================
# Test 1: Pair-level precision at budget
# ===================================================================

def test1_pair_precision(
    df: pd.DataFrame,
    label_col: str,
    score_col: str,
    output_dir: Path,
    seed: int = 42,
) -> dict:
    """Among top-K model-ranked gene pairs, what fraction are true SL?"""
    print("\n" + "=" * 70)
    print("  TEST 1: Pair-Level Precision at Budget")
    print("=" * 70)
    os.makedirs(output_dir, exist_ok=True)

    y_true = df[label_col].values
    y_score = df[score_col].values
    n_total = len(df)
    n_pos = int(y_true.sum())
    base_rate = n_pos / n_total

    # Sort by predicted probability (descending)
    order = np.argsort(y_score)[::-1]

    # Compute precision at various K values
    ks = np.unique(np.concatenate([
        np.arange(100, 1001, 100),
        np.arange(1000, 5001, 500),
        np.arange(5000, 20001, 1000),
        np.arange(20000, min(50001, n_total + 1), 5000),
        [n_pos],  # K = number of true SL pairs
    ]))
    ks = np.sort(ks[ks <= n_total])

    rows = []
    for k in ks:
        top_k_idx = order[:k]
        tp = int(y_true[top_k_idx].sum())
        precision = tp / k
        recall = tp / n_pos if n_pos > 0 else 0.0
        lift = precision / base_rate if base_rate > 0 else 0.0
        rows.append({
            "K": int(k),
            "true_positives": tp,
            "precision": precision,
            "recall": recall,
            "lift_over_random": lift,
            "random_precision": base_rate,
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "pair_precision_at_k.tsv", sep="\t", index=False)

    # Print key results
    for _, r in results_df.iterrows():
        if r["K"] in [100, 500, 1000, n_pos, 5000, 10000]:
            print(f"  K={int(r['K']):>6,}:  Precision={r['precision']:.4f}  "
                  f"Recall={r['recall']:.4f}  Lift={r['lift_over_random']:.1f}x  "
                  f"(TP={int(r['true_positives'])})")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Precision@K
    axes[0].plot(results_df["K"], results_df["precision"], "b-", lw=2,
                 label="Model")
    axes[0].axhline(y=base_rate, color="grey", ls="--", lw=1.5,
                    label=f"Random ({base_rate:.5f})")
    axes[0].set_xlabel("K (pairs examined)")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision@K")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Lift over random
    axes[1].plot(results_df["K"], results_df["lift_over_random"], "r-", lw=2)
    axes[1].axhline(y=1.0, color="grey", ls="--", lw=1.5, label="Random (1x)")
    axes[1].set_xlabel("K (pairs examined)")
    axes[1].set_ylabel("Lift (precision / base rate)")
    axes[1].set_title("Lift Over Random")
    axes[1].set_xscale("log")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Recall@K
    axes[2].plot(results_df["K"], results_df["recall"], "g-", lw=2,
                 label="Model")
    # Random recall at K:  K * base_rate / n_pos = K / n_total
    random_recall = results_df["K"].values / n_total
    axes[2].plot(results_df["K"], random_recall, "grey", ls="--", lw=1.5,
                 label="Random")
    axes[2].set_xlabel("K (pairs examined)")
    axes[2].set_ylabel("Recall")
    axes[2].set_title("Recall@K")
    axes[2].set_xscale("log")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Pair-Level Precision: Model Ranking vs Random",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "pair_precision.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Summary stats
    k_eq_npos = results_df[results_df["K"] == n_pos].iloc[0]
    summary = {
        "n_total_pairs": n_total,
        "n_true_sl": n_pos,
        "base_rate": base_rate,
        "precision_at_100": float(results_df[results_df["K"] == 100]["precision"].iloc[0]),
        "precision_at_1000": float(results_df[results_df["K"] == 1000]["precision"].iloc[0]),
        f"precision_at_{n_pos}": float(k_eq_npos["precision"]),
        f"recall_at_{n_pos}": float(k_eq_npos["recall"]),
        f"lift_at_{n_pos}": float(k_eq_npos["lift_over_random"]),
    }
    print(f"\n  At K={n_pos} (same budget as # true SL):")
    print(f"    Precision = {k_eq_npos['precision']:.4f}  "
          f"(vs random {base_rate:.5f})")
    print(f"    Recall    = {k_eq_npos['recall']:.4f}")
    print(f"    Lift      = {k_eq_npos['lift_over_random']:.1f}x")

    return summary


# ===================================================================
# Test 2: Discovery of novel SL hubs
# ===================================================================

def test2_sl_hub_discovery(
    df: pd.DataFrame,
    label_col: str,
    score_col: str,
    query_col: str,
    gene_col: str,
    output_dir: Path,
    seed: int = 42,
) -> dict:
    """Does model-guided selection preferentially discover hub genes?"""
    print("\n" + "=" * 70)
    print("  TEST 2: Discovery of Novel SL Hubs")
    print("=" * 70)
    os.makedirs(output_dir, exist_ok=True)

    sl, sl_per_query, sl_genes_per_query, gene_query_counts = _sl_stats(
        df, label_col, query_col, gene_col
    )
    all_queries = sorted(df[query_col].unique())
    Q = len(all_queries)

    # Classify library genes by how many queries they're SL with
    hub_thresholds = [1, 2, 3, 5, 10]
    hub_genes = {}
    for t in hub_thresholds:
        hub_genes[t] = {g for g, c in gene_query_counts.items() if c >= t}
        print(f"  Genes with ≥{t} SL query partners: {len(hub_genes[t])}")

    # Model-guided order (Σ pred_prob)
    s = score_queries_by_sum(df, query_col, score_col)
    guided_order = list(s.index)

    # Compute cumulative hub discovery curves
    rng = np.random.default_rng(seed)
    n_perm = 5_000

    results = {}
    for t in hub_thresholds:
        target_genes = hub_genes[t]
        if not target_genes:
            continue

        # For each query, which target hub genes does it discover?
        hub_per_query = {}
        for q, genes in sl_genes_per_query.items():
            hub_per_query[q] = genes & target_genes

        # Guided curve
        guided_cum = np.zeros(Q + 1, dtype=int)
        covered = set()
        for i, q in enumerate(guided_order):
            covered |= hub_per_query.get(q, set())
            guided_cum[i + 1] = len(covered)

        # Random curves
        rand_mat = np.zeros((n_perm, Q + 1), dtype=int)
        for p in range(n_perm):
            perm = list(rng.permutation(all_queries))
            covered_r = set()
            for i, q in enumerate(perm):
                covered_r |= hub_per_query.get(q, set())
                rand_mat[p, i + 1] = len(covered_r)

        rand_mean = rand_mat.mean(axis=0)
        rand_sd = rand_mat.std(axis=0, ddof=1)

        guided_nauc = normalized_auc(guided_cum)
        rand_nauc = normalized_auc(rand_mean)

        results[t] = {
            "n_target_genes": len(target_genes),
            "guided_nauc": guided_nauc,
            "random_nauc": rand_nauc,
            "guided_cum": guided_cum,
            "rand_mean": rand_mean,
            "rand_sd_lo": rand_mean - rand_sd,
            "rand_sd_hi": rand_mean + rand_sd,
        }
        print(f"  Hub≥{t}: guided nAUC={guided_nauc:.4f}  "
              f"random nAUC={rand_nauc:.4f}  "
              f"improvement={guided_nauc - rand_nauc:+.4f}")

    # Plot
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5),
                             squeeze=False)
    ks = np.arange(Q + 1)
    for idx, (t, res) in enumerate(results.items()):
        ax = axes[0, idx]
        ax.fill_between(ks, res["rand_sd_lo"], res["rand_sd_hi"],
                        alpha=0.15, color="grey", label="Random ± 1 SD")
        ax.plot(ks, res["rand_mean"], "grey", ls="--", lw=1.5, label="Random")
        ax.plot(ks, res["guided_cum"], "#1f77b4", lw=2,
                label="Guided (Σ pred_prob)")
        ax.axhline(y=res["n_target_genes"], color="black", ls="-",
                   lw=0.5, alpha=0.3)
        ax.set_xlabel("Queries screened")
        ax.set_ylabel(f"Hub genes (≥{t} SL partners) found")
        ax.set_title(f"Hubs ≥{t} partners (n={res['n_target_genes']})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Hub Gene Discovery: Model-Guided vs Random",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "hub_discovery.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save table
    rows = []
    for t, res in results.items():
        rows.append({
            "hub_threshold": t,
            "n_hub_genes": res["n_target_genes"],
            "guided_nAUC": res["guided_nauc"],
            "random_nAUC": res["random_nauc"],
            "nAUC_improvement": res["guided_nauc"] - res["random_nauc"],
        })
    pd.DataFrame(rows).to_csv(output_dir / "hub_discovery.tsv",
                               sep="\t", index=False)

    return {r["hub_threshold"]: {k: v for k, v in r.items() if k != "hub_threshold"}
            for r in rows}


# ===================================================================
# Test 3: Efficiency under a fixed discovery budget
# ===================================================================

def test3_efficiency_budget(
    df: pd.DataFrame,
    label_col: str,
    score_col: str,
    query_col: str,
    gene_col: str,
    output_dir: Path,
    seed: int = 42,
) -> dict:
    """How many queries needed to reach various SL-pair / coverage targets?"""
    print("\n" + "=" * 70)
    print("  TEST 3: Efficiency Under a Fixed Discovery Budget")
    print("=" * 70)
    os.makedirs(output_dir, exist_ok=True)

    sl, sl_per_query, sl_genes_per_query, _ = _sl_stats(
        df, label_col, query_col, gene_col
    )
    all_queries = sorted(df[query_col].unique())
    Q = len(all_queries)
    total_sl = len(sl)
    total_sl_genes = len(sl[gene_col].unique())

    # Model-guided order
    s = score_queries_by_sum(df, query_col, score_col)
    guided_order = list(s.index)

    guided_sl_cum = cumulative_sl_pairs(guided_order, sl_per_query)
    guided_cov_cum = cumulative_gene_coverage(guided_order, sl_genes_per_query)

    # Random baseline (5000 permutations)
    rng = np.random.default_rng(seed)
    n_perm = 5_000
    rand_sl_mat = np.zeros((n_perm, Q + 1), dtype=int)
    rand_cov_mat = np.zeros((n_perm, Q + 1), dtype=int)
    for p in range(n_perm):
        perm = list(rng.permutation(all_queries))
        rand_sl_mat[p] = cumulative_sl_pairs(perm, sl_per_query)
        rand_cov_mat[p] = cumulative_gene_coverage(perm, sl_genes_per_query)

    def _queries_to_reach(curve, target):
        """First K where curve[K] >= target."""
        idx = np.where(curve >= target)[0]
        return int(idx[0]) if len(idx) > 0 else Q + 1

    # --- SL pair milestones ---
    sl_targets_pct = [25, 50, 75, 90]
    sl_rows = []
    print(f"\n  SL Pair Discovery (total = {total_sl}):")
    for pct in sl_targets_pct:
        target = int(np.ceil(total_sl * pct / 100))
        guided_k = _queries_to_reach(guided_sl_cum, target)
        rand_ks = np.array([_queries_to_reach(rand_sl_mat[p], target)
                            for p in range(n_perm)])
        rand_mean = float(np.mean(rand_ks))
        rand_median = float(np.median(rand_ks))
        rand_sd = float(np.std(rand_ks, ddof=1))
        savings = rand_mean - guided_k
        savings_pct = savings / rand_mean * 100 if rand_mean > 0 else 0

        sl_rows.append({
            "target_pct": pct,
            "target_sl_pairs": target,
            "guided_queries": guided_k,
            "random_mean_queries": rand_mean,
            "random_median_queries": rand_median,
            "random_sd": rand_sd,
            "queries_saved": savings,
            "savings_pct": savings_pct,
        })
        print(f"  {pct}% ({target} pairs):  Guided={guided_k}  "
              f"Random={rand_mean:.1f} ± {rand_sd:.1f}  "
              f"Savings={savings:.1f} queries ({savings_pct:.1f}%)")

    # --- Coverage milestones ---
    cov_targets_pct = [25, 50, 75, 90]
    cov_rows = []
    print(f"\n  Library Gene Coverage (total = {total_sl_genes}):")
    for pct in cov_targets_pct:
        target = int(np.ceil(total_sl_genes * pct / 100))
        guided_k = _queries_to_reach(guided_cov_cum, target)
        rand_ks = np.array([_queries_to_reach(rand_cov_mat[p], target)
                            for p in range(n_perm)])
        rand_mean = float(np.mean(rand_ks))
        rand_median = float(np.median(rand_ks))
        rand_sd = float(np.std(rand_ks, ddof=1))
        savings = rand_mean - guided_k
        savings_pct = savings / rand_mean * 100 if rand_mean > 0 else 0

        cov_rows.append({
            "target_pct": pct,
            "target_genes": target,
            "guided_queries": guided_k,
            "random_mean_queries": rand_mean,
            "random_median_queries": rand_median,
            "random_sd": rand_sd,
            "queries_saved": savings,
            "savings_pct": savings_pct,
        })
        print(f"  {pct}% ({target} genes):  Guided={guided_k}  "
              f"Random={rand_mean:.1f} ± {rand_sd:.1f}  "
              f"Savings={savings:.1f} queries ({savings_pct:.1f}%)")

    pd.DataFrame(sl_rows).to_csv(output_dir / "efficiency_sl_pairs.tsv",
                                  sep="\t", index=False)
    pd.DataFrame(cov_rows).to_csv(output_dir / "efficiency_coverage.tsv",
                                   sep="\t", index=False)

    # Plot: bar chart comparing guided vs random for each milestone
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # SL pairs
    x = np.arange(len(sl_targets_pct))
    g_vals = [r["guided_queries"] for r in sl_rows]
    r_vals = [r["random_mean_queries"] for r in sl_rows]
    r_err_lo = [r["random_sd"] for r in sl_rows]
    r_err_hi = [r["random_sd"] for r in sl_rows]
    w = 0.35
    axes[0].bar(x - w / 2, g_vals, w, label="Guided", color="#1f77b4")
    axes[0].bar(x + w / 2, r_vals, w, label="Random",
                color="grey", yerr=[r_err_lo, r_err_hi],
                capsize=4, error_kw={"lw": 1})
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{p}%\n({sl_rows[i]['target_sl_pairs']})"
                             for i, p in enumerate(sl_targets_pct)])
    axes[0].set_xlabel("Target (% of total SL pairs)")
    axes[0].set_ylabel("# Queries needed")
    axes[0].set_title("Queries to Reach SL-Pair Milestones")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Coverage
    g_vals_c = [r["guided_queries"] for r in cov_rows]
    r_vals_c = [r["random_mean_queries"] for r in cov_rows]
    r_err_lo_c = [r["random_sd"] for r in cov_rows]
    r_err_hi_c = [r["random_sd"] for r in cov_rows]
    axes[1].bar(x - w / 2, g_vals_c, w, label="Guided", color="#1f77b4")
    axes[1].bar(x + w / 2, r_vals_c, w, label="Random",
                color="grey", yerr=[r_err_lo_c, r_err_hi_c],
                capsize=4, error_kw={"lw": 1})
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{p}%\n({cov_rows[i]['target_genes']})"
                             for i, p in enumerate(cov_targets_pct)])
    axes[1].set_xlabel("Target (% of total library genes)")
    axes[1].set_ylabel("# Queries needed")
    axes[1].set_title("Queries to Reach Coverage Milestones")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Screening Efficiency: Guided vs Random",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "efficiency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "sl_pairs": sl_rows,
        "gene_coverage": cov_rows,
    }


# ===================================================================
# Test 4: Enrichment in rare SL genes
# ===================================================================

def test4_rare_gene_enrichment(
    df: pd.DataFrame,
    label_col: str,
    score_col: str,
    query_col: str,
    gene_col: str,
    output_dir: Path,
    seed: int = 42,
) -> dict:
    """Does model guidance help more for rare SL genes (hard to find)?"""
    print("\n" + "=" * 70)
    print("  TEST 4: Enrichment in Rare SL Genes")
    print("=" * 70)
    os.makedirs(output_dir, exist_ok=True)

    sl, sl_per_query, sl_genes_per_query, gene_query_counts = _sl_stats(
        df, label_col, query_col, gene_col
    )
    all_queries = sorted(df[query_col].unique())
    Q = len(all_queries)

    # Partition library genes: "rare" (SL with 1-2 queries) vs "common" (≥3)
    rare_genes = {g for g, c in gene_query_counts.items() if c <= 2}
    common_genes = {g for g, c in gene_query_counts.items() if c >= 3}
    print(f"  Rare SL genes (1-2 query partners): {len(rare_genes)}")
    print(f"  Common SL genes (≥3 query partners): {len(common_genes)}")

    # Model-guided order
    s = score_queries_by_sum(df, query_col, score_col)
    guided_order = list(s.index)

    # Compute separate cumulative curves for rare vs common
    def _gene_subset_coverage(query_order, target_genes):
        cum = np.zeros(len(query_order) + 1, dtype=int)
        covered = set()
        for i, q in enumerate(query_order):
            new = sl_genes_per_query.get(q, set()) & target_genes
            covered |= new
            cum[i + 1] = len(covered)
        return cum

    guided_rare = _gene_subset_coverage(guided_order, rare_genes)
    guided_common = _gene_subset_coverage(guided_order, common_genes)

    rng = np.random.default_rng(seed)
    n_perm = 5_000
    rand_rare_mat = np.zeros((n_perm, Q + 1), dtype=int)
    rand_common_mat = np.zeros((n_perm, Q + 1), dtype=int)
    for p in range(n_perm):
        perm = list(rng.permutation(all_queries))
        rand_rare_mat[p] = _gene_subset_coverage(perm, rare_genes)
        rand_common_mat[p] = _gene_subset_coverage(perm, common_genes)

    def _stats(mat):
        mean = mat.mean(axis=0)
        sd = mat.std(axis=0, ddof=1)
        return {
            "mean": mean,
            "sd_lo": mean - sd,
            "sd_hi": mean + sd,
        }

    rand_rare = _stats(rand_rare_mat)
    rand_common = _stats(rand_common_mat)

    rare_nauc_guided = normalized_auc(guided_rare)
    rare_nauc_random = normalized_auc(rand_rare["mean"])
    common_nauc_guided = normalized_auc(guided_common)
    common_nauc_random = normalized_auc(rand_common["mean"])

    print(f"  Rare genes:   guided nAUC={rare_nauc_guided:.4f}  "
          f"random nAUC={rare_nauc_random:.4f}  "
          f"Δ={rare_nauc_guided - rare_nauc_random:+.4f}")
    print(f"  Common genes: guided nAUC={common_nauc_guided:.4f}  "
          f"random nAUC={common_nauc_random:.4f}  "
          f"Δ={common_nauc_guided - common_nauc_random:+.4f}")

    # What fraction of rare genes are missed at K=50% of queries?
    k_half = Q // 2
    rare_at_half_guided = guided_rare[k_half]
    rare_at_half_random = float(rand_rare["mean"][k_half])
    print(f"\n  At K={k_half} (50% of queries):")
    print(f"    Rare genes found: Guided={rare_at_half_guided}  "
          f"Random={rare_at_half_random:.1f}  "
          f"(of {len(rare_genes)} total)")
    print(f"    Common genes found: Guided={guided_common[k_half]}  "
          f"Random={rand_common['mean'][k_half]:.1f}  "
          f"(of {len(common_genes)} total)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ks = np.arange(Q + 1)

    for ax, (name, guided_cum, rand_stats, total) in zip(axes, [
        ("Rare SL Genes (1-2 partners)", guided_rare, rand_rare, len(rare_genes)),
        ("Common SL Genes (≥3 partners)", guided_common, rand_common, len(common_genes)),
    ]):
        ax.fill_between(ks, rand_stats["sd_lo"], rand_stats["sd_hi"],
                        alpha=0.15, color="grey", label="Random ± 1 SD")
        ax.plot(ks, rand_stats["mean"], "grey", ls="--", lw=1.5, label="Random")
        ax.plot(ks, guided_cum, "#1f77b4", lw=2, label="Guided (Σ pred_prob)")
        ax.axhline(y=total, color="black", ls="-", lw=0.5, alpha=0.3)
        ax.set_xlabel("Queries screened")
        ax.set_ylabel("Genes found")
        ax.set_title(f"{name}\n(n={total})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Rare vs Common SL Gene Recovery",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "rare_vs_common.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_rare_genes": len(rare_genes),
        "n_common_genes": len(common_genes),
        "rare_guided_nAUC": rare_nauc_guided,
        "rare_random_nAUC": rare_nauc_random,
        "rare_nAUC_delta": rare_nauc_guided - rare_nauc_random,
        "common_guided_nAUC": common_nauc_guided,
        "common_random_nAUC": common_nauc_random,
        "common_nAUC_delta": common_nauc_guided - common_nauc_random,
    }
    pd.DataFrame([summary]).to_csv(output_dir / "rare_vs_common.tsv",
                                    sep="\t", index=False)
    return summary


# ===================================================================
# Test 5: Per-model variability
# ===================================================================

def test5_per_model_variability(
    df: pd.DataFrame,
    label_col: str,
    query_col: str,
    gene_col: str,
    model_dir: str,
    data_dir: str,
    n_data_iters: int,
    output_dir: Path,
    seed: int = 42,
) -> dict:
    """Run prioritization with each individual model's predictions."""
    print("\n" + "=" * 70)
    print("  TEST 5: Per-Model Variability")
    print("=" * 70)
    os.makedirs(output_dir, exist_ok=True)

    import joblib
    import torch
    import torch.nn as nn
    from .external_validate_old_models import (
        NeuralNetwork,
        _predict,
        _register_legacy_class,
    )
    from .data import NON_FEATURE_COLS, filter_unique_pairs, load_tsv_iterations

    _register_legacy_class()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir_p = Path(model_dir)

    sl, sl_per_query, sl_genes_per_query, _ = _sl_stats(
        df, label_col, query_col, gene_col
    )
    all_queries = sorted(df[query_col].unique())
    Q = len(all_queries)

    # Load the raw data with features for model inference
    print(f"  Loading raw features from {data_dir} ...")
    raw = load_tsv_iterations(data_dir, n_iters=n_data_iters)
    raw = filter_unique_pairs(raw)
    feature_cols = [c for c in raw.columns if c not in NON_FEATURE_COLS]
    X = raw[feature_cols].dropna(how="any")
    # Align indices with df (both come from the same data, same order after
    # filter_unique_pairs, but let's be safe)
    raw = raw.loc[X.index].reset_index(drop=True)
    X = X.reset_index(drop=True)
    print(f"  Features shape: {X.shape}")

    # Random baseline (reuse from previous)
    rng = np.random.default_rng(seed)
    n_perm = 5_000
    rand_sl_mat = np.zeros((n_perm, Q + 1), dtype=int)
    rand_cov_mat = np.zeros((n_perm, Q + 1), dtype=int)
    for p in range(n_perm):
        perm = list(rng.permutation(all_queries))
        rand_sl_mat[p] = cumulative_sl_pairs(perm, sl_per_query)
        rand_cov_mat[p] = cumulative_gene_coverage(perm, sl_genes_per_query)
    rand_sl_nauc = normalized_auc(rand_sl_mat.mean(axis=0))
    rand_cov_nauc = normalized_auc(rand_cov_mat.mean(axis=0))

    # Run each model
    model_results = []
    n_folds = 10
    for i in range(1, n_folds + 1):
        mp = model_dir_p / f"CV2_811_GIV_NN_LR1e2_50e_p10_d01_{i}.pth"
        sp = model_dir_p / f"CV2_811_seed{i}.joblib"
        if not mp.exists() or not sp.exists():
            print(f"  Warning: skipping model {i} (files not found)")
            continue

        tag = f"model_{i}"
        print(f"  Running {tag} ...")

        scaler = joblib.load(sp)
        n_scaler_feat = scaler.n_features_in_
        X_arr = X.values
        X_sub = X_arr[:, :n_scaler_feat] if X_arr.shape[1] != n_scaler_feat else X_arr
        X_scaled = scaler.transform(X_sub)

        model = torch.load(mp, map_location=device, weights_only=False)
        model.eval()
        y_score = _predict(model, X_scaled, device)

        # Rank queries by sum of predicted probabilities
        df_tmp = raw[[query_col]].copy()
        df_tmp["_score"] = y_score
        q_scores = df_tmp.groupby(query_col)["_score"].sum().sort_values(
            ascending=False
        )
        order = list(q_scores.index)

        sl_cum = cumulative_sl_pairs(order, sl_per_query)
        cov_cum = cumulative_gene_coverage(order, sl_genes_per_query)
        sl_nauc = normalized_auc(sl_cum)
        cov_nauc = normalized_auc(cov_cum)

        model_results.append({
            "model": tag,
            "sl_nAUC": sl_nauc,
            "cov_nAUC": cov_nauc,
            "sl_cum": sl_cum,
            "cov_cum": cov_cum,
        })
        print(f"    SL nAUC={sl_nauc:.4f}  Coverage nAUC={cov_nauc:.4f}")

    # Also add ensemble
    s_ens = score_queries_by_sum(df, query_col, "predict_proba")
    ens_order = list(s_ens.index)
    ens_sl_cum = cumulative_sl_pairs(ens_order, sl_per_query)
    ens_cov_cum = cumulative_gene_coverage(ens_order, sl_genes_per_query)
    ens_sl_nauc = normalized_auc(ens_sl_cum)
    ens_cov_nauc = normalized_auc(ens_cov_cum)

    # Summary table
    df_res = pd.DataFrame([{k: v for k, v in r.items()
                            if k not in ("sl_cum", "cov_cum")}
                           for r in model_results])
    df_res.to_csv(output_dir / "per_model_nauc.tsv", sep="\t", index=False)

    sl_naucs = df_res["sl_nAUC"].values
    cov_naucs = df_res["cov_nAUC"].values
    n_models = len(df_res)

    print(f"\n  Summary across {n_models} models:")
    print(f"    SL nAUC:  {sl_naucs.mean():.4f} ± {sl_naucs.std(ddof=1):.4f}  "
          f"[{sl_naucs.min():.4f} – {sl_naucs.max():.4f}]  "
          f"(random={rand_sl_nauc:.4f}, ensemble={ens_sl_nauc:.4f})")
    print(f"    Cov nAUC: {cov_naucs.mean():.4f} ± {cov_naucs.std(ddof=1):.4f}  "
          f"[{cov_naucs.min():.4f} – {cov_naucs.max():.4f}]  "
          f"(random={rand_cov_nauc:.4f}, ensemble={ens_cov_nauc:.4f})")
    # How many models beat random?
    n_beat_sl = int((sl_naucs > rand_sl_nauc).sum())
    n_beat_cov = int((cov_naucs > rand_cov_nauc).sum())
    print(f"    Models beating random: {n_beat_sl}/{n_models} (SL), "
          f"{n_beat_cov}/{n_models} (Coverage)")

    # Plot: spaghetti plot of cumulative curves from each model
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ks = np.arange(Q + 1)

    for ax, metric_key, ylabel, title, rand_mean, ens_cum in [
        (axes[0], "sl_cum", "Cumulative SL pairs", "SL Discovery",
         rand_sl_mat.mean(axis=0), ens_sl_cum),
        (axes[1], "cov_cum", "Library genes covered", "Library Coverage",
         rand_cov_mat.mean(axis=0), ens_cov_cum),
    ]:
        # Random
        r_mat = rand_sl_mat if "sl" in metric_key else rand_cov_mat
        r_sd = r_mat.std(axis=0, ddof=1)
        ax.fill_between(ks,
                        rand_mean - r_sd,
                        rand_mean + r_sd,
                        alpha=0.1, color="grey", label="Random ± 1 SD")
        ax.plot(ks, rand_mean, "grey", ls="--", lw=1.5, label="Random")

        # Individual models
        for r in model_results:
            ax.plot(ks, r[metric_key], alpha=0.4, lw=1, color="#aec7e8")
        # Dummy for legend
        ax.plot([], [], alpha=0.4, lw=1, color="#aec7e8",
                label=f"Individual models (n={n_models})")

        # Ensemble
        ax.plot(ks, ens_cum, "#1f77b4", lw=2.5, label="Ensemble")

        ax.set_xlabel("Queries screened")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Per-Model Variability in Query Prioritization",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "per_model_variability.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    return {
        "n_models": n_models,
        "sl_nAUC_mean": float(sl_naucs.mean()),
        "sl_nAUC_std": float(sl_naucs.std(ddof=1)),
        "sl_nAUC_min": float(sl_naucs.min()),
        "sl_nAUC_max": float(sl_naucs.max()),
        "cov_nAUC_mean": float(cov_naucs.mean()),
        "cov_nAUC_std": float(cov_naucs.std(ddof=1)),
        "random_sl_nAUC": rand_sl_nauc,
        "random_cov_nAUC": rand_cov_nauc,
        "ensemble_sl_nAUC": ens_sl_nauc,
        "ensemble_cov_nAUC": ens_cov_nauc,
        "n_models_beat_random_sl": n_beat_sl,
        "n_models_beat_random_cov": n_beat_cov,
    }


# ===================================================================
# Test 6: Stratified analysis by query difficulty
# ===================================================================

def test6_stratified_difficulty(
    df: pd.DataFrame,
    label_col: str,
    score_col: str,
    query_col: str,
    gene_col: str,
    query_metrics_path: str,
    output_dir: Path,
    seed: int = 42,
) -> dict:
    """Split queries by prediction difficulty and test benefit per stratum."""
    print("\n" + "=" * 70)
    print("  TEST 6: Stratified Analysis by Query Difficulty")
    print("=" * 70)
    os.makedirs(output_dir, exist_ok=True)

    sl, sl_per_query, sl_genes_per_query, _ = _sl_stats(
        df, label_col, query_col, gene_col
    )
    all_queries = sorted(df[query_col].unique())

    # Load per-query AUROC
    qm = pd.read_csv(query_metrics_path, sep="\t")
    qm = qm[qm["AUROC_mean"].notna()]  # drop queries with undefined AUROC

    # Split into terciles by AUROC
    tercile_labels = ["Hard (low AUROC)", "Medium", "Easy (high AUROC)"]
    qm["difficulty"] = pd.qcut(qm["AUROC_mean"], q=3, labels=tercile_labels)

    # Map query -> difficulty
    query_difficulty = dict(zip(qm["Query"], qm["difficulty"]))
    # Queries not in qm (e.g. 0 positives, NaN AUROC) → mark as "Undefined"
    for q in all_queries:
        if q not in query_difficulty:
            query_difficulty[q] = "Undefined"

    difficulties = tercile_labels
    strata = {d: [q for q in all_queries if query_difficulty.get(q) == d]
              for d in difficulties}

    # Print AUROC ranges per stratum
    for d in difficulties:
        sub = qm[qm["difficulty"] == d]
        if len(sub) == 0:
            continue
        lo, hi = sub["AUROC_mean"].min(), sub["AUROC_mean"].max()
        n_sl = sum(sl_per_query.get(q, 0) for q in strata[d])
        print(f"  {d}: {len(strata[d])} queries, "
              f"AUROC [{lo:.4f} - {hi:.4f}], {n_sl} SL pairs")

    # Model-guided order (full ranking, then look at position of each stratum)
    s = score_queries_by_sum(df, query_col, score_col)
    guided_order = list(s.index)

    rng = np.random.default_rng(seed)
    n_perm = 5_000

    stratum_results = {}
    for d in difficulties:
        qs = strata[d]
        if not qs:
            continue
        Q_s = len(qs)

        # subset SL stats
        sl_pq_s = {q: sl_per_query.get(q, 0) for q in qs}
        sl_gpq_s = {q: sl_genes_per_query.get(q, set()) for q in qs}

        # Guided: order within this stratum (preserving global ranking)
        guided_sub = [q for q in guided_order if q in set(qs)]
        g_sl = cumulative_sl_pairs(guided_sub, sl_pq_s)
        g_cov = cumulative_gene_coverage(guided_sub, sl_gpq_s)

        # Random within stratum
        r_sl_mat = np.zeros((n_perm, Q_s + 1), dtype=int)
        r_cov_mat = np.zeros((n_perm, Q_s + 1), dtype=int)
        for p in range(n_perm):
            perm = list(rng.permutation(qs))
            r_sl_mat[p] = cumulative_sl_pairs(perm, sl_pq_s)
            r_cov_mat[p] = cumulative_gene_coverage(perm, sl_gpq_s)

        g_sl_nauc = normalized_auc(g_sl)
        r_sl_nauc = normalized_auc(r_sl_mat.mean(axis=0))
        g_cov_nauc = normalized_auc(g_cov)
        r_cov_nauc = normalized_auc(r_cov_mat.mean(axis=0))

        total_sl_s = sum(sl_pq_s.values())
        total_genes_s = len(set().union(*sl_gpq_s.values())) if sl_gpq_s else 0

        stratum_results[d] = {
            "n_queries": Q_s,
            "total_sl_pairs": total_sl_s,
            "total_sl_genes": total_genes_s,
            "guided_sl_nAUC": g_sl_nauc,
            "random_sl_nAUC": r_sl_nauc,
            "sl_nAUC_delta": g_sl_nauc - r_sl_nauc,
            "guided_cov_nAUC": g_cov_nauc,
            "random_cov_nAUC": r_cov_nauc,
            "cov_nAUC_delta": g_cov_nauc - r_cov_nauc,
            # cumulative curves for plotting
            "_g_sl": g_sl, "_g_cov": g_cov,
            "_r_sl": r_sl_mat, "_r_cov": r_cov_mat,
        }
        print(f"  {d}:  SL Δ={g_sl_nauc - r_sl_nauc:+.4f}  "
              f"Coverage Δ={g_cov_nauc - r_cov_nauc:+.4f}")

    # Save summary table
    rows = []
    for d in difficulties:
        if d not in stratum_results:
            continue
        r = stratum_results[d]
        rows.append({
            "stratum": d,
            "n_queries": r["n_queries"],
            "total_sl_pairs": r["total_sl_pairs"],
            "total_sl_genes": r["total_sl_genes"],
            "guided_sl_nAUC": r["guided_sl_nAUC"],
            "random_sl_nAUC": r["random_sl_nAUC"],
            "sl_nAUC_delta": r["sl_nAUC_delta"],
            "guided_cov_nAUC": r["guided_cov_nAUC"],
            "random_cov_nAUC": r["random_cov_nAUC"],
            "cov_nAUC_delta": r["cov_nAUC_delta"],
        })
    pd.DataFrame(rows).to_csv(output_dir / "stratified_difficulty.tsv",
                               sep="\t", index=False)

    # Plot: one row per stratum, 2 columns (SL, coverage)
    n_strata = len([d for d in difficulties if d in stratum_results])
    fig, axes = plt.subplots(n_strata, 2, figsize=(14, 5 * n_strata),
                             squeeze=False)

    for row_idx, d in enumerate(d for d in difficulties
                                if d in stratum_results):
        r = stratum_results[d]
        ks_s = np.arange(r["n_queries"] + 1)

        for col_idx, (g_cum, r_mat, ylabel, total) in enumerate([
            (r["_g_sl"], r["_r_sl"], "Cumulative SL pairs", r["total_sl_pairs"]),
            (r["_g_cov"], r["_r_cov"], "Library genes covered", r["total_sl_genes"]),
        ]):
            ax = axes[row_idx, col_idx]
            r_mean = r_mat.mean(axis=0)
            r_sd = r_mat.std(axis=0, ddof=1)
            ax.fill_between(ks_s, r_mean - r_sd, r_mean + r_sd,
                            alpha=0.15, color="grey", label="Random ± 1 SD")
            ax.plot(ks_s, r_mean, "grey", ls="--", lw=1.5, label="Random")
            ax.plot(ks_s, g_cum, "#1f77b4", lw=2, label="Guided")
            ax.axhline(y=total, color="black", ls="-", lw=0.5, alpha=0.3)
            ax.set_xlabel("Queries screened")
            ax.set_ylabel(ylabel)
            metric_type = "SL" if col_idx == 0 else "Cov"
            delta = r[f"{'sl' if col_idx == 0 else 'cov'}_nAUC_delta"]
            ax.set_title(f"{d} (n={r['n_queries']})  "
                         f"[{metric_type} ΔnAUC={delta:+.4f}]")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Stratified by Query Difficulty: Model Benefit per Stratum",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "stratified_difficulty.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # Clean up internal arrays before returning
    return {d: {k: v for k, v in r.items() if not k.startswith("_")}
            for d, r in stratum_results.items()}


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extended evaluations for model-guided query prioritization",
    )
    p.add_argument(
        "--predictions",
        default="outputs/external_val/external_val_old_models/ensemble_predictions.tsv",
    )
    p.add_argument(
        "--query-metrics",
        default="outputs/external_val/external_val_old_models/query_metrics.tsv",
        help="Per-query metrics TSV from external validation (with AUROC_mean)",
    )
    p.add_argument(
        "--model-dir",
        default="data/interim/ReLU128_f_a075_g15_10folds_pt10",
        help="Directory with old model .pth and scaler .joblib files",
    )
    p.add_argument(
        "--data-dir",
        default="data/input/GIV_24Q4_heldout",
        help="Directory with external validation TSV files (for Test 5 features)",
    )
    p.add_argument("--n-data-iters", type=int, default=11,
                   help="Number of TSV iterations in data-dir")
    p.add_argument("--label-col", default="GI_stringent_Type2")
    p.add_argument("--gene-col", default="Gene")
    p.add_argument("--query-col", default="Query")
    p.add_argument("--score-col", default="predict_proba")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        default="outputs/external_val/external_val_old_models/extended_eval",
    )
    return p.parse_args()


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    # Load data once
    print(f"Loading predictions from {args.predictions} ...")
    df = _load_ensemble(args.predictions, args.label_col)
    print(f"  {len(df):,} gene pairs, {df[args.query_col].nunique()} queries")

    all_summaries = {}

    # --- Test 1 ---
    all_summaries["test1_pair_precision"] = test1_pair_precision(
        df, args.label_col, args.score_col,
        output_dir / "test1_pair_precision", args.seed,
    )

    # --- Test 2 ---
    all_summaries["test2_hub_discovery"] = test2_sl_hub_discovery(
        df, args.label_col, args.score_col,
        args.query_col, args.gene_col,
        output_dir / "test2_hub_discovery", args.seed,
    )

    # --- Test 3 ---
    all_summaries["test3_efficiency"] = test3_efficiency_budget(
        df, args.label_col, args.score_col,
        args.query_col, args.gene_col,
        output_dir / "test3_efficiency", args.seed,
    )

    # --- Test 4 ---
    all_summaries["test4_rare_genes"] = test4_rare_gene_enrichment(
        df, args.label_col, args.score_col,
        args.query_col, args.gene_col,
        output_dir / "test4_rare_genes", args.seed,
    )

    # --- Test 5 ---
    all_summaries["test5_per_model"] = test5_per_model_variability(
        df, args.label_col,
        args.query_col, args.gene_col,
        args.model_dir,
        args.data_dir,
        args.n_data_iters,
        output_dir / "test5_per_model", args.seed,
    )

    # --- Test 6 ---
    all_summaries["test6_stratified"] = test6_stratified_difficulty(
        df, args.label_col, args.score_col,
        args.query_col, args.gene_col,
        args.query_metrics,
        output_dir / "test6_stratified", args.seed,
    )

    # Save master summary
    with open(output_dir / "extended_eval_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"  All 6 tests complete. Outputs in {output_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
