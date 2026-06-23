"""Downstream analysis of an inferred SL gene-pair table (e.g. the top-0.3% file).

Designed for the ensemble output of infer_all_pairs.py:
    columns = gene1, gene2, prob_fold1..prob_foldM, mean
(falls back gracefully to a single `score` column).

Produces, under <outdir>:
  Tables
    score_summary.txt          - quantiles / describe of the ensemble mean
    gene_degree.tsv            - per-gene degree (number of partners) in this list
    top_hubs.tsv               - highest-degree genes
    top_robust_edges.tsv       - high-mean + low-disagreement edges
    network_summary.txt        - graph-level stats
  Figures
    mean_distribution.png      - histogram + ECDF of the ensemble mean
    fold_boxplot.png           - per-fold probability distributions
    fold_correlation.png       - inter-fold correlation heatmap
    degree_distribution.png    - degree histogram + rank/log-log
    top_hubs.png               - top-20 hub genes
    mean_vs_std.png            - score vs ensemble disagreement (hexbin)
    top_subnetwork.png         - spring layout of the strongest edges

Usage:
    python analysis_top_pairs.py \
        --input ../data/interim/all_pairs_pred/all_pairs_ensemble10_top0.3pct.tsv \
        --outdir ../data/interim/all_pairs_pred/analysis
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import networkx as nx
    HAVE_NX = True
except Exception:
    HAVE_NX = False


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--score-col", default=None, help="ranking column (auto: 'mean' or 'score')")
    ap.add_argument("--subnet-edges", type=int, default=200, help="# strongest edges in the subnetwork plot")
    ap.add_argument("--robust-top", type=int, default=2000, help="# rows to scan for robust high-confidence edges")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input, sep="\t")
    score_col = args.score_col or ("mean" if "mean" in df.columns else "score")
    fold_cols = [c for c in df.columns if c.startswith("prob_")]
    print(f"Loaded {len(df):,} edges | score col = '{score_col}' | fold cols = {len(fold_cols)}")

    # ------------------------------------------------------------------ #
    # 1. Score distribution
    # ------------------------------------------------------------------ #
    s = df[score_col].values
    qs = [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100]
    qvals = np.percentile(s, qs)
    lines = ["=== Ensemble mean score distribution ===",
             f"n_edges = {len(s):,}",
             f"min  = {s.min():.6f}   <-- minimal mean predicted probability in this file",
             f"max  = {s.max():.6f}",
             f"mean = {s.mean():.6f}   median = {np.median(s):.6f}   std = {s.std():.6f}",
             "", "percentiles:"]
    for q, v in zip(qs, qvals):
        lines.append(f"  {q:>5}%  {v:.6f}")
    summary_txt = "\n".join(lines)
    with open(os.path.join(args.outdir, "score_summary.txt"), "w") as fh:
        fh.write(summary_txt + "\n")
    print(summary_txt)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].hist(s, bins=80, color="#3477eb", edgecolor="white")
    ax[0].axvline(s.min(), color="red", ls="--", label=f"min={s.min():.4f}")
    ax[0].set(xlabel="ensemble mean probability", ylabel="# edges", title="Score histogram")
    ax[0].legend()
    xs = np.sort(s)
    ax[1].plot(xs, np.linspace(0, 1, len(xs)), color="#eb6a34")
    ax[1].set(xlabel="ensemble mean probability", ylabel="cumulative fraction", title="ECDF")
    ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "mean_distribution.png"), dpi=150); plt.close(fig)

    # ------------------------------------------------------------------ #
    # 2. Per-fold model behaviour + ensemble disagreement
    # ------------------------------------------------------------------ #
    if fold_cols:
        fold_std = df[fold_cols].std(axis=1)
        df["_ensemble_std"] = fold_std
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.boxplot([df[c].values for c in fold_cols], labels=[c.replace("prob_", "") for c in fold_cols], showfliers=False)
        ax.set(ylabel="predicted probability", title="Per-fold probability distribution (top list)")
        plt.xticks(rotation=45)
        fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "fold_boxplot.png"), dpi=150); plt.close(fig)

        corr = df[fold_cols].corr()
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(corr, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(fold_cols))); ax.set_xticklabels([c.replace("prob_", "") for c in fold_cols], rotation=90)
        ax.set_yticks(range(len(fold_cols))); ax.set_yticklabels([c.replace("prob_", "") for c in fold_cols])
        ax.set_title("Inter-fold correlation (top list)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "fold_correlation.png"), dpi=150); plt.close(fig)

        # score vs disagreement
        fig, ax = plt.subplots(figsize=(7, 5))
        hb = ax.hexbin(df[score_col], fold_std, gridsize=60, cmap="magma", bins="log")
        ax.set(xlabel="ensemble mean", ylabel="std across folds (disagreement)",
               title="Confidence: score vs ensemble disagreement")
        fig.colorbar(hb, ax=ax, label="log10(count)")
        fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "mean_vs_std.png"), dpi=150); plt.close(fig)

        # robust high-confidence edges: high mean & low std
        robust = (df.sort_values(score_col, ascending=False)
                    .head(args.robust_top)
                    .sort_values("_ensemble_std")
                    .head(50)[["gene1", "gene2", score_col, "_ensemble_std"] + fold_cols])
        robust.to_csv(os.path.join(args.outdir, "top_robust_edges.tsv"), sep="\t", index=False)

    # ------------------------------------------------------------------ #
    # 3. Network degree / hub analysis
    # ------------------------------------------------------------------ #
    deg = pd.concat([df["gene1"], df["gene2"]]).value_counts()
    deg.name = "degree"
    deg_df = deg.rename_axis("gene").reset_index()
    deg_df.to_csv(os.path.join(args.outdir, "gene_degree.tsv"), sep="\t", index=False)
    deg_df.head(50).to_csv(os.path.join(args.outdir, "top_hubs.tsv"), sep="\t", index=False)

    n_unique = len(deg)
    g = gini(deg.values)
    top10_genes = deg.head(10)
    edges_in_top10 = df[df["gene1"].isin(top10_genes.index) | df["gene2"].isin(top10_genes.index)].shape[0]
    share_top10 = edges_in_top10 / len(df)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].hist(deg.values, bins=60, color="#2ca25f", edgecolor="white")
    ax[0].set(xlabel="degree (# partners in top list)", ylabel="# genes",
              title=f"Degree distribution (Gini={g:.3f})")
    ax[0].set_yscale("log")
    ranks = np.arange(1, len(deg) + 1)
    ax[1].loglog(ranks, deg.values, ".", ms=3, color="#2ca25f")
    ax[1].set(xlabel="rank", ylabel="degree", title="Degree rank plot (log-log)")
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "degree_distribution.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    top20 = deg.head(20)
    ax.barh(top20.index[::-1], top20.values[::-1], color="#756bb1")
    ax.set(xlabel="degree (# partners)", title="Top-20 hub genes in the top list")
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "top_hubs.png"), dpi=150); plt.close(fig)

    # ------------------------------------------------------------------ #
    # 4. Graph-level structure
    # ------------------------------------------------------------------ #
    net_lines = ["=== Network structure (top list as a graph) ===",
                 f"edges (gene pairs) = {len(df):,}",
                 f"unique genes (nodes) = {n_unique:,}  (of 17,840 with embeddings)",
                 f"mean degree = {2*len(df)/n_unique:.2f}   max degree = {int(deg.max())} ({deg.index[0]})",
                 f"degree Gini = {g:.3f}  (0=uniform, 1=one hub)",
                 f"edges touching top-10 hubs = {edges_in_top10:,} ({share_top10:.1%} of edges)",
                 f"self-loops = {(df['gene1']==df['gene2']).sum()}"]
    if HAVE_NX:
        G = nx.from_pandas_edgelist(df, "gene1", "gene2")
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        gcc = comps[0]
        net_lines += [
            f"density = {nx.density(G):.2e}",
            f"connected components = {len(comps)}",
            f"largest component = {len(gcc):,} nodes ({len(gcc)/n_unique:.1%} of nodes)",
            f"isolated-ish (singleton-degree=1) genes = {(deg==1).sum():,}",
        ]
        # subnetwork plot of strongest edges
        top_edges = df.sort_values(score_col, ascending=False).head(args.subnet_edges)
        SG = nx.from_pandas_edgelist(top_edges, "gene1", "gene2", edge_attr=score_col)
        pos = nx.spring_layout(SG, seed=1, k=0.5)
        deg_sub = dict(SG.degree())
        fig, ax = plt.subplots(figsize=(11, 9))
        nx.draw_networkx_edges(SG, pos, alpha=0.3, ax=ax)
        nx.draw_networkx_nodes(SG, pos, node_size=[40 + 30 * deg_sub[n] for n in SG.nodes()],
                               node_color=[deg_sub[n] for n in SG.nodes()], cmap="autumn_r", ax=ax)
        hub_labels = {n: n for n in sorted(deg_sub, key=deg_sub.get, reverse=True)[:25]}
        nx.draw_networkx_labels(SG, pos, labels=hub_labels, font_size=7, ax=ax)
        ax.set_title(f"Strongest {args.subnet_edges} inferred SL edges (node size/color = degree)")
        ax.axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "top_subnetwork.png"), dpi=150); plt.close(fig)
    else:
        net_lines.append("networkx not available - skipped component/subnetwork analysis")

    net_txt = "\n".join(net_lines)
    with open(os.path.join(args.outdir, "network_summary.txt"), "w") as fh:
        fh.write(net_txt + "\n")
    print("\n" + net_txt)
    print(f"\nAll outputs saved under: {args.outdir}")


if __name__ == "__main__":
    main()
