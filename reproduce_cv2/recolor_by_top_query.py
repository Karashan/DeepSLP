"""Task 1: recolor the EXISTING UMAP/t-SNE embeddings by individual top query genes.

Reuses the embeddings + metadata already saved by visualize_feature_space.py
(no recompute). Highlights the N most frequent query genes with distinct colors;
all other points are light gray. This shows whether the per-query "islands"
seen in the kNN-purity analysis are made of individual query genes.

Usage:
    python reproduce_cv2/recolor_by_top_query.py            # latest feature_viz dir, top 12
    python reproduce_cv2/recolor_by_top_query.py --viz-dir <dir> --top 15
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def latest_viz_dir() -> str:
    dirs = sorted(glob.glob(os.path.join(REPO_ROOT, "data", "interim", "feature_viz_*")))
    if not dirs:
        raise FileNotFoundError("No feature_viz_* directory found.")
    return dirs[-1]


def plot_method(ax, emb, meta, top_queries, colors, title):
    q = meta["Query"].to_numpy()
    other = ~np.isin(q, top_queries)
    ax.scatter(emb[other, 0], emb[other, 1], c="lightgray", s=2, linewidths=0, alpha=0.35)
    for gene, col in zip(top_queries, colors):
        m = q == gene
        ax.scatter(emb[m, 0], emb[m, 1], c=[col], s=6, linewidths=0, alpha=0.85,
                   label=f"{gene} (n={int(m.sum())})")
    ax.set_title(title)
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    ax.legend(markerscale=3, fontsize=8, loc="best", ncol=2, framealpha=0.9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--viz-dir", default=None)
    p.add_argument("--top", type=int, default=12)
    args = p.parse_args()

    viz = args.viz_dir or latest_viz_dir()
    print("Using viz dir:", viz)

    # Pick top query genes from the UMAP sample (largest sample), shared across plots
    meta_u = pd.read_csv(os.path.join(viz, "sample_meta_umap.tsv"), sep="\t")
    top_queries = meta_u["Query"].value_counts().head(args.top).index.to_numpy()
    print(f"Top {args.top} query genes:", list(top_queries))

    cmap = cm.get_cmap("tab20", max(args.top, 3))
    colors = [cmap(i) for i in range(len(top_queries))]

    for method in ["umap", "tsne"]:
        emb = np.load(os.path.join(viz, f"{method}_2d.npy"))
        meta = pd.read_csv(os.path.join(viz, f"sample_meta_{method}.tsv"), sep="\t")
        fig, ax = plt.subplots(figsize=(12, 10))
        plot_method(ax, emb, meta, top_queries,
                    colors, f"{method.upper()} — colored by top {args.top} query genes")
        fig.tight_layout()
        out = os.path.join(viz, f"{method}_by_top_query.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print("Saved", out)


if __name__ == "__main__":
    main()
