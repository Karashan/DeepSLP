"""Binarized SL core-network analysis (cutoff on ensemble mean_pred_proba).

Reads a filtered pairwise file (gene1, gene2, mean_pred_proba), saves a sorted
core-network TSV, and reports basic raw-space network statistics with plots:
degree distribution, hub analysis, edge-score distribution, connectivity,
clustering, and a strongest-edges subnetwork view.
"""
from __future__ import annotations
import argparse, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np, pandas as pd
try:
    import networkx as nx; HAVE_NX = True
except Exception:
    HAVE_NX = False

N_UNIVERSE = 17840


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    if n == 0 or x.sum() == 0: return float("nan")
    c = np.cumsum(x); return float((n + 1 - 2 * (c / c[-1]).sum()) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--cutoff", type=float, required=True)
    ap.add_argument("--core-out", required=True, help="path to save sorted core-network tsv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--score-col", default="mean_pred_proba")
    ap.add_argument("--subnet-edges", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    sc = args.score_col

    df = pd.read_csv(args.input, sep="\t")
    df = df[df[sc] >= args.cutoff].sort_values(sc, ascending=False).reset_index(drop=True)
    df.to_csv(args.core_out, sep="\t", index=False)

    # ---- degree ----
    deg = pd.concat([df["gene1"], df["gene2"]]).value_counts()
    deg.rename_axis("gene").reset_index(name="degree").to_csv(
        os.path.join(args.outdir, "gene_degree.tsv"), sep="\t", index=False)
    n_edges = len(df); n_nodes = deg.size
    dvals = deg.values
    qs = [0, 25, 50, 75, 90, 95, 99, 100]

    L = [f"=== SL core network (mean_pred_proba >= {args.cutoff}) ===",
         f"edges (SL pairs)        : {n_edges:,}",
         f"nodes (unique genes)    : {n_nodes:,}  ({n_nodes/N_UNIVERSE:.1%} of {N_UNIVERSE} universe)",
         f"mean degree             : {2*n_edges/n_nodes:.2f}",
         f"median degree           : {np.median(dvals):.1f}",
         f"max degree              : {int(dvals.max())} ({deg.index[0]})",
         f"degree Gini             : {gini(dvals):.3f}",
         f"genes with degree==1    : {(deg==1).sum():,} ({(deg==1).mean():.1%})",
         "",
         "degree percentiles:"]
    for q in qs:
        L.append(f"   {q:>3}%  {np.percentile(dvals,q):.1f}")
    L += ["", "edge score (mean_pred_proba):",
          f"   min={df[sc].min():.4f}  median={df[sc].median():.4f}  "
          f"mean={df[sc].mean():.4f}  max={df[sc].max():.4f}",
          "", "top 20 hub genes:",
          "   " + ", ".join(f"{g}({d})" for g, d in deg.head(20).items())]

    if HAVE_NX:
        G = nx.from_pandas_edgelist(df, "gene1", "gene2")
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        L += ["",
              f"density                 : {nx.density(G):.2e}",
              f"connected components    : {len(comps):,}",
              f"largest component       : {len(comps[0]):,} nodes ({len(comps[0])/n_nodes:.1%})",
              f"components with >=3 nodes: {sum(len(c)>=3 for c in comps):,}",
              f"avg clustering coeff    : {nx.average_clustering(G):.4f}",
              f"transitivity            : {nx.transitivity(G):.4f}",
              f"triangles (total)       : {sum(nx.triangles(G).values())//3:,}"]
        try:
            L.append(f"degree assortativity    : {nx.degree_assortativity_coefficient(G):.4f}")
        except Exception:
            pass

    txt = "\n".join(L)
    with open(os.path.join(args.outdir, "network_stats.txt"), "w") as fh:
        fh.write(txt + "\n")
    print(txt)

    # ---- plots ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].hist(dvals, bins=60, color="#2c7fb8", edgecolor="white"); ax[0].set_yscale("log")
    ax[0].set(xlabel="degree (# SL partners)", ylabel="# genes (log)",
              title=f"Degree distribution (Gini={gini(dvals):.3f})")
    ranks = np.arange(1, n_nodes + 1)
    ax[1].loglog(ranks, dvals, ".", ms=3, color="#2c7fb8")
    ax[1].set(xlabel="rank", ylabel="degree", title="Degree rank (log-log)"); ax[1].grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "degree_distribution.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6)); top = deg.head(25)
    ax.barh(top.index[::-1], top.values[::-1], color="#756bb1")
    ax.set(xlabel="degree (# SL partners)", title="Top-25 hub genes in core SL network")
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "top_hubs.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df[sc], bins=60, color="#d95f0e", edgecolor="white")
    ax.set(xlabel="mean_pred_proba", ylabel="# edges", title="Core edge score distribution")
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "edge_score_distribution.png"), dpi=150); plt.close(fig)

    if HAVE_NX:
        te = df.head(args.subnet_edges)
        SG = nx.from_pandas_edgelist(te, "gene1", "gene2", edge_attr=sc)
        pos = nx.spring_layout(SG, seed=1, k=0.5)
        d = dict(SG.degree())
        fig, ax = plt.subplots(figsize=(11, 9))
        nx.draw_networkx_edges(SG, pos, alpha=0.3, ax=ax)
        nx.draw_networkx_nodes(SG, pos, node_size=[30 + 25 * d[n] for n in SG], node_color=[d[n] for n in SG],
                               cmap="autumn_r", ax=ax)
        lab = {n: n for n in sorted(d, key=d.get, reverse=True)[:30]}
        nx.draw_networkx_labels(SG, pos, labels=lab, font_size=7, ax=ax)
        ax.set_title(f"Strongest {args.subnet_edges} SL edges (node size/color = degree)"); ax.axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "top_subnetwork.png"), dpi=150); plt.close(fig)

    print(f"\nCore network -> {args.core_out}")
    print(f"Analysis      -> {args.outdir}")


if __name__ == "__main__":
    main()
