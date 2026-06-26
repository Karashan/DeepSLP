"""Deep analysis of the binarized SL core network:
(a) degree vs mean incident-edge score (are hubs high-confidence or threshold-grazing?)
(b) community detection + per-module functional summary (annotated with CORUM)
(c) overlap / enrichment of core SL edges with CORUM co-complex pairs (gold standard)
"""
from __future__ import annotations
import os, sys, itertools
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import networkx as nx
from scipy.stats import hypergeom, spearmanr

REPO = os.path.expanduser("~/DeepSLP")
CORE = f"{REPO}/data/interim/all_pairs_pred/SL_core_network_cutoff0.162.tsv"
CORUM = f"{REPO}/data/input/external/corum_humanComplexes_5p3.txt"
AE3_KO = f"{REPO}/data/input/AE_3L/AE_std100_CRISPRGeneEffect_24Q4_imputed_gene_wise_mean_align_qGI2021.txt"
AE3_EX = f"{REPO}/data/input/AE_3L/AE_std100_Expression_BC_24Q4_align_qGI2021.txt"
OUT = f"{REPO}/data/interim/all_pairs_pred/core_network_analysis"
os.makedirs(OUT, exist_ok=True)


def canon(a, b):
    return (a, b) if a < b else (b, a)


def main():
    df = pd.read_csv(CORE, sep="\t")
    sc = "mean_pred_proba"
    G = nx.from_pandas_edgelist(df, "gene1", "gene2", edge_attr=sc)
    report = []

    # ---------------- (a) degree vs incident score ----------------
    st = pd.concat([
        df[["gene1", sc]].rename(columns={"gene1": "gene"}),
        df[["gene2", sc]].rename(columns={"gene2": "gene"}),
    ])
    g = st.groupby("gene")[sc].agg(degree="count", mean_incident="mean", max_incident="max").reset_index()
    g = g.sort_values("degree", ascending=False)
    g.to_csv(os.path.join(OUT, "gene_incident_stats.tsv"), sep="\t", index=False)
    rho, p = spearmanr(g["degree"], g["mean_incident"])
    hub = g.head(50); nonhub = g[g.degree <= 3]
    report += ["=== (a) Degree vs mean incident-edge score ===",
               f"Spearman(degree, mean_incident_score) = {rho:+.3f} (p={p:.1e})",
               f"top-50 hubs: mean incident score = {hub.mean_incident.mean():.4f} "
               f"(median {hub.mean_incident.median():.4f})",
               f"degree<=3 genes: mean incident score = {nonhub.mean_incident.mean():.4f}",
               f"global cutoff = 0.162; core median edge score = {df[sc].median():.4f}",
               "Interpretation: " + ("hubs are threshold-grazing (low incident score)"
                                      if hub.mean_incident.mean() < df[sc].median() + 0.01
                                      else "hubs are high-confidence")]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(g["degree"], g["mean_incident"], s=8, alpha=0.4)
    ax.axhline(0.162, color="red", ls="--", lw=1, label="cutoff 0.162")
    ax.set_xscale("log"); ax.set(xlabel="degree (log)", ylabel="mean incident edge score",
                                 title=f"Degree vs mean incident score (Spearman {rho:+.2f})")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(OUT, "degree_vs_incident_score.png"), dpi=150); plt.close(fig)

    # ---------------- CORUM co-complex pairs ----------------
    cor = pd.read_csv(CORUM, sep="\t", low_memory=False)
    cor = cor[cor["organism"] == "Human"]
    complex_genes = {}
    cocomplex = set()
    for cid, name, sub in zip(cor["complex_id"], cor["complex_name"], cor["subunits_gene_name"]):
        if pd.isna(sub):
            continue
        genes = sorted({s.strip() for s in str(sub).split(";") if s.strip()})
        complex_genes[(cid, name)] = set(genes)
        for a, b in itertools.combinations(genes, 2):
            cocomplex.add(canon(a, b))
    corum_genes = set().union(*complex_genes.values()) if complex_genes else set()

    # universe of predictable genes (AE_3L intersection)
    ko = pd.read_csv(AE3_KO, sep="\t", index_col=0); ex = pd.read_csv(AE3_EX, sep="\t", index_col=0)
    universe = set(ko.index) & set(ex.index)
    U = corum_genes & universe                      # CORUM genes that are predictable

    # ---------------- (c) enrichment ----------------
    core_pairs = {canon(a, b) for a, b in zip(df.gene1, df.gene2)}
    # restrict to both-endpoints-in-U
    core_in_U = {(a, b) for (a, b) in core_pairs if a in U and b in U}
    cocomplex_in_U = {(a, b) for (a, b) in cocomplex if a in U and b in U}
    E = len(core_in_U); E_cc = len(core_in_U & cocomplex_in_U)
    total_pairs = len(U) * (len(U) - 1) // 2
    P_cc = len(cocomplex_in_U)
    bg_rate = P_cc / total_pairs if total_pairs else float("nan")
    obs_rate = E_cc / E if E else float("nan")
    fold = obs_rate / bg_rate if bg_rate else float("nan")
    # hypergeometric: population total_pairs, successes P_cc, draws E, observed E_cc
    pval = hypergeom.sf(E_cc - 1, total_pairs, P_cc, E)
    recall = E_cc / P_cc if P_cc else float("nan")

    # save validated co-complex core edges (with complex name + score)
    pair2name = {}
    for (cid, name), genes in complex_genes.items():
        gl = sorted(genes)
        for a, b in itertools.combinations(gl, 2):
            pair2name.setdefault(canon(a, b), name)
    val = []
    smap = {canon(a, b): s for a, b, s in zip(df.gene1, df.gene2, df[sc])}
    for pr in (core_in_U & cocomplex_in_U):
        val.append((pr[0], pr[1], smap[pr], pair2name.get(pr, "")))
    val = pd.DataFrame(val, columns=["gene1", "gene2", sc, "corum_complex"]).sort_values(sc, ascending=False)
    val.to_csv(os.path.join(OUT, "corum_validated_core_edges.tsv"), sep="\t", index=False)

    report += ["", "=== (c) CORUM co-complex enrichment ===",
               f"CORUM human complexes: {len(complex_genes):,}; CORUM genes: {len(corum_genes):,}",
               f"predictable CORUM genes (U): {len(U):,}; possible pairs C(|U|,2): {total_pairs:,}",
               f"co-complex pairs among U (P_cc): {P_cc:,}  (background rate {bg_rate:.2e})",
               f"core SL edges with both genes in U (E): {E:,}",
               f"core edges that are co-complex (E_cc): {E_cc:,}  (observed rate {obs_rate:.4f})",
               f"FOLD ENRICHMENT: {fold:.1f}x   hypergeometric p = {pval:.2e}",
               f"recall of co-complex pairs: {recall:.3%} ({E_cc}/{P_cc})",
               f"saved {len(val)} validated co-complex core edges -> corum_validated_core_edges.tsv",
               "top validated edges: " + ", ".join(
                   f"{r.gene1}-{r.gene2}({r.corum_complex[:24]})" for _, r in val.head(6).iterrows())]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    exp = E * bg_rate
    ax.bar(["expected\n(chance)", "observed"], [exp, E_cc], color=["#9e9e9e", "#2c7fb8"])
    ax.set(ylabel="# co-complex SL edges", title=f"CORUM enrichment: {fold:.1f}x (p={pval:.1e})")
    for i, v in enumerate([exp, E_cc]):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "corum_enrichment.png"), dpi=150); plt.close(fig)

    # ---------------- (b) communities + functional summary ----------------
    try:
        comms = nx.community.louvain_communities(G, seed=1, weight=sc)
    except Exception:
        comms = list(nx.community.greedy_modularity_communities(G))
    comms = sorted(comms, key=len, reverse=True)
    mod = nx.community.modularity(G, comms, weight=sc)
    rows, memb = [], []
    deg = dict(G.degree())
    for i, c in enumerate(comms):
        if len(c) < 3:
            continue
        genes = sorted(c, key=lambda x: deg[x], reverse=True)
        # dominant CORUM complex = max overlap
        best, bestov = "", 0
        for (cid, name), cg in complex_genes.items():
            ov = len(c & cg)
            if ov > bestov:
                bestov, best = ov, name
        rows.append({"community": i, "size": len(c), "top_genes": ",".join(genes[:8]),
                     "dominant_corum_complex": best, "corum_overlap": bestov})
        for gname in c:
            memb.append({"gene": gname, "community": i})
    comm_df = pd.DataFrame(rows)
    comm_df.to_csv(os.path.join(OUT, "community_summary.tsv"), sep="\t", index=False)
    pd.DataFrame(memb).to_csv(os.path.join(OUT, "community_membership.tsv"), sep="\t", index=False)

    report += ["", "=== (b) Community structure (Louvain) ===",
               f"communities (>=3 genes): {len(comm_df)}; modularity = {mod:.3f}",
               "largest modules (size | dominant CORUM complex | overlap | top genes):"]
    for _, r in comm_df.head(12).iterrows():
        report.append(f"  C{int(r.community):<3} n={int(r['size']):<4} "
                      f"[{(r.dominant_corum_complex or 'n/a')[:34]:<34}] ov={int(r.corum_overlap):<3} {r.top_genes}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sizes = comm_df["size"].values
    ax.bar(range(len(sizes[:30])), sizes[:30], color="#41ab5d")
    ax.set(xlabel="community (rank)", ylabel="# genes", title=f"Top community sizes ({len(comm_df)} modules, Q={mod:.2f})")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "community_sizes.png"), dpi=150); plt.close(fig)

    txt = "\n".join(report)
    with open(os.path.join(OUT, "deep_analysis_stats.txt"), "w") as fh:
        fh.write(txt + "\n")
    print(txt)
    print(f"\nOutputs -> {OUT}")


if __name__ == "__main__":
    main()
