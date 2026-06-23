"""Quantify structure in the saved UMAP/t-SNE embeddings of the GIV feature space.

Answers, with numbers:
  * Are the maps organized by GENE IDENTITY (Query / library Gene)?  -> kNN purity
  * Do GI-positive pairs CO-LOCALIZE?                                  -> kNN positive enrichment
  * Is there a qGI_score GRADIENT?                                     -> local autocorrelation
"""
import os, sys, glob
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

OUT = sys.argv[1] if len(sys.argv) > 1 else sorted(
    glob.glob(os.path.expanduser("~/DeepSLP/data/interim/feature_viz_*")))[-1]
print("Analyzing:", OUT)


def knn_purity_categorical(emb, labels, k=15):
    """Mean fraction of a point's k nearest neighbours sharing its label."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    _, idx = nn.kneighbors(emb)
    idx = idx[:, 1:]
    lab = labels.to_numpy()
    same = (lab[idx] == lab[:, None])
    return float(same.mean())


def knn_positive_enrichment(emb, y, k=15):
    """Among neighbours of POSITIVE points, positive rate vs global base rate."""
    y = (y == 1).to_numpy()
    base = y.mean()
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    _, idx = nn.kneighbors(emb[y])
    idx = idx[:, 1:]
    neigh_pos_rate = y[idx].mean()
    return float(neigh_pos_rate), float(base), float(neigh_pos_rate / base)


def morans_like(emb, value, k=15):
    """Local autocorrelation: corr between a point's value and its kNN mean value."""
    v = value.to_numpy().astype(float)
    m = np.isfinite(v)
    emb, v = emb[m], v[m]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    _, idx = nn.kneighbors(emb)
    neigh_mean = v[idx[:, 1:]].mean(axis=1)
    return float(np.corrcoef(v, neigh_mean)[0, 1])


def shuffled_baseline_purity(labels, k=15):
    """Expected purity if labels were random (= sum p_i^2 roughly), for reference."""
    p = labels.value_counts(normalize=True).to_numpy()
    return float((p ** 2).sum())


for method in ["umap", "tsne"]:
    emb = np.load(os.path.join(OUT, f"{method}_2d.npy"))
    meta = pd.read_csv(os.path.join(OUT, f"sample_meta_{method}.tsv"), sep="\t")
    n = len(meta)
    print(f"\n===== {method.upper()}  (n={n}) =====")

    nq = meta["Query"].nunique()
    ng = meta["Gene"].nunique()
    print(f"unique Query genes: {nq} | unique library Genes: {ng}")

    q_pur = knn_purity_categorical(emb, meta["Query"], k=15)
    q_base = shuffled_baseline_purity(meta["Query"])
    g_pur = knn_purity_categorical(emb, meta["Gene"], k=15)
    g_base = shuffled_baseline_purity(meta["Gene"])
    print(f"kNN purity by Query : {q_pur:.3f}  (random ~{q_base:.4f}, enrichment {q_pur/q_base:.0f}x)")
    print(f"kNN purity by Gene  : {g_pur:.3f}  (random ~{g_base:.4f}, enrichment {g_pur/g_base:.0f}x)")

    npr, base, enr = knn_positive_enrichment(emb, meta["GI_stringent_Type2"], k=15)
    print(f"positive co-localization: neighbours-of-positives positive rate {npr:.4f} "
          f"vs base {base:.4f}  -> {enr:.1f}x enrichment")

    qgi_ac = morans_like(emb, meta["qGI_score"], k=15)
    print(f"qGI_score local autocorrelation (point vs kNN mean): r = {qgi_ac:.3f}")
