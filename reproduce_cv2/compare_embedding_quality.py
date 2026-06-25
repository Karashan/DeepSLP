"""Compare embedding quality: 5L AE (data/input/AE) vs 3L AE (data/input/AE_3L).

Both were trained to embed the original DepMap 24Q4 matrices
(exvivo/DepMap_24Q4/), so a better embedding more faithfully preserves the
structure of that original data. For each modality (CRISPR, Expression) and each
AE version we compute, on a shared random sample of genes:

  * kNN recall@k        : overlap of a gene's k nearest neighbours in the
                          embedding vs the original DepMap profile (higher=better)
  * trustworthiness     : sklearn metric, penalizes false neighbours (->1 best)
  * dist Spearman       : rank-corr of pairwise distances embedding vs original
  * linear recon R^2    : variance of the original profile linearly recoverable
                          from the 128-d embedding (information retained)
  * co-ess recovery AUROC: can embedding cosine-sim rank the strongest original
                          gene-gene relationships (top 0.5% |corr|)

Simple sampled test (default 3000 genes); same genes/profiles used for both
versions so the comparison is apples-to-apples.
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPMAP = os.path.expanduser("~/exvivo/DepMap_24Q4")

MODALITIES = {
    "CRISPR": {
        "depmap": os.path.join(DEPMAP, "CRISPRGeneEffect.csv"),
        "ae5L": "AE/AE128_5L_std100_CRISPRGeneEffect_24Q4_imputed_gene_wise_mean_align_qGI2021.txt",
        "ae3L": "AE_3L/AE_std100_CRISPRGeneEffect_24Q4_imputed_gene_wise_mean_align_qGI2021.txt",
    },
    "Expression": {
        "depmap": os.path.join(DEPMAP, "OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv"),
        "ae5L": "AE/AE128_5L_std100_Expression_BC_24Q4_align_qGI2021.txt",
        "ae3L": "AE_3L/AE_std100_Expression_BC_24Q4_align_qGI2021.txt",
    },
}


def log(m): print(m, flush=True)


def load_ae(rel):
    d = pd.read_csv(os.path.join(REPO, "data", "input", rel), sep="\t", index_col=0)
    d.columns = [int(c) for c in d.columns]
    return d[~d.index.duplicated()].dropna()


def load_depmap_profiles(path, genes):
    """Read only the columns for `genes` from a DepMap (cell_lines x genes) csv.
    Returns genes x cell_lines profile (mean-imputed, gene-mean-centered)."""
    header = pd.read_csv(path, nrows=0)
    idx_col = header.columns[0]
    sym2col = {}
    for c in header.columns[1:]:
        sym = c.split(" (")[0]
        sym2col.setdefault(sym, c)
    usable = [g for g in genes if g in sym2col]
    cols = [idx_col] + [sym2col[g] for g in usable]
    df = pd.read_csv(path, usecols=cols, index_col=0)
    df.columns = [c.split(" (")[0] for c in df.columns]
    O = df[usable].T  # genes x cell_lines
    # impute NaN per gene (row) with that gene's mean across cell lines
    O = O.apply(lambda r: r.fillna(r.mean()), axis=1)
    O = O.dropna(how="any")  # drop genes still all-NaN
    return O


def knn_recall(O, E, k=20):
    nnO = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(O)
    nnE = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(E)
    _, iO = nnO.kneighbors(O); iO = iO[:, 1:]
    _, iE = nnE.kneighbors(E); iE = iE[:, 1:]
    rec = [len(set(a) & set(b)) / k for a, b in zip(iO, iE)]
    return float(np.mean(rec))


def dist_spearman(O, E, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(O.shape[0], size=min(n, O.shape[0]), replace=False)
    from scipy.spatial.distance import pdist
    dO = pdist(O[idx], metric="cosine")
    dE = pdist(E[idx], metric="cosine")
    return float(spearmanr(dO, dE).correlation)


def recon_r2(O, E, seed=0):
    Xtr, Xte, Ytr, Yte = train_test_split(E, O, test_size=0.3, random_state=seed)
    m = Ridge(alpha=1.0).fit(Xtr, Ytr)
    return float(r2_score(Yte, m.predict(Xte), multioutput="variance_weighted"))


def coess_auroc(O, E, top_frac=0.005, n=2500, seed=0):
    """Gold = top |corr| original gene pairs; score = embedding cosine sim."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(O.shape[0], size=min(n, O.shape[0]), replace=False)
    Os, Es = O[idx], E[idx]
    # original pairwise correlation (genes as rows -> correlate gene profiles)
    Oc = np.corrcoef(Os)
    En = Es / (np.linalg.norm(Es, axis=1, keepdims=True) + 1e-9)
    Ecos = En @ En.T
    iu = np.triu_indices(len(idx), k=1)
    g = np.abs(Oc[iu]); s = Ecos[iu]
    thr = np.quantile(g, 1 - top_frac)
    y = (g >= thr).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-genes", type=int, default=3000)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = []
    for mod, paths in MODALITIES.items():
        log(f"\n########## {mod} ##########")
        e5 = load_ae(paths["ae5L"]); e3 = load_ae(paths["ae3L"])
        common = sorted(set(e5.index) & set(e3.index))
        log(f"genes in both AE versions: {len(common)}")
        # sample, then keep those present in DepMap profile
        samp = rng.choice(common, size=min(args.n_genes, len(common)), replace=False)
        log(f"loading DepMap profiles for {len(samp)} sampled genes ...")
        O = load_depmap_profiles(paths["depmap"], list(samp))
        genes = list(O.index)  # genes with valid profiles
        log(f"usable genes (profile + both embeddings): {len(genes)} | profile dim: {O.shape[1]}")

        Ostd = StandardScaler().fit_transform(O.to_numpy(np.float32))
        for ver, emb in [("5L", e5), ("3L", e3)]:
            E = StandardScaler().fit_transform(emb.loc[genes].to_numpy(np.float32))
            rec = knn_recall(Ostd, E, k=args.k)
            tw = trustworthiness(Ostd, E, n_neighbors=args.k)
            ds = dist_spearman(Ostd, E)
            r2 = recon_r2(Ostd, E, seed=args.seed)
            au = coess_auroc(Ostd, E, seed=args.seed)
            rows.append(dict(modality=mod, version=ver, knn_recall=rec, trustworthiness=tw,
                             dist_spearman=ds, recon_R2=r2, coess_auroc=au))
            log(f"  [{ver}] kNN-recall@{args.k}={rec:.3f}  trust={tw:.3f}  "
                f"distSpearman={ds:.3f}  reconR2={r2:.3f}  coessAUROC={au:.3f}")

    df = pd.DataFrame(rows)
    out = os.path.join(REPO, "data", "interim", "embedding_quality_5L_vs_3L.tsv")
    df.to_csv(out, sep="\t", index=False)
    log("\n================ SUMMARY (higher = better for all metrics) ================")
    log(df.to_string(index=False))
    log(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
