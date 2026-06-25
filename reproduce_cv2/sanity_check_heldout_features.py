"""Sanity check: were the held-out GIV features built from the same AE files
(data/input/AE/) and the same method as the training GIV features?

Strategy
--------
1. Load the per-gene AE embeddings (CRISPR/KO and Expression).
2. For a sample of TRAINING pairs, find which simple construction of the two
   genes' embeddings reproduces the pair's ko_* / exp_* features (query-only,
   library-only, product, difference, mean, ...).
3. Apply the SAME best-matching construction to a sample of HELD-OUT pairs.
4. If training reconstructs (≈0 error) but held-out does not, the held-out
   features were generated differently / from a different source.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AE = os.path.join(REPO, "data", "input", "AE")
KO = [f"ko_{i}" for i in range(128)]
EXP = [f"exp_{i}" for i in range(128)]

def load_ae(fname):
    df = pd.read_csv(os.path.join(AE, fname), sep="\t", index_col=0)
    df.columns = [int(c) for c in df.columns]
    return df  # index=gene, cols 0..127

def first_shard(d, pattern):
    import glob
    fs = sorted(glob.glob(os.path.join(REPO, d, pattern)),
                key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return fs[0]

def candidates(qv, lv):
    """Return dict of candidate 128-d constructions from query/library vectors."""
    return {
        "query_only": qv,
        "library_only": lv,
        "product": qv * lv,
        "difference_q_minus_l": qv - lv,
        "difference_l_minus_q": lv - qv,
        "mean": (qv + lv) / 2.0,
        "sum": qv + lv,
        "abs_difference": np.abs(qv - lv),
    }

def test_dataset(name, shard_path, crispr, expr, n=2000):
    df = pd.read_csv(shard_path, sep="\t", low_memory=False)
    # only rows whose both genes exist in both AE tables
    gset_c, gset_e = set(crispr.index), set(expr.index)
    ok = df["Query"].isin(gset_c) & df["Gene"].isin(gset_c) & \
         df["Query"].isin(gset_e) & df["Gene"].isin(gset_e)
    df = df[ok]
    cover = ok.mean()
    df = df.head(n)
    print(f"\n===== {name} =====")
    print(f"  rows with both genes in BOTH AE tables: {cover*100:.1f}%  (testing {len(df)} pairs)")
    if len(df) == 0:
        print("  -> no testable pairs (genes not found in AE files!)")
        return

    Cq = crispr.loc[df["Query"]].to_numpy(); Cl = crispr.loc[df["Gene"]].to_numpy()
    Eq = expr.loc[df["Query"]].to_numpy();   El = expr.loc[df["Gene"]].to_numpy()
    KOf = df[KO].to_numpy(float); EXPf = df[EXP].to_numpy(float)

    for block, (qv, lv, feat) in {
        "KO (ko_*) vs CRISPR-AE": (Cq, Cl, KOf),
        "EXP (exp_*) vs Expression-AE": (Eq, El, EXPf),
    }.items():
        print(f"  -- {block} --")
        best = None
        for cname in candidates(qv[0], lv[0]).keys():
            cand = candidates(qv, lv)[cname]
            # mean absolute error per element, and correlation
            mae = np.nanmean(np.abs(cand - feat))
            # guard correlation
            cc = np.corrcoef(cand.ravel(), feat.ravel())[0, 1]
            tag = ""
            if best is None or mae < best[1]:
                best = (cname, mae, cc)
            print(f"       {cname:24s} MAE={mae:.4f}  corr={cc:+.3f}")
        print(f"     -> best match: {best[0]} (MAE={best[1]:.4f}, corr={best[2]:+.3f})")


def main():
    crispr = load_ae("AE128_5L_std100_CRISPRGeneEffect_24Q4_imputed_gene_wise_mean_align_qGI2021.txt")
    expr = load_ae("AE128_5L_std100_Expression_BC_24Q4_align_qGI2021.txt")
    print(f"AE CRISPR: {crispr.shape}, AE Expression: {expr.shape}")

    tr = first_shard("data/input/GIV_24Q4/ReLU128_5L", "qGI_24Q4_GIV_ReLU128_5L_*.tsv")
    he = first_shard("data/input/GIV_24Q4_heldout", "qGI_heldout_24Q4_GIV_*.tsv")
    test_dataset("TRAINING shard", tr, crispr, expr)
    test_dataset("HELD-OUT shard", he, crispr, expr)


if __name__ == "__main__":
    main()
