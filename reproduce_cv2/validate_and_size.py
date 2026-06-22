"""Validate GIV_24Q4 features against AE raw data + size the gene-pair universe."""
import os, glob, numpy as np, pandas as pd

AE = os.path.expanduser("~/DeepSLP/data/input/AE")
KO = os.path.join(AE, "AE128_5L_std100_CRISPRGeneEffect_24Q4_imputed_gene_wise_mean_align_qGI2021.txt")
EXP = os.path.join(AE, "AE128_5L_std100_Expression_BC_24Q4_align_qGI2021.txt")
GIV_DIR = os.path.expanduser("~/DeepSLP/data/input/GIV_24Q4/ReLU128_5L")

df_ko = pd.read_csv(KO, sep="\t", index_col=0)
df_exp = pd.read_csv(EXP, sep="\t", index_col=0)
print(f"AE KO : {df_ko.shape}  (genes x dims)")
print(f"AE EXP: {df_exp.shape}")
overlap = np.intersect1d(df_ko.index, df_exp.index)
print(f"Overlap genes (ko ∩ exp): {len(overlap)}")

def center(row):
    return row.values - row.values.mean()

# Grab one GIV shard, find first row whose Gene & Query are both in overlap
shard = sorted(glob.glob(os.path.join(GIV_DIR, "*.tsv")))[0]
df = pd.read_csv(shard, sep="\t", nrows=50, low_memory=False)
ko_cols = [c for c in df.columns if c.startswith("ko_")]
exp_cols = [c for c in df.columns if c.startswith("exp_")]
print(f"\nShard: {os.path.basename(shard)} | ko cols={len(ko_cols)} exp cols={len(exp_cols)}")

checked = 0
for _, r in df.iterrows():
    g1, g2 = r["Gene"], r["Query"]
    if g1 in overlap and g2 in overlap and not pd.isna(r[ko_cols[0]]):
        ko_pred = center(df_ko.loc[g1]) + center(df_ko.loc[g2])
        exp_pred = center(df_exp.loc[g1]) + center(df_exp.loc[g2])
        ko_stored = r[ko_cols].values.astype(float)
        exp_stored = r[exp_cols].values.astype(float)
        ko_err = np.abs(ko_pred - ko_stored).max()
        exp_err = np.abs(exp_pred - exp_stored).max()
        print(f"\nPair ({g1}, {g2}):")
        print(f"  ko : max|stored-recomputed| = {ko_err:.3e}  (stored[:3]={np.round(ko_stored[:3],4)}, recomputed[:3]={np.round(ko_pred[:3],4)})")
        print(f"  exp: max|stored-recomputed| = {exp_err:.3e}")
        checked += 1
        if checked >= 3:
            break

# Sizing
N = len(overlap)
n_pairs = N * (N - 1) // 2
print("\n================ GENE-PAIR UNIVERSE SIZING ================")
print(f"Genes with AE embeddings (N) = {N}")
print(f"All unique unordered pairs C(N,2) = {n_pairs:,}")
for dt, b in [("float32", 4), ("float16", 2)]:
    print(f"  feature store 256-dim {dt}: {n_pairs*256*b/1e9:.1f} GB")
print(f"  per-gene precomputed 256-dim float32: {N*256*4/1e6:.1f} MB")
print(f"  predictions table (gene1,gene2,score ~24 B/row): {n_pairs*24/1e9:.1f} GB")
