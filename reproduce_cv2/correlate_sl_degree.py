"""Correlate hub-excluded base SL degree with the qGI-screen library SL degree.

- hub-excluded base SL degree: per-gene degree in SL_core_network_nohub_base.tsv
- library SL degree: per library_gene, # of GI_stringent == 'negative' in
  data/input/qGI_pairwise_20211111_nRed_standardGIs.txt

Deliverable: per-gene table (hub-excluded base SL degree, library SL degree,
Pearson Correlation Coefficient) + scatter plot.
"""
import os
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = os.path.expanduser("~/DeepSLP")
QGI = f"{REPO}/data/input/qGI_pairwise_20211111_nRed_standardGIs.txt"
NOHUB = f"{REPO}/data/interim/all_pairs_pred/core_network_analysis/SL_core_network_nohub_base.tsv"
OUT = f"{REPO}/data/interim/all_pairs_pred/core_network_analysis"

# ---- library SL degree (# negative GI_stringent per library_gene) ----
q = pd.read_csv(QGI, sep="\t", usecols=["library_gene", "GI_stringent"], low_memory=False)
print("GI_stringent values:", q["GI_stringent"].value_counts(dropna=False).to_dict())
all_lib = pd.Index(q["library_gene"].dropna().unique(), name="gene")
neg = q[q["GI_stringent"].astype(str).str.lower() == "negative"]
lib_deg = neg.groupby("library_gene").size().reindex(all_lib, fill_value=0)

# ---- hub-excluded base SL degree ----
nb = pd.read_csv(NOHUB, sep="\t")
sl_deg = pd.concat([nb["gene1"], nb["gene2"]]).value_counts()

# ---- merge over library genes (the measured universe) ----
df = pd.DataFrame(index=all_lib)
df["hub-excluded base SL degree"] = sl_deg.reindex(all_lib).fillna(0).astype(int)
df["library SL degree"] = lib_deg.astype(int)

r_all, p_all = pearsonr(df["hub-excluded base SL degree"], df["library SL degree"])
rho_all, _ = spearmanr(df["hub-excluded base SL degree"], df["library SL degree"])
# active subset: genes with signal in at least one measure
act = df[(df.iloc[:, 0] > 0) | (df.iloc[:, 1] > 0)]
r_act, p_act = pearsonr(act.iloc[:, 0], act.iloc[:, 1])

df["Pearson Correlation Coefficient"] = r_all
df = df[["hub-excluded base SL degree", "library SL degree", "Pearson Correlation Coefficient"]]
df = df.sort_values("library SL degree", ascending=False)
out_tsv = os.path.join(OUT, "nohub_base_vs_library_SL_degree.tsv")
df.to_csv(out_tsv, sep="\t")

print(f"\nlibrary genes (universe): {len(df):,}")
print(f"  genes with >0 hub-excluded SL degree : {(df.iloc[:,0]>0).sum():,}")
print(f"  genes with >0 library SL degree      : {(df.iloc[:,1]>0).sum():,}")
print(f"Pearson r (all library genes)  = {r_all:+.4f}  (p={p_all:.2e})")
print(f"Spearman rho (all library genes)= {rho_all:+.4f}")
print(f"Pearson r (either-degree>0, n={len(act):,}) = {r_act:+.4f}  (p={p_act:.2e})")

# ---- scatter ----
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(df["library SL degree"], df["hub-excluded base SL degree"], s=10, alpha=0.35, color="#2c7fb8")
ax.set(xlabel="library SL degree (# negative GI_stringent in qGI screen)",
       ylabel="hub-excluded base SL degree (inferred)",
       title=f"Inferred vs measured SL degree\nPearson r = {r_all:.3f} (n={len(df):,} library genes)")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "nohub_base_vs_library_SL_degree_scatter.png"), dpi=150)
plt.close(fig)
print(f"\nsaved -> {out_tsv}")
print(f"saved -> {os.path.join(OUT, 'nohub_base_vs_library_SL_degree_scatter.png')}")
