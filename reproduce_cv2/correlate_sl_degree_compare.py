"""Compare inferred SL degree (hub-INCLUDED vs hub-EXCLUDED base) against the
qGI-screen library SL degree, and produce a final comparison table.
"""
import os
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = os.path.expanduser("~/DeepSLP")
QGI = f"{REPO}/data/input/qGI_pairwise_20211111_nRed_standardGIs.txt"
D = f"{REPO}/data/interim/all_pairs_pred/core_network_analysis"
BASE_INCL = f"{REPO}/data/interim/all_pairs_pred/SL_core_network_cutoff0.162.tsv"
BASE_EXCL = f"{D}/SL_core_network_nohub_base.tsv"
HUBS = {"UBR4","PAXIP1","UNC50","RAB18","PDS5A","DNAJC9","TSSC1","CTNNA1","RIC8A","INTS6"}


def net_degree(path):
    nb = pd.read_csv(path, sep="\t")
    return pd.concat([nb["gene1"], nb["gene2"]]).value_counts()


def main():
    # library SL degree
    q = pd.read_csv(QGI, sep="\t", usecols=["library_gene", "GI_stringent"], low_memory=False)
    all_lib = pd.Index(q["library_gene"].dropna().unique(), name="gene")
    neg = q[q["GI_stringent"].astype(str).str.lower() == "negative"]
    lib_deg = neg.groupby("library_gene").size().reindex(all_lib, fill_value=0)

    incl = net_degree(BASE_INCL).reindex(all_lib).fillna(0).astype(int)
    excl = net_degree(BASE_EXCL).reindex(all_lib).fillna(0).astype(int)

    df = pd.DataFrame({
        "hub-included base SL degree": incl,
        "hub-excluded base SL degree": excl,
        "library SL degree": lib_deg.astype(int),
    }, index=all_lib).sort_values("library SL degree", ascending=False)
    df.to_csv(os.path.join(D, "base_SL_degree_vs_library_merged.tsv"), sep="\t")

    def stats(col):
        x = df[col].values; y = df["library SL degree"].values
        r_all, p_all = pearsonr(x, y)
        rho_all, _ = spearmanr(x, y)
        m = (x > 0) | (y > 0)
        r_act, p_act = pearsonr(x[m], y[m])
        return dict(pearson_r_all=r_all, p_all=p_all, spearman_rho_all=rho_all,
                    pearson_r_active=r_act, n_active=int(m.sum()),
                    n_genes_inferred_pos=int((x > 0).sum()))

    rows = []
    for name, col in [("hub-included base (0.162)", "hub-included base SL degree"),
                      ("hub-excluded base", "hub-excluded base SL degree")]:
        s = stats(col); s = {"version": name, **s}; rows.append(s)
    comp = pd.DataFrame(rows)[["version", "pearson_r_all", "spearman_rho_all",
                               "pearson_r_active", "n_active", "n_genes_inferred_pos", "p_all"]]
    comp.to_csv(os.path.join(D, "sl_degree_comparison_table.tsv"), sep="\t", index=False)

    print("library genes universe:", len(df))
    print("\n=== FINAL COMPARISON TABLE ===")
    with pd.option_context("display.width", 160, "display.float_format", lambda v: f"{v:.4f}"):
        print(comp.to_string(index=False))

    # side-by-side scatter
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
    for a, (name, col, r) in zip(ax, [
        ("hub-included base (0.162)", "hub-included base SL degree", rows[0]["pearson_r_all"]),
        ("hub-excluded base", "hub-excluded base SL degree", rows[1]["pearson_r_all"])]):
        a.scatter(df["library SL degree"], df[col], s=10, alpha=0.35, color="#2c7fb8")
        a.set(xlabel="library SL degree (qGI negatives)", ylabel="inferred SL degree",
              title=f"{name}\nPearson r = {r:.3f}")
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(D, "sl_degree_comparison_scatter.png"), dpi=150)
    plt.close(fig)
    print(f"\nsaved -> {D}/base_SL_degree_vs_library_merged.tsv")
    print(f"saved -> {D}/sl_degree_comparison_table.tsv")
    print(f"saved -> {D}/sl_degree_comparison_scatter.png")


if __name__ == "__main__":
    main()
