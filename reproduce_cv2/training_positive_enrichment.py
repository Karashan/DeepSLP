"""(b) Are the training SL positives enriched for the hub genes?

For each gene, computes from the training GIV_24Q4 shards (label GI_stringent_Type2)
its number of positive SL pairs and positive rate, then correlates with the
model's solo/marginal propensity and checks the hub genes specifically.
"""
import os, glob, numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

REPO = os.path.expanduser("~/DeepSLP")
SHARDS = f"{REPO}/data/input/GIV_24Q4_3L"
DRIVERS = f"{REPO}/data/interim/all_pairs_pred/hcls1_investigation/per_gene_drivers.tsv"
OUT = f"{REPO}/data/interim/all_pairs_pred/hcls1_investigation"
LABEL = "GI_stringent_Type2"


def main():
    frames = []
    for f in sorted(glob.glob(os.path.join(SHARDS, "*.tsv"))):
        frames.append(pd.read_csv(f, sep="\t", usecols=["Gene", "Query", LABEL], low_memory=False))
    df = pd.concat(frames, ignore_index=True).dropna(subset=["Gene", "Query", LABEL])
    df = df[df["Gene"] != df["Query"]]
    print(f"training pairs: {len(df):,} | positives: {int(df[LABEL].sum()):,} "
          f"({df[LABEL].mean():.4%})")

    # per-gene: total appearances and positive appearances (as Gene or Query)
    stacked = pd.concat([
        df[["Gene", LABEL]].rename(columns={"Gene": "gene"}),
        df[["Query", LABEL]].rename(columns={"Query": "gene"}),
    ], ignore_index=True)
    g = stacked.groupby("gene")[LABEL].agg(n_pairs="count", n_pos="sum")
    g["pos_rate"] = g["n_pos"] / g["n_pairs"]
    g = g.reset_index()

    drv = pd.read_csv(DRIVERS, sep="\t")
    m = drv.merge(g, on="gene", how="inner")
    print(f"genes with solo + training stats: {len(m)}")

    lines = ["", "## 7. Training-positive enrichment of the hub genes",
             f"training pairs={len(df):,}, positives={int(df[LABEL].sum()):,} "
             f"({df[LABEL].mean():.4%}), genes={len(m):,}", "",
             "Spearman correlation of model propensity with training SL frequency:",
             "| propensity | vs n_pos | vs pos_rate |", "|---|---|---|"]
    for col in ["solo", "marginal_m", "solo_exp"]:
        r1 = spearmanr(m[col], m["n_pos"])[0]
        r2 = spearmanr(m[col], m["pos_rate"])[0]
        lines.append(f"| {col} | {r1:+.3f} | {r2:+.3f} |")

    # are top-solo genes enriched among training positives?
    top = set(m.sort_values("solo", ascending=False).head(200)["gene"])
    m["top200_solo"] = m["gene"].isin(top)
    a = m[m.top200_solo]["pos_rate"]; b = m[~m.top200_solo]["pos_rate"]
    u, p = mannwhitneyu(a, b, alternative="greater")
    lines += ["",
              f"Training SL positive-rate: top-200 solo genes median={a.median():.4f} "
              f"vs rest median={b.median():.4f} (Mann-Whitney one-sided p={p:.1e})",
              f"  mean: top-200={a.mean():.4f} vs rest={b.mean():.4f} "
              f"(x{a.mean()/max(b.mean(),1e-9):.1f})",
              "", "Hub genes — training SL stats vs model propensity:",
              f"{'gene':<10}{'n_pairs':>9}{'n_pos':>7}{'pos_rate':>10}{'solo':>8}{'marg':>8}"]
    pop_rate = m["pos_rate"].median()
    for gg in ["HCLS1", "KRT5", "ARHGDIB", "LCP1", "KRT14", "KRT6A", "IL1B", "CSTA",
               "SERPINB2", "NCKAP1L"]:
        r = m[m.gene == gg]
        if len(r):
            r = r.iloc[0]
            lines.append(f"{gg:<10}{int(r.n_pairs):>9}{int(r.n_pos):>7}{r.pos_rate:>10.4f}"
                         f"{r.solo:>8.3f}{r.marginal_m:>8.3f}")
    lines += ["", f"population median pos_rate = {pop_rate:.4f}"]
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "training_enrichment_summary.txt"), "w") as fh:
        fh.write(txt + "\n")
    m.to_csv(os.path.join(OUT, "solo_vs_training_pos.tsv"), sep="\t", index=False)
    print(f"\nsaved -> {OUT}/training_enrichment_summary.txt, solo_vs_training_pos.tsv")


if __name__ == "__main__":
    main()
