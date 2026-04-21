"""Side-by-side K562 vs Jurkat comparison of GIV vs LLM features.

Loads per-seed performance TSVs from:
  - K562  GIV: <k562_giv_root>/K562/CV{1,2,3}/*_performance_stats_10runs.tsv
  - K562  LLM: <k562_llm_root>/K562/repeats/CV{1,2,3}/*_performance_stats_10runs.tsv
  - Jurkat GIV: <jurkat_giv_root>/Jurkat/CV{1,2,3}/*_performance_stats_10runs.tsv
  - Jurkat LLM: <jurkat_llm_root>/Jurkat/CV{1,2,3}/*_performance_stats_10runs.tsv

and writes:
  - side-by-side bar plots (K562 | Jurkat) per metric, with SEM error bars;
  - a per-seed long-form TSV;
  - a paired statistical test TSV (paired t-test + Wilcoxon signed-rank,
    pairing GIV vs LLM by seed within each (cell_line, CV, metric)).
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sstats


def _find_stats_file(feature_root: str, cell_line: str, cv: int) -> str:
    patterns = [
        os.path.join(feature_root, cell_line, f"CV{cv}", "*_performance_stats_*runs.tsv"),
        os.path.join(feature_root, cell_line, "repeats", f"CV{cv}", "*_performance_stats_*runs.tsv"),
    ]
    for p in patterns:
        matches = sorted(glob.glob(p), key=os.path.getmtime, reverse=True)
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No stats file for {cell_line} CV{cv} under {feature_root}")


def load_perseed(feature_roots: Dict[str, Dict[str, str]], cv_list: List[int]) -> pd.DataFrame:
    """feature_roots = {cell_line: {feature: root_path}}."""
    rows = []
    for cell_line, per_feature in feature_roots.items():
        for feature, root in per_feature.items():
            for cv in cv_list:
                path = _find_stats_file(root, cell_line, cv)
                df = pd.read_csv(path, sep="\t")
                metric_cols = [c for c in df.columns if c != "seed"]
                melt = df.melt(id_vars=["seed"], value_vars=metric_cols,
                               var_name="metric", value_name="value")
                melt["feature"] = feature
                melt["cell_line"] = cell_line
                melt["cv"] = f"CV{cv}"
                melt["source_file"] = path
                rows.append(melt)
                print(f"[{cell_line}][{feature}][CV{cv}] n_runs={len(df)} | {path}")
    return pd.concat(rows, ignore_index=True)


def plot_side_by_side(df_long: pd.DataFrame, metric: str,
                      cell_line_order=("K562", "Jurkat"),
                      cv_list=("CV1", "CV2", "CV3"),
                      feature_order=("GIV", "LLM"),
                      colors=("#4C72B0", "#DD8452"),
                      save_path: str = None):
    sub = df_long[df_long["metric"] == metric]
    if sub.empty:
        raise ValueError(f"No data for metric {metric!r}")

    g = (sub.groupby(["cell_line", "feature", "cv"])["value"]
            .agg(mean="mean", std="std", count="count").reset_index())
    g["sem"] = g["std"] / np.sqrt(g["count"])

    fig, axes = plt.subplots(1, len(cell_line_order),
                              figsize=(5.5 * len(cell_line_order), 4.2),
                              sharey=True)
    if len(cell_line_order) == 1:
        axes = [axes]

    x = np.arange(len(cv_list))
    bar_width = 0.38
    offsets = np.linspace(-bar_width / 2, bar_width / 2, len(feature_order))

    for ax, cl in zip(axes, cell_line_order):
        gcl = g[g["cell_line"] == cl]
        for i, feat in enumerate(feature_order):
            means, sems, xs = [], [], []
            for j, cv in enumerate(cv_list):
                row = gcl[(gcl["feature"] == feat) & (gcl["cv"] == cv)]
                if row.empty:
                    means.append(np.nan); sems.append(np.nan)
                else:
                    means.append(row["mean"].values[0])
                    sems.append(row["sem"].values[0])
                xs.append(x[j] + offsets[i])
            ax.bar(xs, means, width=bar_width, yerr=sems, capsize=4,
                   color=colors[i % len(colors)], label=feat, alpha=0.9,
                   edgecolor="black", linewidth=0.6)
            # overlay individual per-seed points
            pts = sub[(sub["cell_line"] == cl) & (sub["feature"] == feat)]
            for j, cv in enumerate(cv_list):
                vals = pts[pts["cv"] == cv]["value"].values
                jitter = (np.random.RandomState(0).uniform(-0.05, 0.05, len(vals)))
                ax.scatter(np.full(len(vals), xs[j]) + jitter, vals,
                           s=12, color="black", alpha=0.35, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(cv_list)
        ax.set_title(cl)
        ax.set_xlabel("CV strategy")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    axes[0].set_ylabel(metric)
    axes[-1].legend(title="Feature", loc="best")
    fig.suptitle(f"{metric} — K562 vs Jurkat (mean ± SEM, per-seed points)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig, axes, g


def paired_tests(df_long: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """Paired t-test + Wilcoxon signed-rank, GIV vs LLM per (cell_line, CV, metric)."""
    rows = []
    for cl in df_long["cell_line"].unique():
        for cv in sorted(df_long["cv"].unique()):
            for metric in metrics:
                sub = df_long[(df_long["cell_line"] == cl) &
                              (df_long["cv"] == cv) &
                              (df_long["metric"] == metric)]
                giv = sub[sub["feature"] == "GIV"].set_index("seed")["value"]
                llm = sub[sub["feature"] == "LLM"].set_index("seed")["value"]
                common = giv.index.intersection(llm.index)
                if len(common) < 2:
                    continue
                g = giv.loc[common].values
                l = llm.loc[common].values
                diff = g - l
                t = sstats.ttest_rel(g, l)
                # Wilcoxon breaks on all-zero diffs; guard.
                try:
                    w = sstats.wilcoxon(g, l, zero_method="wilcox",
                                         alternative="two-sided")
                    w_stat, w_p = float(w.statistic), float(w.pvalue)
                except ValueError:
                    w_stat, w_p = float("nan"), float("nan")
                rows.append({
                    "cell_line": cl, "cv": cv, "metric": metric, "n": len(common),
                    "GIV_mean": g.mean(), "LLM_mean": l.mean(),
                    "diff_mean": diff.mean(), "diff_sem": diff.std(ddof=1) / np.sqrt(len(diff)),
                    "t_stat": float(t.statistic), "t_pvalue": float(t.pvalue),
                    "wilcoxon_stat": w_stat, "wilcoxon_pvalue": w_p,
                    "winner": "GIV" if diff.mean() > 0 else ("LLM" if diff.mean() < 0 else "tie"),
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k562-giv-root", required=True)
    ap.add_argument("--k562-llm-root", required=True)
    ap.add_argument("--jurkat-giv-root", required=True)
    ap.add_argument("--jurkat-llm-root", required=True)
    ap.add_argument("--cvs", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--metrics", nargs="+",
                    default=["AUROC", "Precision@100", "AUPR", "Recall@100", "AP"])
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    feature_roots = {
        "K562":   {"GIV": args.k562_giv_root,   "LLM": args.k562_llm_root},
        "Jurkat": {"GIV": args.jurkat_giv_root, "LLM": args.jurkat_llm_root},
    }

    os.makedirs(args.output_dir, exist_ok=True)
    df_long = load_perseed(feature_roots, args.cvs)
    long_path = os.path.join(args.output_dir, "K562_vs_Jurkat_perseed_long.tsv")
    df_long.to_csv(long_path, sep="\t", index=False)
    print(f"Wrote: {long_path}")

    for m in args.metrics:
        safe = m.replace("@", "at").replace("/", "_")
        plot_side_by_side(df_long, metric=m,
                          save_path=os.path.join(args.output_dir,
                                                 f"K562_vs_Jurkat_{safe}_bar.pdf"))

    stats_df = paired_tests(df_long, args.metrics)
    stats_path = os.path.join(args.output_dir, "K562_vs_Jurkat_paired_tests.tsv")
    stats_df.to_csv(stats_path, sep="\t", index=False)
    print(f"Wrote: {stats_path}")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
