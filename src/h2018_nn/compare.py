"""Cross-feature (GIV vs LLM) comparison: load per-run stats and plot bars."""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


STATS_GLOB = "*performance_stats_*runs.tsv"
DEFAULT_CVS = ["CV1", "CV2", "CV3"]


def _find_stats_file(feature_root: str, cell_line: str, cv: str) -> str:
    """Locate the `*_performance_stats_*runs.tsv` file for one (feature, cell_line, cv).

    Supports both layouts we've written from notebooks/pipelines:
        {feature_root}/{cell_line}/{cv}/*.tsv
        {feature_root}/{cell_line}/repeats/{cv}/*.tsv
    """
    patterns = [
        os.path.join(feature_root, cell_line, cv, STATS_GLOB),
        os.path.join(feature_root, cell_line, "repeats", cv, STATS_GLOB),
    ]
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            if len(matches) > 1:
                matches.sort(key=os.path.getmtime)
            return matches[-1]
    raise FileNotFoundError(
        f"No stats file found for {cv} under any of: {patterns}")


def load_perf_stats(feature_roots: Dict[str, str],
                    cell_line: str = "K562",
                    cv_list: Sequence[str] = DEFAULT_CVS,
                    verbose: bool = True) -> pd.DataFrame:
    """Return a long-form DataFrame with columns:
    `feature, cv, seed, metric, value, source_file`.
    """
    rows = []
    for feat_name, feat_root in feature_roots.items():
        for cv in cv_list:
            path = _find_stats_file(feat_root, cell_line, cv)
            df = pd.read_csv(path, sep="\t")
            metric_cols = [c for c in df.columns if c != "seed"]
            id_vars = ["seed"] if "seed" in df.columns else []
            long = df.melt(id_vars=id_vars, value_vars=metric_cols,
                           var_name="metric", value_name="value")
            long["feature"] = feat_name
            long["cv"] = cv
            long["source_file"] = path
            rows.append(long)
            if verbose:
                print(f"[{feat_name}][{cv}] n_runs={len(df)} | {path}")
    return pd.concat(rows, ignore_index=True)


def plot_metric_bar(df_long: pd.DataFrame, metric: str,
                    cv_list: Sequence[str] = DEFAULT_CVS,
                    feature_order: Tuple[str, ...] = ("GIV", "LLM"),
                    colors: Tuple[str, ...] = ("#4C72B0", "#DD8452"),
                    bar_width: float = 0.35,
                    figsize=(7, 4.5),
                    show_points: bool = True,
                    save_path: Optional[str] = None,
                    ax=None):
    """Grouped bar plot for one metric: N CV groups × M feature sources.

    Bar height = mean across runs; error bars = SEM (std / sqrt(n_runs)).
    Returns (fig, ax, stats_df).
    """
    sub = df_long[df_long["metric"] == metric]
    if sub.empty:
        raise ValueError(
            f"metric '{metric}' not found. Available: {sorted(df_long['metric'].unique())}")

    stats = (sub.groupby(["feature", "cv"])["value"]
               .agg(["mean", "std", "count"]).reset_index())
    stats["sem"] = stats["std"] / np.sqrt(stats["count"])

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = np.arange(len(cv_list))
    offsets = (np.linspace(-(len(feature_order) - 1) / 2,
                            (len(feature_order) - 1) / 2,
                            len(feature_order)) * bar_width)

    for i, feat in enumerate(feature_order):
        means, sems = [], []
        for cv in cv_list:
            row = stats[(stats["feature"] == feat) & (stats["cv"] == cv)]
            means.append(row["mean"].values[0] if not row.empty else np.nan)
            sems.append(row["sem"].values[0]  if not row.empty else np.nan)
        ax.bar(x + offsets[i], means, width=bar_width, yerr=sems, capsize=4,
               label=feat, color=colors[i % len(colors)], edgecolor="black",
               linewidth=0.6)

        if show_points:
            for j, cv in enumerate(cv_list):
                pts = sub[(sub["feature"] == feat) & (sub["cv"] == cv)]["value"].values
                if len(pts) == 0:
                    continue
                jitter = np.random.RandomState(0).uniform(-0.06, 0.06, size=len(pts))
                ax.scatter(np.full_like(pts, x[j] + offsets[i], dtype=float) + jitter,
                           pts, s=14, color="black", alpha=0.55, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(cv_list)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} — mean ± SEM ({', '.join(feature_order)})")
    ax.legend(title="Features", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print("Saved:", save_path)
    return fig, ax, stats


def plot_all_metrics_grid(df_long: pd.DataFrame,
                          metrics: Optional[List[str]] = None,
                          ncols: int = 3,
                          save_path: Optional[str] = None,
                          **kwargs):
    """Render one subplot per metric in a grid."""
    if metrics is None:
        order = ["AUROC", "AUPR", "AP"]
        metrics = sorted(df_long["metric"].unique(),
                         key=lambda m: order.index(m) if m in order else 99)
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.2 * nrows))
    axes = np.atleast_2d(axes).ravel()
    for i, m in enumerate(metrics):
        plot_metric_bar(df_long, metric=m, ax=axes[i], **kwargs)
    for j in range(len(metrics), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print("Saved:", save_path)
    return fig, axes
