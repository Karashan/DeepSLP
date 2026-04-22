"""Plot a simplified Library Gene Coverage prioritization curve (v2).

Reads the pre-computed ``library_coverage_curves.tsv`` and produces a PDF
figure with the following modifications compared to the original:

1. Only the "greedy coverage" oracle is shown (no "SL count" oracle).
2. The "mean pred_prob" and "top-100 pred_prob" guided lines are removed.
3. Output is PDF at 300 dpi.

Usage
-----
    python -m src.plot_library_coverage_v2 \
        --input outputs/external_val/external_val_old_models/query_prioritization/library_coverage_curves.tsv \
        --output outputs/external_val/external_val_old_models/query_prioritization/library_coverage_v2.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot simplified library coverage curve (v2)")
    p.add_argument(
        "--input",
        default="outputs/external_val/external_val_old_models/query_prioritization/library_coverage_curves.tsv",
        help="Path to library_coverage_curves.tsv",
    )
    p.add_argument(
        "--output",
        default="outputs/external_val/external_val_old_models/query_prioritization/library_coverage_v2.pdf",
        help="Output PDF path",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, sep="\t")

    ks = df["k"].values
    total_queries = int(ks[-1])

    # Random baseline
    random_mean = df["random_mean"].values
    random_sd_lo = df["random_sd_lo"].values
    random_sd_hi = df["random_sd_hi"].values

    # Guided strategies to keep
    guided = {
        "sum_proba":     {"color": "#1f77b4", "label": "Guided (Σ pred_prob)"},
        "hits_baserate": {"color": "#9467bd", "label": "Guided (# pred > base-rate cutoff)"},
    }

    # Oracle to keep (greedy coverage only)
    oracle = {
        "oracle_cov": {"color": "#8B0000", "ls": "-.", "lw": 1.8, "label": "Oracle (greedy coverage)"},
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Random band
    ax.fill_between(ks, random_sd_lo, random_sd_hi,
                    alpha=0.15, color="grey", label="Random ± 1 SD")
    ax.plot(ks, random_mean, color="grey", ls="--", lw=1.5, label="Random")

    # Guided lines
    for col, style in guided.items():
        ax.plot(ks, df[col].values, color=style["color"], lw=2, label=style["label"])

    # Oracle line
    for col, style in oracle.items():
        ax.plot(ks, df[col].values, color=style["color"],
                ls=style["ls"], lw=style["lw"], label=style["label"])

    ax.set_xlabel("Number of queries screened")
    ax.set_ylabel("Unique library genes covered")
    ax.set_title("Model-Guided vs Random Query Selection:\nLibrary Gene Coverage")
    ax.set_xlim(0, total_queries)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
