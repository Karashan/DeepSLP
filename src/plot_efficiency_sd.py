"""Regenerate test3 efficiency plots as two separate PDFs with SD error bars.

Reads the two TSV outputs produced by
``src.evaluate_extended.test3_efficiency_budget``
(``efficiency_sl_pairs.tsv`` and ``efficiency_coverage.tsv``) and produces
two separate PDF figures (dpi=300):

* ``efficiency_sl_pairs_sd.pdf`` – queries to reach SL-pair milestones.
* ``efficiency_coverage_sd.pdf`` – queries to reach library-gene coverage
  milestones.

Error bars on the random baseline use the standard deviation
(``random_sd`` column in the TSVs), matching the original
``efficiency.png``. No permutations are re-run.

Usage
-----
    python -m src.plot_efficiency_sd \\
        --input-dir outputs/external_val/external_val_old_models/extended_eval/test3_efficiency
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _plot_one(
    df: pd.DataFrame,
    target_col: str,
    x_label: str,
    title: str,
    out_path: Path,
) -> None:
    x = np.arange(len(df))
    w = 0.35

    guided = df["guided_queries"].to_numpy(dtype=float)
    rand_mean = df["random_mean_queries"].to_numpy(dtype=float)
    rand_sd = df["random_sd"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.bar(x - w / 2, guided, w, label="Guided", color="#1f77b4")
    ax.bar(
        x + w / 2,
        rand_mean,
        w,
        label="Random",
        color="grey",
        yerr=[rand_sd, rand_sd],
        capsize=4,
        error_kw={"lw": 1},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{int(p)}%\n({int(t)})" for p, t in zip(df["target_pct"], df[target_col])]
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("# Queries needed")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "outputs/external_val/external_val_old_models/"
            "extended_eval/test3_efficiency"
        ),
        help="Directory containing efficiency_sl_pairs.tsv and "
             "efficiency_coverage.tsv. Output PDFs are written here too.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the PDFs (defaults to --input-dir).",
    )
    args = ap.parse_args()

    in_dir: Path = args.input_dir
    out_dir: Path = args.output_dir or in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sl_df = pd.read_csv(in_dir / "efficiency_sl_pairs.tsv", sep="\t")
    cov_df = pd.read_csv(in_dir / "efficiency_coverage.tsv", sep="\t")

    _plot_one(
        sl_df,
        target_col="target_sl_pairs",
        x_label="Target (% of total SL pairs)",
        title="Queries to Reach SL-Pair Milestones",
        out_path=out_dir / "efficiency_sl_pairs_sd.pdf",
    )
    _plot_one(
        cov_df,
        target_col="target_genes",
        x_label="Target (% of total library genes)",
        title="Queries to Reach Coverage Milestones",
        out_path=out_dir / "efficiency_coverage_sd.pdf",
    )


if __name__ == "__main__":
    main()
