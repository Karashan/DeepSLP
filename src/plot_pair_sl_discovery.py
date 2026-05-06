"""Pair-level SL discovery prioritization curve.

Question
--------
If we test gene pairs one-by-one in order of model-predicted probability,
how quickly do we accumulate true SL pairs (label = 1), compared to:

* **Random** selection (analytical mean ± 1 SD, hypergeometric).
* **Oracle** selection (sort by true label, upper bound).

x-axis : number of pairs screened (1 ... N)
y-axis : cumulative number of true SL pairs found

Tie handling
------------
Many pairs share the same ``predict_proba`` value. Rather than break ties
by an arbitrary secondary key, we compute the **expected** cumulative SL
count under uniformly-random tie-breaking. For a tie group of size ``n_g``
containing ``k_g`` positives, the curve increases linearly from
``cum_before`` to ``cum_before + k_g`` over the ``n_g`` x-steps.
This gives a deterministic, well-defined curve.

Inputs
------
TSV with columns ``GI_stringent_Type2`` (label, 0/1) and ``predict_proba``.

Outputs
-------
PDF figure at 300 dpi.

Usage
-----
    python -m src.plot_pair_sl_discovery \
        --predictions outputs/external_val/external_val_old_models/ensemble_predictions.tsv \
        --output      outputs/external_val/external_val_old_models/query_prioritization/pair_sl_discovery.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pair-level SL discovery prioritization plot")
    p.add_argument(
        "--predictions",
        default="outputs/external_val/external_val_old_models/ensemble_predictions.tsv",
        help="Ensemble predictions TSV with predict_proba and label columns",
    )
    p.add_argument("--label-col", default="GI_stringent_Type2")
    p.add_argument("--score-col", default="predict_proba")
    p.add_argument("--gene-col", default="Gene")
    p.add_argument("--query-col", default="Query")
    p.add_argument(
        "--gene-subset",
        default=None,
        help="Optional tab-separated file whose first column is a list of genes "
             "(no header). Restrict evaluation to pairs whose Gene is in this "
             "list (Query side keeps all genes). If omitted, use all pairs.",
    )
    p.add_argument(
        "--output",
        default="outputs/external_val/external_val_old_models/query_prioritization/pair_sl_discovery.pdf",
        help="Output PDF path",
    )
    p.add_argument(
        "--max-points", type=int, default=5000,
        help="Sub-sample plotted curve to this many points for a smaller PDF "
             "(curve values are kept exact at the sampled x-positions).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------

def guided_curve_with_ties(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Expected cumulative positives under random tie-breaking.

    Returns an array of length N+1: curve[k] = expected #positives in the
    first k pairs when pairs are sorted by ``scores`` (desc), with ties
    broken uniformly at random.
    """
    N = scores.shape[0]
    # Sort by score descending. For tied scores any order yields identical
    # group composition, so a single sort suffices.
    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = labels[order].astype(np.float64)

    curve = np.empty(N + 1, dtype=np.float64)
    curve[0] = 0.0

    # Identify tie group boundaries
    # group_starts[i] = index in s_sorted where the i-th group begins
    diffs = np.diff(s_sorted)
    boundary = np.flatnonzero(diffs != 0) + 1
    group_starts = np.concatenate(([0], boundary, [N]))

    cum = 0.0
    for gi in range(len(group_starts) - 1):
        a, b = group_starts[gi], group_starts[gi + 1]
        n_g = b - a
        k_g = float(y_sorted[a:b].sum())
        # Linearly interpolate within the tie group
        # curve[a+1 .. b] = cum + (1..n_g) * k_g / n_g
        steps = np.arange(1, n_g + 1, dtype=np.float64)
        curve[a + 1 : b + 1] = cum + steps * (k_g / n_g)
        cum += k_g

    return curve


def oracle_curve(labels: np.ndarray) -> np.ndarray:
    """Cumulative positives if we screen all positives first (upper bound)."""
    N = labels.shape[0]
    P = int(labels.sum())
    curve = np.empty(N + 1, dtype=np.float64)
    curve[0] = 0.0
    ks = np.arange(1, N + 1)
    curve[1:] = np.minimum(ks, P).astype(np.float64)
    return curve


def random_curve_analytical(N: int, P: int) -> tuple[np.ndarray, np.ndarray]:
    """Analytical mean and SD for sampling-without-replacement (hypergeometric).

    For drawing k pairs from N (P positives):
      mean = k * P / N
      var  = k * (P/N) * (N-P)/N * (N-k)/(N-1)
    """
    k = np.arange(N + 1, dtype=np.float64)
    p = P / N
    mean = k * p
    if N > 1:
        var = k * p * (1.0 - p) * (N - k) / (N - 1)
    else:
        var = np.zeros_like(k)
    var = np.clip(var, 0.0, None)  # numerical safety at endpoints
    sd = np.sqrt(var)
    return mean, sd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _subsample_indices(N_plus_1: int, max_points: int) -> np.ndarray:
    """Indices into 0..N (inclusive) that always include 0 and N."""
    if N_plus_1 <= max_points:
        return np.arange(N_plus_1)
    idx = np.unique(np.linspace(0, N_plus_1 - 1, max_points).astype(np.int64))
    return idx


def main() -> None:
    args = parse_args()

    print(f"Loading {args.predictions} ...")
    usecols = [args.label_col, args.score_col]
    if args.gene_subset is not None:
        usecols = [args.gene_col] + usecols
    df = pd.read_csv(args.predictions, sep="\t", usecols=usecols)
    print(f"  Loaded {len(df):,} total pairs")

    subset_label = "all genes"
    if args.gene_subset is not None:
        subset_path = Path(args.gene_subset)
        print(f"Loading gene subset from {subset_path} ...")
        subset_df = pd.read_csv(
            subset_path, sep="\t", header=None, usecols=[0],
            names=["gene"], dtype=str,
        )
        subset_genes = set(subset_df["gene"].dropna().str.strip())
        print(f"  {len(subset_genes):,} genes in subset file")
        before = len(df)
        df = df[df[args.gene_col].isin(subset_genes)].reset_index(drop=True)
        overlap = df[args.gene_col].nunique()
        print(f"  Overlap with Gene column: {overlap:,} genes")
        print(f"  Filtered pairs: {len(df):,} (from {before:,})")
        subset_label = f"Gene \u2208 subset ({overlap} genes)"

    labels = df[args.label_col].to_numpy()
    scores = df[args.score_col].to_numpy(dtype=np.float64)
    N = labels.shape[0]
    P = int(labels.sum())
    if N == 0 or P == 0:
        raise SystemExit(f"No pairs (N={N}) or no positives (P={P}) after filtering.")
    print(f"  N = {N:,} pairs, P = {P:,} true SL pairs ({100*P/N:.4f}%)")

    print("Computing guided curve (with analytical tie-breaking) ...")
    guided = guided_curve_with_ties(scores, labels)

    print("Computing oracle curve ...")
    oracle = oracle_curve(labels)

    print("Computing random baseline (analytical hypergeometric) ...")
    rand_mean, rand_sd = random_curve_analytical(N, P)
    rand_lo = rand_mean - rand_sd
    rand_hi = rand_mean + rand_sd

    # Normalized AUC for reporting (x in [0,1], y in [0,1] divided by P)
    x_norm = np.arange(N + 1) / N
    def _nauc(curve):
        return float(np.trapezoid(curve / max(P, 1), x_norm))
    print("\nNormalized AUC (higher = better):")
    print(f"  Guided  : {_nauc(guided):.4f}")
    print(f"  Oracle  : {_nauc(oracle):.4f}")
    print(f"  Random  : {_nauc(rand_mean):.4f}")

    # Sub-sample for plotting (full N+1 ~ 2M points would bloat the PDF)
    idx = _subsample_indices(N + 1, args.max_points)
    ks = np.arange(N + 1)[idx]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot in percentage units on the x-axis
    ks_pct = ks * (100.0 / N)

    ax.fill_between(ks_pct, rand_lo[idx], rand_hi[idx],
                    alpha=0.15, color="grey", label="Random ± 1 SD")
    ax.plot(ks_pct, rand_mean[idx], color="grey", ls="--", lw=1.5, label="Random")
    ax.plot(ks_pct, guided[idx], color="#1f77b4", lw=2,
            label="Guided (rank by pred_prob)")
    ax.plot(ks_pct, oracle[idx], color="black", ls=":", lw=1.8,
            label="Oracle (true label)")

    ax.set_xlabel(f"Percentage of pairs screened (total N = {N:,})")
    ax.set_ylabel("Cumulative SL pairs found")
    ax.set_title(
        f"Pair-Level Prioritization ({subset_label}):\n"
        f"Model-Guided vs Random vs Oracle"
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, P * 1.02)
    # Format primary x ticks as percentages
    from matplotlib.ticker import PercentFormatter
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))

    # Secondary x-axis (top) showing the absolute pair count, with the
    # true total N marked at the right-hand end as a reference.
    sec = ax.secondary_xaxis(
        "top",
        functions=(lambda p: p * N / 100.0, lambda x: x * 100.0 / N),
    )
    sec.set_xlabel("Number of pairs screened")
    # Choose a few round ticks plus ensure the true N appears at the right end
    base_ticks = np.linspace(0, N, 6)
    ticks = np.unique(np.concatenate([base_ticks, [N]]))
    sec.set_xticks(ticks)
    sec.set_xticklabels([f"{int(t):,}" for t in ticks])

    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
