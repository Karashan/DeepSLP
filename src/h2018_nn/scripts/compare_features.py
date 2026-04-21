"""CLI: compare feature sources (e.g. GIV vs LLM) across CV scenarios.

Example:
    python -m h2018_nn.scripts.compare_features \
        --cell-line K562 \
        --feature GIV=/home/b-xiangzhang/DeepSLP/outputs/H2018_reproduce_04172026 \
        --feature LLM=/home/b-xiangzhang/DeepSLP/outputs/H2018_embeddings_04172026 \
        --metric AUPR --all-metrics \
        --output-dir /home/b-xiangzhang/DeepSLP/outputs/H2018_GIV_vs_LLM_comparison/
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

from ..compare import load_perf_stats, plot_all_metrics_grid, plot_metric_bar


def _parse_features(items: List[str]) -> Dict[str, str]:
    out = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"--feature value '{item}' must be NAME=ROOT (e.g. GIV=/path/to/root)")
        name, root = item.split("=", 1)
        out[name.strip()] = os.path.expanduser(root.strip())
    return out


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare feature sources across CVs.")
    p.add_argument("--feature", action="append", required=True,
                   help="NAME=ROOT pair. Repeat for each source. "
                        "ROOT is the parent containing <cell_line>/CV*/ or "
                        "<cell_line>/repeats/CV*/ stats files.")
    p.add_argument("--cell-line", default="K562")
    p.add_argument("--cvs", nargs="+", default=["CV1", "CV2", "CV3"])
    p.add_argument("--metric", default="AUPR",
                   help="Metric to plot (e.g. AUROC, AUPR, AP, Recall@100).")
    p.add_argument("--all-metrics", action="store_true",
                   help="Also render a grid with every available metric.")
    p.add_argument("--output-dir", required=True,
                   help="Directory where bar plots and the merged long table will be written.")
    p.add_argument("--no-points", action="store_true",
                   help="Do not overlay individual per-run points.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    feature_roots = _parse_features(args.feature)
    os.makedirs(args.output_dir, exist_ok=True)

    df_long = load_perf_stats(
        feature_roots=feature_roots,
        cell_line=args.cell_line,
        cv_list=args.cvs,
    )
    long_path = os.path.join(args.output_dir, f"{args.cell_line}_perf_long.tsv")
    df_long.to_csv(long_path, sep="\t", index=False)
    print("Wrote:", long_path)

    feature_order = tuple(feature_roots.keys())
    single_path = os.path.join(args.output_dir, f"{args.cell_line}_{args.metric}_bar.pdf")
    plot_metric_bar(df_long, metric=args.metric, cv_list=args.cvs,
                    feature_order=feature_order,
                    show_points=not args.no_points,
                    save_path=single_path)

    if args.all_metrics:
        grid_path = os.path.join(args.output_dir, f"{args.cell_line}_all_metrics_grid.pdf")
        plot_all_metrics_grid(df_long, cv_list=args.cvs, feature_order=feature_order,
                              show_points=not args.no_points, save_path=grid_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
