"""CLI: run one experiment (cell line × feature source × CVs × N seeds).

Example:
    python -m h2018_nn.scripts.run_experiment \
        --cell-line K562 --feature GIV --cvs 1 2 3 --n-runs 10 \
        --output-root /home/b-xiangzhang/DeepSLP/outputs/h2018_nn_runs/
"""
from __future__ import annotations

import argparse
import sys

from ..config import ExperimentConfig, HyperParams
from ..pipeline import run_experiment


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run DeepSLP NN on H2018 data.")
    p.add_argument("--cell-line", choices=["K562", "Jurkat"], default="K562")
    p.add_argument("--feature",   choices=["GIV", "LLM"], default="GIV")
    p.add_argument("--cvs", type=int, nargs="+", default=[1, 2, 3],
                   choices=[1, 2, 3])
    p.add_argument("--n-runs", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=1842)
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--tag", type=str, default=None,
                   help="Optional extra subfolder under <output-root>/<feature>/<cell-line>/")
    p.add_argument("--output-root", type=str,
                   default="/home/b-xiangzhang/DeepSLP/outputs/h2018_nn_runs/")
    p.add_argument("--giv-dir", type=str,
                   default="/home/b-xiangzhang/DeepSLP/data/input/external/H2018/GIV_H2018/")
    p.add_argument("--giv-iter-num", type=int, default=4)
    p.add_argument("--llm-path", type=str,
                   default=("/home/b-xiangzhang/DeepSLP/data/input/external/H2018/"
                            "Horlbeck2018_pairs_both_ncbi_descriptions_pair_embeddings.tsv"))
    p.add_argument("--gi-threshold", type=float, default=-3.0)

    # Hyperparameters
    p.add_argument("--hidden1", type=int, default=128)
    p.add_argument("--hidden2", type=int, default=64)
    p.add_argument("--hidden3", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--decay-factor", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--focal-alpha", type=float, default=0.75)
    p.add_argument("--focal-gamma", type=float, default=1.5)

    p.add_argument("--no-save-models", action="store_true")
    p.add_argument("--no-plot-losses", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    hp = HyperParams(
        hidden_size1=args.hidden1, hidden_size2=args.hidden2, hidden_size3=args.hidden3,
        batch_size=args.batch_size, num_epochs=args.epochs,
        learning_rate=args.lr, patience=args.patience, decay_factor=args.decay_factor,
        dropout=args.dropout, focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
    )
    cfg = ExperimentConfig(
        cell_line=args.cell_line, feature=args.feature,
        giv_dir=args.giv_dir, giv_iter_num=args.giv_iter_num, llm_path=args.llm_path,
        gi_threshold=args.gi_threshold,
        cvs=list(args.cvs), n_runs=args.n_runs, seed_base=args.seed_base, topk=args.topk,
        hp=hp, output_root=args.output_root, tag=args.tag,
        save_models=not args.no_save_models, plot_losses=not args.no_plot_losses,
    )
    run_experiment(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
