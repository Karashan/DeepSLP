"""Configuration dataclasses and defaults matching the notebook's optimal settings."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Optimal hyperparameters identified in the H2018 notebook (CV1/CV2/CV3 blocks).
@dataclass
class HyperParams:
    hidden_size1: int = 128
    hidden_size2: int = 64
    hidden_size3: int = 32
    batch_size: int = 64
    num_epochs: int = 50
    learning_rate: float = 1e-2
    patience: int = 10            # early stopping on best val AUPR
    decay_factor: float = 0.1     # ReduceLROnPlateau
    scheduler_patience: int = 5
    dropout: float = 0.3
    focal_alpha: float = 0.75
    focal_gamma: float = 1.5


# Per-CV default split ratios (pairs for CV1, query genes for CV2, genes for CV3).
CV_SPLIT_DEFAULTS: Dict[int, Dict[str, float]] = {
    1: {"test_ratio": 0.10, "val_ratio": 0.10},
    2: {"test_ratio": 0.10, "val_ratio": 0.10},
    3: {"test_ratio": 0.21, "val_ratio": 0.21},
}


def cell_line_to_label(cell_line: str) -> str:
    """Map 'K562'/'Jurkat' (or 'K'/'J') to the binary label column name."""
    cl = cell_line.strip().lower()
    if cl in ("k", "k562"):
        return "K_SL_n3"
    if cl in ("j", "jurkat"):
        return "J_SL_n3"
    raise ValueError(f"Unknown cell line: {cell_line!r}. Use 'K562' or 'Jurkat'.")


@dataclass
class ExperimentConfig:
    """One experiment = (cell line, feature source, CV scenario(s), seeds)."""

    # Data
    cell_line: str = "K562"                 # 'K562' | 'Jurkat'
    feature: str = "GIV"                    # 'GIV' | 'LLM'
    giv_dir: str = "/home/b-xiangzhang/DeepSLP/data/input/external/H2018/GIV_H2018/"
    giv_iter_num: int = 4
    llm_path: str = (
        "/home/b-xiangzhang/DeepSLP/data/input/external/H2018/"
        "Horlbeck2018_pairs_both_ncbi_descriptions_pair_embeddings.tsv"
    )
    gi_threshold: float = -3.0              # label = (GI score < threshold)
    query_col: str = "gene1"
    lib_col: str = "gene2"

    # Run
    cvs: List[int] = field(default_factory=lambda: [1, 2, 3])
    n_runs: int = 10
    seed_base: int = 1842                   # seed_i = i + seed_base for i in 1..n_runs
    topk: int = 100

    # Training
    hp: HyperParams = field(default_factory=HyperParams)
    cv_splits: Dict[int, Dict[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in CV_SPLIT_DEFAULTS.items()}
    )

    # Output
    output_root: str = "/home/b-xiangzhang/DeepSLP/outputs/h2018_nn_runs/"
    tag: Optional[str] = None               # optional subfolder under output_root
    save_models: bool = True
    plot_losses: bool = True

    @property
    def label_col(self) -> str:
        return cell_line_to_label(self.cell_line)

    def run_dir(self) -> str:
        """Root directory for this experiment: <output_root>/<feature>/<cell_line>/<tag>/."""
        import os
        parts = [self.output_root, self.feature, self.cell_line]
        if self.tag:
            parts.append(self.tag)
        return os.path.join(*parts)
