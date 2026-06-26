"""Default configuration recovered from the saved CV2 models.

All values here reproduce the run that produced:
    data/interim/ReLU128_f_a075_g15_10folds_pt10/
        CV2_811_GIV_NN_LR1e2_50e_p10_d01_{1..10}.pth
        CV2_811_seed{1..10}.joblib

NOTE: the filename token ``d01`` is misleading; the trained dropout is 0.3,
not 0.1 (dropout was hard-coded in the original NeuralNetwork and never
parameterised). The architecture loaded from every checkpoint confirms p=0.3.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


# Repository root = parent of this package directory. Used to anchor default
# data paths so the scripts work regardless of the current working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _repo_path(*parts: str) -> str:
    return os.path.join(REPO_ROOT, *parts)


# Columns in the input feature tables that are NOT model features.
NON_FEATURE_COLS: List[str] = [
    "Gene",
    "Query",
    "qGI_score",
    "FDR",
    "GI_standard",
    "GI_stringent",
    "GI_standard_Type1",
    "GI_standard_Type2",
    "GI_standard_Type3",
    "GI_stringent_Type1",
    "GI_stringent_Type2",
    "GI_stringent_Type3",
]


@dataclass
class TrainConfig:
    # --- data ---
    input_dir: str = field(default_factory=lambda: _repo_path("data", "input", "GIV_24Q4_3L") + os.sep)
    input_glob_iter: int = 20            # number of *_{i}.tsv shards to concatenate
    file_tail: str = ".tsv"
    label_col: str = "GI_stringent_Type2"
    query_col: str = "Query"
    lib_col: str = "Gene"
    fdr_col: str = "FDR"
    non_feature_cols: List[str] = field(default_factory=lambda: list(NON_FEATURE_COLS))

    # --- cross-validation (CV2: hold out query genes only) ---
    n_folds: int = 10
    test_ratio: float = 0.1
    val_ratio: float = 0.1
    seed_offset: int = 1842              # rand_seed = fold_index + seed_offset

    # --- architecture ---
    hidden_size1: int = 128
    hidden_size2: int = 64
    hidden_size3: int = 32
    output_size: int = 1
    dropout_p: float = 0.3

    # --- optimisation ---
    num_epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-2
    focal_alpha: float = 0.75
    focal_gamma: float = 1.5

    # --- scheduler / early stopping ---
    scheduler_factor: float = 0.1        # ReduceLROnPlateau factor
    scheduler_patience: int = 5          # ReduceLROnPlateau patience (hard-coded in original)
    early_stop_patience: int = 10        # monitors best validation AUPR

    # --- output ---
    output_dir: str = field(default_factory=lambda: _repo_path("reproduce_cv2", "output") + os.sep)
    prefix: str = "CV2_811_GIV_NN_LR1e2_50e_p10_d01"
    topk: int = 100                      # K for Precision@K / Recall@K
