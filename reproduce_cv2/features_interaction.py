"""Interaction feature builder (Tier-1 experiment, Step 1).

The stored GIV features are the SYMMETRIC SUM of two per-gene centered AE
embeddings: ko_i = center(AE_ko[g1])_i + center(AE_ko[g2])_i (same for exp).
Because that is purely additive, the downstream MLP can only learn an additive
score a(g1)+a(g2)+const, which cannot represent a true gene-gene interaction and
drives the hub artifact / generalization ceiling.

This module rebuilds features directly from the per-gene AE_3L embeddings so we
can add genuinely non-additive, still order-invariant interaction terms:
    sum      = c(g1) + c(g2)        (the current additive feature)
    product  = c(g1) * c(g2)        (Hadamard; symmetric, non-additive)
    absdiff  = |c(g1) - c(g2)|      (symmetric, non-additive)
for both the ko (CRISPRGeneEffect) and exp (Expression) embeddings.

Feature layouts:
    mode='sum'         -> [ko_sum, exp_sum]                       (256 dims; == stored GIV)
    mode='interaction' -> [ko_sum, ko_prod, ko_absdiff,
                           exp_sum, exp_prod, exp_absdiff]        (768 dims)

Verified: mode='sum' reproduces the stored ko_*/exp_* features to ~1e-15.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

AE_DIR_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "input", "AE_3L",
)
KO_FILE = "AE_std100_CRISPRGeneEffect_24Q4_imputed_gene_wise_mean_align_qGI2021.txt"
EXP_FILE = "AE_std100_Expression_BC_24Q4_align_qGI2021.txt"


class GeneEmbeddings:
    """Per-gene centered ko and exp embeddings with fast name->row lookup."""

    def __init__(self, ko_mat, ko_idx, exp_mat, exp_idx):
        self.ko_mat = ko_mat            # (G_ko, 128) float32, row-centered
        self.ko_idx = ko_idx            # gene -> row
        self.exp_mat = exp_mat          # (G_exp, 128) float32, row-centered
        self.exp_idx = exp_idx

    @property
    def dim(self) -> int:
        return self.ko_mat.shape[1]


def _row_center(mat: np.ndarray) -> np.ndarray:
    return (mat - mat.mean(axis=1, keepdims=True)).astype(np.float32)


def load_centered_embeddings(ae_dir: str = AE_DIR_DEFAULT) -> GeneEmbeddings:
    ko = pd.read_csv(os.path.join(ae_dir, KO_FILE), sep="\t", index_col=0)
    ex = pd.read_csv(os.path.join(ae_dir, EXP_FILE), sep="\t", index_col=0)
    ko_mat = _row_center(ko.values)
    ex_mat = _row_center(ex.values)
    ko_idx = {g: i for i, g in enumerate(ko.index)}
    ex_idx = {g: i for i, g in enumerate(ex.index)}
    return GeneEmbeddings(ko_mat, ko_idx, ex_mat, ex_idx)


def _map_rows(names, idx: Dict[str, int]) -> np.ndarray:
    return np.fromiter((idx.get(n, -1) for n in names), dtype=np.int64, count=len(names))


def build_pair_features(
    df: pd.DataFrame,
    emb: GeneEmbeddings,
    mode: str = "interaction",
    query_col: str = "Query",
    lib_col: str = "Gene",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, valid_mask).

    X has one row per *valid* pair (both genes present in both ko and exp
    embeddings); valid_mask is a boolean array over df rows so callers can align
    labels / metadata: df_valid = df[valid_mask].
    """
    q = df[query_col].values
    g = df[lib_col].values
    qi_ko = _map_rows(q, emb.ko_idx); gi_ko = _map_rows(g, emb.ko_idx)
    qi_ex = _map_rows(q, emb.exp_idx); gi_ex = _map_rows(g, emb.exp_idx)
    valid = (qi_ko >= 0) & (gi_ko >= 0) & (qi_ex >= 0) & (gi_ex >= 0)

    a_ko = emb.ko_mat[qi_ko[valid]]; b_ko = emb.ko_mat[gi_ko[valid]]
    a_ex = emb.exp_mat[qi_ex[valid]]; b_ex = emb.exp_mat[gi_ex[valid]]

    if mode == "sum":
        parts = [a_ko + b_ko, a_ex + b_ex]
    elif mode == "interaction":
        parts = [
            a_ko + b_ko, a_ko * b_ko, np.abs(a_ko - b_ko),
            a_ex + b_ex, a_ex * b_ex, np.abs(a_ex - b_ex),
        ]
    elif mode == "interaction_only":
        # drop the additive sum entirely; only non-additive terms
        parts = [a_ko * b_ko, np.abs(a_ko - b_ko), a_ex * b_ex, np.abs(a_ex - b_ex)]
    else:
        raise ValueError(f"unknown mode: {mode}")

    X = np.concatenate(parts, axis=1).astype(np.float32)
    return X, valid


def feature_names(mode: str, dim: int = 128):
    base = {
        "sum": ["ko_sum", "exp_sum"],
        "interaction": ["ko_sum", "ko_prod", "ko_absdiff", "exp_sum", "exp_prod", "exp_absdiff"],
        "interaction_only": ["ko_prod", "ko_absdiff", "exp_prod", "exp_absdiff"],
    }[mode]
    return [f"{b}_{i}" for b in base for i in range(dim)]
