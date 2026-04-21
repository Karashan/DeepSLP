"""Data loading: H2018 labels + GIV features and LLM pair embeddings."""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd

# Columns present in the raw GIV files that are not features
_H2018_META_COLS = [
    "gene1",
    "gene2",
    "K562 Replicate Average GI score",
    "Jurkat Replicate Average GI score",
]
# Derived binary label columns we add
_H2018_LABEL_COLS = ["K_SL_n3", "J_SL_n3"]


def _score_col_for(cell_line: str) -> str:
    cl = cell_line.strip().lower()
    if cl in ("k", "k562"):
        return "K562 Replicate Average GI score"
    if cl in ("j", "jurkat"):
        return "Jurkat Replicate Average GI score"
    raise ValueError(f"Unknown cell line: {cell_line!r}")


def _concat_giv_files(giv_dir: str, iter_num: int = 4,
                      tail: str = ".tsv", sep: str = "\t") -> pd.DataFrame:
    """Concatenate the split GIV files (`*_0.tsv`..`*_{N-1}.tsv`) in order."""
    dfs = []
    files = os.listdir(giv_dir)
    for i in range(iter_num):
        suffix = f"_{i}{tail}"
        for f in files:
            if f.endswith(suffix):
                dfs.append(pd.read_csv(os.path.join(giv_dir, f), sep=sep, index_col=None))
                break
        else:
            raise FileNotFoundError(
                f"No file ending in '{suffix}' found in {giv_dir}")
    return pd.concat(dfs, ignore_index=True)


def add_sl_labels(df: pd.DataFrame, gi_threshold: float = -3.0) -> pd.DataFrame:
    """Add binary SL labels `K_SL_n3`/`J_SL_n3` (GI score < threshold)."""
    df = df.copy()
    df["K_SL_n3"] = (df["K562 Replicate Average GI score"]   < gi_threshold).astype(int)
    df["J_SL_n3"] = (df["Jurkat Replicate Average GI score"] < gi_threshold).astype(int)
    return df


def load_giv_features(giv_dir: str, iter_num: int = 4,
                      cell_line: str = "K562",
                      gi_threshold: float = -3.0,
                      query_col: str = "gene1",
                      lib_col: str = "gene2",
                      ) -> Tuple[pd.DataFrame, List[str]]:
    """Load the GIV features, derive SL labels, drop pairs with NA cell-line score
    and self-pairs (gene1 == gene2).

    Returns (dataframe, non_feature_cols).
    """
    df = _concat_giv_files(giv_dir, iter_num=iter_num)
    df = add_sl_labels(df, gi_threshold=gi_threshold)
    df = df[df[_score_col_for(cell_line)].notnull()].copy()
    df = df[df[query_col] != df[lib_col]].copy()
    non_feature_cols = list(_H2018_META_COLS) + list(_H2018_LABEL_COLS)
    return df.reset_index(drop=True), non_feature_cols


def _load_llm_embeddings(path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load the pair-embedding TSV (3 columns: gene1, gene2, comma-sep floats)."""
    raw = pd.read_csv(path, sep="\t", header=None, names=["gene1", "gene2", "embedding"])
    mat = np.stack([np.fromstring(s, sep=",", dtype=np.float32)
                    for s in raw["embedding"].values])
    emb_dim = mat.shape[1]
    emb_cols = [f"emb_{i}" for i in range(emb_dim)]
    df = pd.concat(
        [raw[["gene1", "gene2"]].reset_index(drop=True),
         pd.DataFrame(mat, columns=emb_cols)],
        axis=1,
    )
    df = df.drop_duplicates(subset=["gene1", "gene2"], keep="first").reset_index(drop=True)
    return df, emb_cols


def load_llm_embedding_features(llm_path: str,
                                giv_dir: str,
                                iter_num: int = 4,
                                cell_line: str = "K562",
                                gi_threshold: float = -3.0,
                                query_col: str = "gene1",
                                lib_col: str = "gene2",
                                ) -> Tuple[pd.DataFrame, List[str]]:
    """Load the LLM pair embeddings and inner-join with H2018 pairs (+ labels).

    Uses GIV files only to borrow labels/metadata — the feature columns are purely
    the embedding dimensions.
    """
    giv_df = _concat_giv_files(giv_dir, iter_num=iter_num)
    giv_df = add_sl_labels(giv_df, gi_threshold=gi_threshold)
    score_col = _score_col_for(cell_line)
    giv_df = giv_df[giv_df[score_col].notnull()].copy()

    emb_df, _ = _load_llm_embeddings(llm_path)
    meta_label_cols = list(_H2018_META_COLS) + list(_H2018_LABEL_COLS)
    merged = giv_df[meta_label_cols].merge(emb_df, on=["gene1", "gene2"], how="inner")
    merged = merged[merged[query_col] != merged[lib_col]].reset_index(drop=True)
    return merged, meta_label_cols


def load_features(feature: str, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
    """Dispatch by feature name ('GIV' or 'LLM')."""
    feat = feature.upper()
    if feat == "GIV":
        return load_giv_features(
            giv_dir=kwargs["giv_dir"], iter_num=kwargs.get("iter_num", 4),
            cell_line=kwargs["cell_line"],
            gi_threshold=kwargs.get("gi_threshold", -3.0),
            query_col=kwargs.get("query_col", "gene1"),
            lib_col=kwargs.get("lib_col", "gene2"),
        )
    if feat == "LLM":
        return load_llm_embedding_features(
            llm_path=kwargs["llm_path"],
            giv_dir=kwargs["giv_dir"], iter_num=kwargs.get("iter_num", 4),
            cell_line=kwargs["cell_line"],
            gi_threshold=kwargs.get("gi_threshold", -3.0),
            query_col=kwargs.get("query_col", "gene1"),
            lib_col=kwargs.get("lib_col", "gene2"),
        )
    raise ValueError(f"Unknown feature source: {feature!r}. Use 'GIV' or 'LLM'.")
