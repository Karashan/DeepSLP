"""Data loading, deduplication and CV2 query-holdout splitting.

Independent re-implementation of the data preparation used by the original
notebook (notebooks/cv2_tuning.ipynb), kept self-contained so it has no
dependency on the project's ``src/`` package.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd


def concat_feature_shards(
    input_dir: str, iter_num: int = 20, tail: str = ".tsv", sep: str = "\t"
) -> pd.DataFrame:
    """Concatenate ``*_{i}{tail}`` shards in numeric order (0 .. iter_num-1)."""
    files = os.listdir(input_dir)
    frames: List[pd.DataFrame] = []
    for i in range(iter_num):
        wanted_tail = f"_{i}{tail}"
        for fname in files:
            if fname.endswith(wanted_tail):
                frames.append(pd.read_csv(os.path.join(input_dir, fname), sep=sep, index_col=None))
    if not frames:
        raise FileNotFoundError(
            f"No files matching '*_<i>{tail}' for i in [0,{iter_num}) found in {input_dir}"
        )
    df = pd.concat(frames).reset_index(drop=True)
    return df


def filter_unique_pairs_by_lowest_fdr(
    df: pd.DataFrame, col1: str = "Gene", col2: str = "Query", col_fdr: str = "FDR"
) -> pd.DataFrame:
    """Collapse A-B / B-A duplicate gene pairs, keeping the lower-FDR row.

    Also drops rows with missing values in the key columns and removes
    self-interactions (col1 == col2).
    """
    df = df.dropna(subset=[col1, col2, col_fdr])
    df = df.drop(df[df[col1] == df[col2]].index, axis=0)

    normalized_pairs = df[[col1, col2]].apply(lambda row: tuple(sorted(row)), axis=1)
    normalized_df = pd.DataFrame({"normalized_pair": normalized_pairs, col_fdr: df[col_fdr]})
    min_idx = normalized_df.groupby("normalized_pair")[col_fdr].idxmin()
    return df.loc[min_idx].reset_index(drop=True)


def load_input(
    input_dir: str,
    iter_num: int = 20,
    tail: str = ".tsv",
    query_col: str = "Query",
    lib_col: str = "Gene",
    fdr_col: str = "FDR",
) -> pd.DataFrame:
    """Full input preparation: concat shards, dedup pairs, drop self-GIs."""
    df = concat_feature_shards(input_dir, iter_num=iter_num, tail=tail)
    df = filter_unique_pairs_by_lowest_fdr(df, col1=lib_col, col2=query_col, col_fdr=fdr_col)
    # Defensive: remove any remaining self-interactions
    df = df[df[query_col] != df[lib_col]].reset_index(drop=True)
    return df


def feature_columns(df: pd.DataFrame, non_feature_cols: List[str]) -> List[str]:
    return [c for c in df.columns if c not in non_feature_cols]


def split_cv2_query_holdout(
    df: pd.DataFrame,
    label_col: str,
    non_feature_cols: List[str],
    query_col: str = "Query",
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    rand_seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """CV2 split: hold out whole query genes for validation and test.

    Reproduces the ``cv == 2`` branch of the original notebook, including the
    exact numpy RNG calls so that a given ``rand_seed`` yields the same split.
    """
    features = df.drop(columns=non_feature_cols)
    features = features.dropna(axis=0, how="any")

    df_sub = df.loc[features.index, [query_col]]
    labels = df.loc[features.index, label_col]

    query_genes = df_sub[query_col].unique()
    test_len = int(len(query_genes) * test_ratio)
    val_len = int(len(query_genes) * val_ratio)

    np.random.seed(rand_seed)
    test_query = np.random.choice(query_genes, size=test_len, replace=False, p=None)
    train_query = list(set(query_genes) - set(test_query))
    val_query = np.random.choice(train_query, size=val_len, replace=False, p=None)
    train_query = list(set(train_query) - set(val_query))

    train_idx = df_sub[df_sub[query_col].isin(train_query)].index
    val_idx = df_sub[df_sub[query_col].isin(val_query)].index
    test_idx = df_sub[df_sub[query_col].isin(test_query)].index

    return {
        "X_train": features.loc[train_idx],
        "X_val": features.loc[val_idx],
        "X_test": features.loc[test_idx],
        "y_train": labels.loc[train_idx],
        "y_val": labels.loc[val_idx],
        "y_test": labels.loc[test_idx],
    }
