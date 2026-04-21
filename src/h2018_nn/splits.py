"""Cross-validation splits: CV1 random pairs / CV2 query holdout / CV3 double holdout."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_val_test_cv(
    df: pd.DataFrame,
    label_col: str,
    non_feature_cols: List[str],
    query_col: str = "gene1",
    lib_col: str = "gene2",
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    rand_seed: int = 42,
    cv: int = 1,
) -> Dict[str, pd.DataFrame]:
    """Return a dict with X_train/X_val/X_test and y_train/y_val/y_test.

    cv=1: stratified random pair split.
    cv=2: hold out unique query genes (`query_col`).
    cv=3: double holdout — gene sets assigned to train/val/test; test pairs have
          both genes from the held-out set.
    """
    features = df.drop(columns=non_feature_cols)
    features.dropna(axis=0, how="any", inplace=True)

    if cv == 1:
        df_sub = df.loc[features.index]
        train_df, temp_df = train_test_split(
            df_sub, test_size=(test_ratio + val_ratio),
            stratify=df_sub[label_col], random_state=rand_seed)
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df, test_size=(1 - val_size),
            stratify=temp_df[label_col], random_state=rand_seed)
        X_train = train_df.drop(columns=non_feature_cols); y_train = train_df[label_col]
        X_val   = val_df.drop(columns=non_feature_cols);   y_val   = val_df[label_col]
        X_test  = test_df.drop(columns=non_feature_cols);  y_test  = test_df[label_col]

    elif cv == 2:
        df_sub = df.loc[features.index, [query_col]]
        labels = df.loc[features.index, label_col]
        query_genes = df_sub[query_col].unique()
        test_len = int(len(query_genes) * test_ratio)
        val_len  = int(len(query_genes) * val_ratio)
        rng = np.random.default_rng(rand_seed)
        # np.random.seed + np.random.choice (no-replace) to match the notebook exactly
        np.random.seed(rand_seed)
        test_query  = np.random.choice(query_genes, size=test_len, replace=False)
        train_query = list(set(query_genes) - set(test_query))
        val_query   = np.random.choice(train_query, size=val_len, replace=False)
        train_query = list(set(train_query) - set(val_query))

        train_idx = df_sub[df_sub[query_col].isin(train_query)].index
        val_idx   = df_sub[df_sub[query_col].isin(val_query)].index
        test_idx  = df_sub[df_sub[query_col].isin(test_query)].index
        X_train = features.loc[train_idx]; y_train = labels.loc[train_idx]
        X_val   = features.loc[val_idx];   y_val   = labels.loc[val_idx]
        X_test  = features.loc[test_idx];  y_test  = labels.loc[test_idx]

    elif cv == 3:
        df_sub = df.loc[features.index, [query_col, lib_col, label_col]]
        labels = df_sub[label_col]
        potential = (df.loc[features.index]
                       .groupby(query_col).agg({label_col: "sum"}).reset_index())
        potential.columns = ["gene", "num_SL_partner"]
        training_percent = 1 - test_ratio - val_ratio
        val_percent = val_ratio
        num_samples = potential.shape[0]
        all_idx = list(range(num_samples))
        np.random.seed(rand_seed)
        np.random.shuffle(all_idx)

        tr = potential.iloc[all_idx[:int(num_samples * training_percent)]]
        va = potential.iloc[all_idx[int(num_samples * training_percent):int(num_samples * (training_percent + val_percent))]]
        te = potential.iloc[all_idx[int(num_samples * (training_percent + val_percent)):]]
        training_genes = tr["gene"].values
        val_genes      = va["gene"].values
        test_genes     = te["gene"].values

        pair_train = df_sub[(df_sub[query_col].isin(training_genes)) & (df_sub[lib_col].isin(training_genes))]
        pair_val   = df_sub[(df_sub[query_col].isin(val_genes))      & (df_sub[lib_col].isin(val_genes))]
        pair_test  = df_sub[(df_sub[query_col].isin(test_genes))     & (df_sub[lib_col].isin(test_genes))]

        X_train = features.loc[pair_train.index]; y_train = labels.loc[pair_train.index]
        X_val   = features.loc[pair_val.index];   y_val   = labels.loc[pair_val.index]
        X_test  = features.loc[pair_test.index];  y_test  = labels.loc[pair_test.index]

    else:
        raise ValueError(f"cv must be 1, 2, or 3 — got {cv!r}")

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }
