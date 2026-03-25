"""Data loading, cleaning, cross-validation splitting, and preprocessing."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_tsv_iterations(directory: str | Path, n_iters: int = 20) -> pd.DataFrame:
    """Concatenate numbered TSV files (``*_0.tsv`` … ``*_{n-1}.tsv``)."""
    directory = Path(directory)
    frames: list[pd.DataFrame] = []
    for i in range(n_iters):
        suffix = f"_{i}.tsv"
        for f in sorted(directory.iterdir()):
            if f.name.endswith(suffix):
                frames.append(pd.read_csv(f, sep="\t"))
    if not frames:
        raise FileNotFoundError(f"No matching TSV files found in {directory}")
    return pd.concat(frames, ignore_index=True)


def filter_unique_pairs(
    df: pd.DataFrame,
    gene_col: str = "Gene",
    query_col: str = "Query",
    fdr_col: str = "FDR",
) -> pd.DataFrame:
    """Keep one row per unordered gene pair, choosing the lowest FDR."""
    df = df.dropna(subset=[gene_col, query_col, fdr_col])
    # Remove self-interactions
    df = df[df[gene_col] != df[query_col]]
    # Canonical pair key
    pair = df[[gene_col, query_col]].apply(lambda r: tuple(sorted(r)), axis=1)
    df = df.assign(_pair=pair)
    keep_idx = df.groupby("_pair")[fdr_col].idxmin()
    return df.loc[keep_idx].drop(columns="_pair").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cross-validation splitting
# ---------------------------------------------------------------------------

NON_FEATURE_COLS = [
    "Gene", "Query", "qGI_score", "FDR",
    "GI_standard", "GI_stringent",
    "GI_standard_Type1", "GI_standard_Type2", "GI_standard_Type3",
    "GI_stringent_Type1", "GI_stringent_Type2", "GI_stringent_Type3",
]


def _extract_features_labels(
    df: pd.DataFrame,
    label_col: str,
    non_feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y): features and binary labels, dropping NaN rows."""
    if non_feature_cols is None:
        non_feature_cols = NON_FEATURE_COLS
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    X = df[feature_cols].dropna(how="any")
    y = df.loc[X.index, label_col]
    return X, y


def split_random_stratified(
    df: pd.DataFrame,
    label_col: str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    non_feature_cols: list[str] | None = None,
) -> dict:
    """CV strategy 1: random stratified train/val/test split."""
    X, y = _extract_features_labels(df, label_col, non_feature_cols)
    hold = val_ratio + test_ratio
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=hold, stratify=y, random_state=seed,
    )
    val_frac = val_ratio / hold
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=1 - val_frac, stratify=y_tmp, random_state=seed,
    )
    return dict(X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test)


def split_query_holdout(
    df: pd.DataFrame,
    label_col: str,
    query_col: str = "Query",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    non_feature_cols: list[str] | None = None,
) -> dict:
    """CV strategy 2: hold out entire query genes for val/test."""
    X, y = _extract_features_labels(df, label_col, non_feature_cols)
    queries = df.loc[X.index, query_col].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(queries)
    n_test = max(1, int(len(queries) * test_ratio))
    n_val = max(1, int(len(queries) * val_ratio))
    test_q = set(queries[:n_test])
    val_q = set(queries[n_test:n_test + n_val])
    train_q = set(queries[n_test + n_val:])

    q = df.loc[X.index, query_col]
    train_idx = X.index[q.isin(train_q)]
    val_idx = X.index[q.isin(val_q)]
    test_idx = X.index[q.isin(test_q)]
    return dict(X_train=X.loc[train_idx], y_train=y.loc[train_idx],
                X_val=X.loc[val_idx], y_val=y.loc[val_idx],
                X_test=X.loc[test_idx], y_test=y.loc[test_idx])


def split_query_kfold(
    df: pd.DataFrame,
    label_col: str,
    query_col: str = "Query",
    n_folds: int = 10,
    fold: int = 1,
    seed: int = 42,
    non_feature_cols: list[str] | None = None,
) -> dict:
    """CV strategy 4: k-fold split on query genes.

    ``fold`` is 1-indexed.  The fold is used as test, the previous fold as
    validation, and all remaining folds as training.
    """
    X, y = _extract_features_labels(df, label_col, non_feature_cols)
    queries = np.array(sorted(df.loc[X.index, query_col].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(queries)
    folds = np.array_split(queries, n_folds)

    test_i = fold - 1
    val_i = (fold - 2) % n_folds
    test_q = set(folds[test_i])
    val_q = set(folds[val_i])
    train_q = set(np.concatenate([folds[i] for i in range(n_folds) if i not in (test_i, val_i)]))

    q = df.loc[X.index, query_col]
    train_idx = X.index[q.isin(train_q)]
    val_idx = X.index[q.isin(val_q)]
    test_idx = X.index[q.isin(test_q)]
    return dict(X_train=X.loc[train_idx], y_train=y.loc[train_idx],
                X_val=X.loc[val_idx], y_val=y.loc[val_idx],
                X_test=X.loc[test_idx], y_test=y.loc[test_idx])


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def scale_splits(
    split: dict,
    scaler_path: str | Path | None = None,
) -> tuple[dict, StandardScaler]:
    """Fit StandardScaler on training data; transform all splits in-place.

    Constant features (zero variance in training set) are dropped before
    scaling to avoid NaN from division by zero.

    Returns the modified *split* dict (values become numpy arrays) and the
    fitted scaler.
    """
    import pandas as pd

    X_train = split["X_train"]
    if isinstance(X_train, pd.DataFrame):
        # Identify and drop zero-variance columns
        stds = X_train.std()
        keep = stds[stds > 0].index
        dropped = len(stds) - len(keep)
        if dropped > 0:
            print(f"  [scale_splits] Dropping {dropped} constant feature(s)")
        split["X_train"] = X_train[keep]
        split["X_val"] = split["X_val"][keep]
        split["X_test"] = split["X_test"][keep]
    else:
        # numpy array path
        stds = np.std(X_train, axis=0)
        keep_mask = stds > 0
        dropped = int((~keep_mask).sum())
        if dropped > 0:
            print(f"  [scale_splits] Dropping {dropped} constant feature(s)")
            split["X_train"] = X_train[:, keep_mask]
            split["X_val"] = split["X_val"][:, keep_mask]
            split["X_test"] = split["X_test"][:, keep_mask]

    scaler = StandardScaler()
    split["X_train"] = scaler.fit_transform(split["X_train"])
    split["X_val"] = scaler.transform(split["X_val"])
    split["X_test"] = scaler.transform(split["X_test"])
    if scaler_path is not None:
        import joblib
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
    return split, scaler
