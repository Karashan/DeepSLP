"""High-level pipeline: single-run and multi-seed repeats across CV scenarios."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import ExperimentConfig, HyperParams
from .data import load_features
from .metrics import ensure_dir, perf_columns
from .splits import split_train_val_test_cv
from .train import train_validate_test_nn


def optim_nn_pipeline(
    df_score: pd.DataFrame,
    label_col: str,
    non_feature_cols: List[str],
    cv: int,
    rand_seed: int,
    output_dir: str,
    plt_name: str,
    table_name: str,
    model_name: str = "model.pth",
    scaler_name: str = "scaler.joblib",
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    query_col: str = "gene1",
    lib_col: str = "gene2",
    hp: Optional[HyperParams] = None,
    save_model: bool = True,
    plot_losses: bool = True,
    topk: int = 100,
):
    """Split one CV scenario, then call `train_validate_test_nn`."""
    ensure_dir(output_dir)
    hp = hp or HyperParams()
    print(f"Data split: CV{cv} | seed={rand_seed}")

    df = df_score[df_score[query_col] != df_score[lib_col]].copy()
    splits = split_train_val_test_cv(
        df, label_col=label_col, query_col=query_col, lib_col=lib_col,
        non_feature_cols=non_feature_cols,
        test_ratio=test_ratio, val_ratio=val_ratio,
        rand_seed=rand_seed, cv=cv,
    )
    return train_validate_test_nn(
        splits=splits, df=df, non_feature_cols=non_feature_cols,
        output_dir=output_dir, plt_name=plt_name, table_name=table_name,
        hp=hp, model_name=model_name, scaler_name=scaler_name,
        save_model=save_model, plot_losses=plot_losses, topk=topk,
    )


def run_cv_repeats(
    cv: int,
    df_score: pd.DataFrame,
    label_col: str,
    non_feature_cols: List[str],
    output_root: str,
    cv_split: Dict[int, Dict[str, float]],
    hp: HyperParams,
    n_runs: int = 10,
    seed_base: int = 1842,
    topk: int = 100,
    query_col: str = "gene1",
    lib_col: str = "gene2",
    prefix_tag: Optional[str] = None,
    save_model: bool = True,
    plot_losses: bool = True,
) -> pd.DataFrame:
    """Run `n_runs` repeats of one CV scenario. Writes per-run artifacts +
    a `*_performance_stats_{n_runs}runs.tsv` under `{output_root}/CV{cv}/`.
    Returns the per-run performance DataFrame.
    """
    ratios = cv_split[cv]
    tag = prefix_tag or f"CV{cv}"
    cv_dir = os.path.join(output_root, f"CV{cv}/")
    ensure_dir(cv_dir)

    prefix = f"{tag}_NN_h{hp.hidden_size1}_{hp.hidden_size2}_{hp.hidden_size3}" \
             f"_lr{hp.learning_rate:g}_{hp.num_epochs}e_p{hp.patience}_d{hp.decay_factor:g}_focal"

    rows = []
    for i in range(1, n_runs + 1):
        seed = i + seed_base
        r = str(i)
        print(f"\n===== {tag} run {i}/{n_runs} (seed={seed}) =====")
        _, _, perf = optim_nn_pipeline(
            df_score=df_score, label_col=label_col, non_feature_cols=non_feature_cols,
            cv=cv, rand_seed=seed,
            test_ratio=ratios["test_ratio"], val_ratio=ratios["val_ratio"],
            output_dir=cv_dir,
            plt_name=prefix + f"_ROC_PR_curves_{r}.pdf",
            table_name=prefix + f"_preds_{r}.tsv",
            model_name=prefix + f"_{r}.pth",
            scaler_name=f"CV{cv}_seed{seed}.joblib",
            hp=hp, save_model=save_model, plot_losses=plot_losses,
            query_col=query_col, lib_col=lib_col, topk=topk,
        )
        rows.append([seed] + list(perf))

    df_perf = pd.DataFrame(rows, columns=["seed"] + perf_columns(topk))
    out_path = os.path.join(cv_dir, prefix + f"_performance_stats_{n_runs}runs.tsv")
    df_perf.to_csv(out_path, sep="\t", index=False)

    metric_cols = [c for c in df_perf.columns if c != "seed"]
    print("\n--- Mean across runs ---"); print(df_perf[metric_cols].mean(axis=0))
    print("--- Std  across runs ---");  print(df_perf[metric_cols].std(axis=0))
    print(f"Wrote: {out_path}")
    return df_perf


def _summarize(df_perf: pd.DataFrame, cv_name: str) -> pd.DataFrame:
    metric_cols = [c for c in df_perf.columns if c != "seed"]
    return pd.DataFrame({
        "metric": metric_cols,
        "mean":   df_perf[metric_cols].mean(axis=0).values,
        "std":    df_perf[metric_cols].std(axis=0).values,
        "cv":     cv_name,
        "n_runs": len(df_perf),
    })


def run_experiment(cfg: ExperimentConfig) -> Dict[str, pd.DataFrame]:
    """Run an experiment across all CVs in `cfg.cvs`.

    Returns a dict mapping 'CV1'/'CV2'/'CV3' -> per-run performance DataFrame.
    Writes a `summary_mean_std.tsv` under `cfg.run_dir()`.
    """
    out_root = cfg.run_dir()
    ensure_dir(out_root)

    df, non_feature_cols = load_features(
        feature=cfg.feature,
        giv_dir=cfg.giv_dir, iter_num=cfg.giv_iter_num, llm_path=cfg.llm_path,
        cell_line=cfg.cell_line, gi_threshold=cfg.gi_threshold,
        query_col=cfg.query_col, lib_col=cfg.lib_col,
    )
    print(f"[{cfg.feature}][{cfg.cell_line}] data shape: {df.shape}  "
          f"| pos rate: {df[cfg.label_col].mean():.4f}  "
          f"| #features: {df.shape[1] - len(non_feature_cols)}")

    results: Dict[str, pd.DataFrame] = {}
    for cv in cfg.cvs:
        prefix_tag = f"{cfg.feature}_CV{cv}"
        df_perf = run_cv_repeats(
            cv=cv, df_score=df, label_col=cfg.label_col, non_feature_cols=non_feature_cols,
            output_root=out_root, cv_split=cfg.cv_splits, hp=cfg.hp,
            n_runs=cfg.n_runs, seed_base=cfg.seed_base, topk=cfg.topk,
            query_col=cfg.query_col, lib_col=cfg.lib_col,
            prefix_tag=prefix_tag, save_model=cfg.save_models,
            plot_losses=cfg.plot_losses,
        )
        results[f"CV{cv}"] = df_perf

    summary = pd.concat([_summarize(df_perf, cv_name) for cv_name, df_perf in results.items()],
                        ignore_index=True)
    summary_path = os.path.join(out_root, "summary_mean_std.tsv")
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"Wrote: {summary_path}")
    return results
