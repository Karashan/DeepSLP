"""Hyperparameter tuning with Optuna for the SL prediction MLP.

Usage
-----
    python -m src.tune --data-dir data/input/GIV_24Q4/ReLU128_5L \\
                       --label-col GI_stringent_Type2 \\
                       --n-trials 50 \\
                       --output-dir outputs/tuning

Optuna explores model architecture (hidden sizes, dropout, number of layers),
optimiser settings (learning rate), loss function parameters (focal alpha/gamma),
and training knobs (batch size) — all driven by validation AUPR.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import optuna
import torch

from .data import (
    NON_FEATURE_COLS,
    filter_unique_pairs,
    load_tsv_iterations,
    scale_splits,
    split_query_holdout,
    split_query_kfold,
    split_random_stratified,
)
from .models import FocalLoss, SLPredictorMLP
from .training import evaluate, get_device, make_loaders, train_model


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def _build_objective(
    df,
    label_col: str,
    cv: int,
    n_folds: int,
    seed: int,
    non_feature_cols: list[str],
    epochs: int,
    device: torch.device,
):
    """Return an Optuna objective function closed over the dataset."""

    def objective(trial: optuna.Trial) -> float:
        # --- sample hyperparameters ---
        n_layers = trial.suggest_int("n_layers", 2, 5)
        hidden_sizes: list[int] = []
        prev = None
        for i in range(n_layers):
            lo = 16
            hi = min(prev or 512, 512)
            h = trial.suggest_int(f"hidden_{i}", lo, hi, log=True)
            hidden_sizes.append(h)
            prev = h

        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.05)
        lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.5, 1.0, 5.0])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        focal_alpha = trial.suggest_float("focal_alpha", 0.25, 1.0, step=0.05)
        focal_gamma = trial.suggest_float("focal_gamma", 0.5, 3.0, step=0.25)
        scheduler_type = trial.suggest_categorical("scheduler_type", ["plateau", "cosine"])
        balanced_sampling = trial.suggest_categorical("balanced_sampling", [True, False])

        # --- cross-validation loop ---
        fold_auprs: list[float] = []
        folds = range(1, n_folds + 1) if cv == 4 else [1]  # single split for cv 1/2

        for fold in folds:
            if cv == 1:
                split = split_random_stratified(df, label_col, seed=seed,
                                                non_feature_cols=non_feature_cols)
            elif cv == 2:
                split = split_query_holdout(df, label_col, seed=seed,
                                            non_feature_cols=non_feature_cols)
            else:
                split = split_query_kfold(df, label_col, fold=fold, seed=seed,
                                          n_folds=n_folds,
                                          non_feature_cols=non_feature_cols)

            split, _ = scale_splits(split)

            # Compute positive-class prior for output head bias init
            import numpy as _np
            y_tr = split["y_train"]
            if hasattr(y_tr, "values"):
                y_tr = y_tr.values
            pos_prior = float(_np.mean(y_tr))

            train_loader, val_loader, _ = make_loaders(
                split, batch_size, balanced_sampling=balanced_sampling,
            )

            input_size = split["X_train"].shape[1]
            model = SLPredictorMLP(input_size, hidden_sizes, dropout, pos_prior=pos_prior)
            criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

            history = train_model(
                model, train_loader, val_loader,
                criterion=criterion, lr=lr, weight_decay=weight_decay,
                max_grad_norm=max_grad_norm, epochs=epochs,
                patience=5, scheduler_type=scheduler_type, device=device,
            )
            model.load_state_dict(history["best_state"])
            model.to(device)
            aupr, _ = evaluate(model, val_loader, device)
            fold_auprs.append(aupr)

            # Optuna pruning (report intermediate value per fold)
            trial.report(aupr, fold - 1)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(sum(fold_auprs) / len(fold_auprs))

    return objective


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hyperparameter tuning for DeepSLP")
    p.add_argument("--data-dir", required=True, help="Directory with numbered TSV files")
    p.add_argument("--n-iters", type=int, default=20, help="Number of TSV iterations")
    p.add_argument("--label-col", default="GI_stringent_Type2", help="Binary label column")
    p.add_argument("--cv", type=int, default=2, choices=[1, 2, 4],
                   help="CV strategy: 1=random-stratified, 2=query-holdout, 4=k-fold")
    p.add_argument("--n-folds", type=int, default=10, help="Number of folds for cv=4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50, help="Max epochs per trial")
    p.add_argument("--n-trials", type=int, default=50, help="Optuna trial budget")
    p.add_argument("--output-dir", default="outputs/tuning")
    p.add_argument("--study-name", default="deepslp_tune")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()

    # load data
    print(f"Loading data from {args.data_dir} ...")
    df = load_tsv_iterations(args.data_dir, n_iters=args.n_iters)
    df = filter_unique_pairs(df)
    print(f"Loaded {len(df)} unique gene pairs")

    # create Optuna study (maximise validation AUPR)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )

    objective = _build_objective(
        df, args.label_col, args.cv, args.n_folds, args.seed,
        NON_FEATURE_COLS, args.epochs, device,
    )

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # --- report ---
    print("\n=== Best Trial ===")
    best = study.best_trial
    print(f"  Value (mean val AUPR): {best.value:.4f}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # save results
    results = {
        "best_value": best.value,
        "best_params": best.params,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
            for t in study.trials
        ],
    }
    results_path = output_dir / "tuning_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Optuna built-in plots (saved as HTML)
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(str(output_dir / "optimization_history.html"))
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(str(output_dir / "param_importances.html"))
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(str(output_dir / "parallel_coordinate.html"))
        print("Optuna visualisation HTMLs saved.")
    except Exception:
        print("(Optuna visualisation skipped — install plotly for interactive plots)")


if __name__ == "__main__":
    main()
