"""H2018 NN workflow: modular pipeline for training/evaluating the DeepSLP NN
on Horlbeck 2018 data with switchable cell line (K562/Jurkat),
feature source (GIV / LLM embeddings), and CV scenario (CV1/CV2/CV3).

See `scripts/run_experiment.py` and `scripts/compare_features.py` for CLIs.
"""

from .config import ExperimentConfig, HyperParams, CV_SPLIT_DEFAULTS
from .model import NeuralNetwork, FocalLoss
from .data import (
    load_giv_features,
    load_llm_embedding_features,
    load_features,
)
from .splits import split_train_val_test_cv
from .train import train_validate_test_nn
from .pipeline import optim_nn_pipeline, run_cv_repeats, run_experiment
from .compare import load_perf_stats, plot_metric_bar

__all__ = [
    "ExperimentConfig",
    "HyperParams",
    "CV_SPLIT_DEFAULTS",
    "NeuralNetwork",
    "FocalLoss",
    "load_giv_features",
    "load_llm_embedding_features",
    "load_features",
    "split_train_val_test_cv",
    "train_validate_test_nn",
    "optim_nn_pipeline",
    "run_cv_repeats",
    "run_experiment",
    "load_perf_stats",
    "plot_metric_bar",
]
