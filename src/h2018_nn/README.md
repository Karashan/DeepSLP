# h2018_nn

Modular refactor of the `DeepSLP_H2018_LLM.ipynb` workflow for training the
DeepSLP neural network on Horlbeck 2018 synthetic-lethal data. The public API
matches the notebook's final optimal settings exactly (same architecture, loss,
optimizer, scheduler, and early stopping on validation AUPR), while allowing
easy switching of:

- **Cell line**: `K562` or `Jurkat` (labels `K_SL_n3` / `J_SL_n3`, threshold `GI < -3`).
- **Feature source**: `GIV` (concat'd `H2018_map3_24Q4_GIV_*.tsv`) or `LLM`
  (pair-level NCBI-description embeddings TSV).
- **CV scenario(s)**: any combination of `CV1` (random pair holdout),
  `CV2` (query-gene holdout), `CV3` (double gene-set holdout).

## Layout

    src/h2018_nn/
        __init__.py
        config.py       # ExperimentConfig, HyperParams, per-CV default ratios
        model.py        # NeuralNetwork, FocalLoss, pick_device
        data.py         # load_giv_features / load_llm_embedding_features / load_features
        splits.py       # split_train_val_test_cv (CV1/2/3)
        metrics.py      # ROC/PR plots, Recall@K / Precision@K, evaluate_metrics
        train.py        # train_validate_test_nn (one full train+eval)
        pipeline.py     # optim_nn_pipeline, run_cv_repeats, run_experiment
        compare.py      # load_perf_stats + plot_metric_bar (cross-feature bars)
        scripts/
            run_experiment.py     # CLI for a full sweep
            compare_features.py   # CLI for GIV-vs-LLM comparisons

## Quick start (Python)

```python
from h2018_nn import ExperimentConfig, run_experiment

cfg = ExperimentConfig(cell_line="K562", feature="GIV",
                       cvs=[1, 2, 3], n_runs=10,
                       output_root="/home/b-xiangzhang/DeepSLP/outputs/h2018_nn_runs/")
run_experiment(cfg)
```

Re-run with Jurkat or LLM features by changing `cell_line` / `feature`.

## CLI

Run a full experiment (10 seeds × 3 CVs):

```bash
cd /home/b-xiangzhang/DeepSLP/src
python -m h2018_nn.scripts.run_experiment \
    --cell-line K562 --feature GIV --cvs 1 2 3 --n-runs 10 \
    --output-root /home/b-xiangzhang/DeepSLP/outputs/h2018_nn_runs/
```

Compare two pre-computed runs (GIV vs LLM):

```bash
python -m h2018_nn.scripts.compare_features \
    --cell-line K562 \
    --feature GIV=/home/b-xiangzhang/DeepSLP/outputs/H2018_reproduce_04172026 \
    --feature LLM=/home/b-xiangzhang/DeepSLP/outputs/H2018_embeddings_04172026 \
    --metric AUPR --all-metrics \
    --output-dir /home/b-xiangzhang/DeepSLP/outputs/H2018_GIV_vs_LLM_comparison/
```

The comparison script auto-detects both layouts used by the notebook:
`<ROOT>/<cell_line>/CV*/*_performance_stats_*runs.tsv` and
`<ROOT>/<cell_line>/repeats/CV*/*_performance_stats_*runs.tsv`.

## Output layout

Each experiment writes under `<output_root>/<feature>/<cell_line>/[<tag>/]`:

    CV1/
        CV1_NN_...pth              # per-seed checkpoints
        CV1_NN_..._preds_<i>.tsv   # per-seed test predictions
        CV1_NN_..._ROC_PR_curves_<i>.pdf
        CV{1,2,3}_seed<seed>.joblib  # fitted StandardScaler
        <prefix>_performance_stats_<N>runs.tsv
    CV2/ ...
    CV3/ ...
    summary_mean_std.tsv           # long-format mean/std per metric per CV
