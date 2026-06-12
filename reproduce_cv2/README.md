# CV2 DeepSLP — Standalone Reproduction Package

A clean, self-contained re-implementation of the 10-fold **CV2** genetic-interaction
(SL) classifier whose trained checkpoints live in:

```
data/interim/ReLU128_f_a075_g15_10folds_pt10/
    CV2_811_GIV_NN_LR1e2_50e_p10_d01_{1..10}.pth   # whole pickled models
    CV2_811_seed{1..10}.joblib                     # matching StandardScaler per fold
    CV2_811_GIV_NN_LR1e2_50e_p10_d01_{1..10}.tsv   # held-out test predictions
```

This package is **independent** of the project's `src/` pipeline — it only depends
on `torch`, `scikit-learn`, `pandas`, `numpy`, and `joblib` (already in the
`deepslp` conda environment / `environment.yml`).

## Files

| File | Purpose |
|------|---------|
| `model.py`   | `NeuralNetwork` (256→128→64→32→1 MLP) and `FocalLoss`. |
| `config.py`  | `TrainConfig` dataclass with all recovered hyperparameters + the non-feature column list. |
| `data.py`    | Shard concatenation, A-B/B-A dedup by lowest FDR, CV2 query-holdout splitting. |
| `metrics.py` | ROC-AUC, PR-AUC, average precision, Precision@K / Recall@K. |
| `train.py`   | Reproduces the full 10-fold training run. |
| `predict.py` | Loads a saved model + scaler and scores **new** data. |

## Recovered configuration

| Setting | Value |
|---------|-------|
| Architecture | `Linear(256→128) → BN → ReLU → Dropout(0.3)` ×, then `128→64`, `64→32`, `32→1` |
| Parameters | 43,713 per model |
| Input features | 256 (`ko_0..ko_127` + `exp_0..exp_127`) |
| Label | `GI_stringent_Type2` (binary) |
| CV scheme | CV2 — hold out **whole query genes**; 10 independent splits |
| Split ratio | test 0.1 / val 0.1 / train 0.8 ("811") |
| Seeds | `fold + 1842` → 1843 … 1852 |
| Loss | `FocalLoss(alpha=0.75, gamma=1.5)` |
| Optimizer | Adam, lr = 1e-2 |
| Scheduler | `ReduceLROnPlateau(factor=0.1, patience=5)` on val loss |
| Early stopping | patience = 10, on **best validation AUPR** (best weights restored) |
| Epochs / batch | 50 / 64 |
| Scaling | `StandardScaler` fit on train, saved per fold |

> **Note on `d01`:** the filename token `d01` suggests dropout 0.1, but every saved
> checkpoint has **dropout p = 0.3** (it was hard-coded, never parameterised).
> This package uses the *actual* trained value, 0.3.

## Usage

All commands assume the `deepslp` environment is active:

```bash
conda activate deepslp
```

### Reproduce training (all 10 folds)

```bash
python reproduce_cv2/train.py \
    --input-dir data/input/GIV_24Q4/ReLU128_5L/ \
    --output-dir reproduce_cv2/output/
```

Common options: `--n-folds`, `--start-fold`, `--epochs`, `--lr`, `--batch-size`,
`--label-col`, `--topk`, `--device`. See `--help`.

Outputs per fold (mirroring the originals): `<prefix>_<fold>.pth`,
`CV2_811_seed<fold>.joblib`, `<prefix>_<fold>.tsv`, plus a
`<prefix>_performance_stats_<n>folds.tsv` summary.

### Predict on new data with a saved model

```bash
python reproduce_cv2/predict.py \
    --model  data/interim/ReLU128_f_a075_g15_10folds_pt10/CV2_811_GIV_NN_LR1e2_50e_p10_d01_1.pth \
    --scaler data/interim/ReLU128_f_a075_g15_10folds_pt10/CV2_811_seed1.joblib \
    --input  new_pairs.tsv \
    --output predictions.tsv
```

- New data must contain the 256 feature columns (`ko_0..ko_127`, `exp_0..exp_127`).
- Add `--label-col GI_stringent_Type2 --topk 100` to also report metrics when the
  ground truth is available.
- Output is the input table plus a `predict_proba` column, sorted by score.

The saved models were written with `torch.save(model, ...)` (whole module), so
`predict.py` registers `NeuralNetwork`/`FocalLoss` under `__main__` before
unpickling — no extra setup required.

## Notes / caveats

- Exact bit-for-bit reproduction of the original weights is not guaranteed:
  PyTorch/CUDA/sklearn versions differ from training time (models were saved
  under scikit-learn 1.5.1) and GPU nondeterminism applies. The *procedure*,
  *architecture*, *splits* (given the seeds) and *hyperparameters* are reproduced
  faithfully.
- The original ran from `iter_num=20` feature shards; keep `--iter-num` consistent
  with your input directory.
