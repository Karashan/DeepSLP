"""Phase 3: train the ULTIMATE 10-fold ensemble with the chosen config and run a
SINGLE external held-out test.

Chosen config (from Phase 1 internal screening): BCE loss with a balanced-batch
WeightedRandomSampler (oversamples positives so each batch is ~class-balanced),
which gave the strongest genuine positive-class learning and best internal AP.

Protocol (no external leakage):
  * train 10 CV2 folds (seeds 1843..1852) entirely within the ~4M training pairs,
  * per fold, select the best epoch by INTERNAL validation AUPR (restore weights),
  * ensemble the 10 models and evaluate ONCE on the external held-out set.

All artifacts go to a fresh timestamped dir under data/interim/.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

_PKG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PKG)
from config import TrainConfig, REPO_ROOT  # noqa: E402
from data import load_input, split_cv2_query_holdout, feature_columns  # noqa: E402
from metrics import compute_metrics, safe_auroc  # noqa: E402
from model import NeuralNetwork  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 25
BATCH = 512
N_FOLDS = 10
SEED_OFFSET = 1842


def log(m):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {m}", flush=True)


def score(model, Xt, bs=16384):
    model.eval()
    out = np.empty(Xt.shape[0], np.float32)
    with torch.no_grad():
        for i in range(0, Xt.shape[0], bs):
            b = Xt[i:i + bs].to(DEV)
            out[i:i + b.shape[0]] = torch.sigmoid(model(b)).cpu().numpy()
    return out


def train_fold(cfg, Xtr, ytr, Xval, yval, Xte, yte):
    npos, nneg = int(ytr.sum()), int((ytr == 0).sum())
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    w = torch.where(ytr_t == 1, torch.tensor(1.0 / npos), torch.tensor(1.0 / nneg)).double()
    sampler = WeightedRandomSampler(w, num_samples=len(ytr_t), replacement=True)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=BATCH, sampler=sampler, drop_last=True)

    Xval_t = torch.tensor(Xval, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)

    torch.manual_seed(0)
    model = NeuralNetwork(Xtr.shape[1], cfg.hidden_size1, cfg.hidden_size2, cfg.hidden_size3,
                          cfg.output_size, dropout_p=cfg.dropout_p).to(DEV)
    crit = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best = {"val_aupr": -1.0, "state": None, "epoch": 0}
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb = xb.to(DEV); yb = yb.to(DEV)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
        val_aupr = average_precision_score(yval, score(model, Xval_t))
        if val_aupr > best["val_aupr"]:
            best = {"val_aupr": val_aupr, "state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    "epoch": ep + 1}
    model.load_state_dict(best["state"])
    pte = score(model, Xte_t)
    m = compute_metrics(yte, pte, k=cfg.topk)
    m["auroc_sklearn"] = safe_auroc(yte, pte)
    m["best_epoch"] = best["epoch"]
    m["val_aupr"] = best["val_aupr"]
    return model, pte, m


def main():
    cfg = TrainConfig()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(REPO_ROOT, "data", "interim", f"ultimate_balanced_{N_FOLDS}folds_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    heldout_dir = os.path.expanduser("~/DeepSLP/data/input/GIV_24Q4_heldout/")

    log("=== ULTIMATE ensemble: BCE + balanced sampler ===")
    log(f"Run dir: {run_dir} | device {DEV} | folds {N_FOLDS} | epochs {EPOCHS} | batch {BATCH}")
    with open(os.path.join(run_dir, "run_config.json"), "w") as fh:
        json.dump({"loss": "BCEWithLogitsLoss", "sampler": "WeightedRandomSampler(balanced)",
                   "epochs": EPOCHS, "batch": BATCH, "lr": cfg.learning_rate,
                   "hidden": [cfg.hidden_size1, cfg.hidden_size2, cfg.hidden_size3],
                   "dropout": cfg.dropout_p, "select": "best val AUPR", "seeds":
                   [f + SEED_OFFSET for f in range(1, N_FOLDS + 1)]}, fh, indent=2)

    log(f"Loading training data from {cfg.input_dir}")
    df = load_input(cfg.input_dir, iter_num=cfg.input_glob_iter, tail=cfg.file_tail,
                    query_col=cfg.query_col, lib_col=cfg.lib_col, fdr_col=cfg.fdr_col)
    log(f"Processed training data: {df.shape[0]} pairs")

    models = []
    internal = []
    for fold in range(1, N_FOLDS + 1):
        seed = fold + SEED_OFFSET
        log(f"===== Fold {fold}/{N_FOLDS} (seed={seed}) =====")
        sp = split_cv2_query_holdout(df, label_col=cfg.label_col, non_feature_cols=cfg.non_feature_cols,
                                     query_col=cfg.query_col, test_ratio=cfg.test_ratio,
                                     val_ratio=cfg.val_ratio, rand_seed=seed)
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(sp["X_train"]); ytr = sp["y_train"].values
        Xval = scaler.transform(sp["X_val"]); yval = sp["y_val"].values
        Xte = scaler.transform(sp["X_test"]); yte = sp["y_test"].values
        t0 = time.time()
        model, pte, m = train_fold(cfg, Xtr, ytr, Xval, yval, Xte, yte)
        m["fold"] = fold; m["seed"] = seed
        m["n_test_pairs"] = int(len(yte)); m["n_test_positives"] = int(yte.sum())
        internal.append(m)
        torch.save(model, os.path.join(run_dir, f"fold{fold}_model.pth"))
        joblib.dump(scaler, os.path.join(run_dir, f"fold{fold}_scaler.joblib"))
        models.append((model, scaler))
        log(f"Fold {fold}: best_ep={m['best_epoch']} valAUPR={m['val_aupr']:.4f} "
            f"testROC={m['roc_auc']:.4f} testAP={m['average_precision']:.4f} "
            f"P@{cfg.topk}={m[f'precision_at_{cfg.topk}']:.4f} ({time.time()-t0:.0f}s)")

    perf = pd.DataFrame(internal)
    perf.to_csv(os.path.join(run_dir, "internal_test_metrics.tsv"), sep="\t", index=False)
    log(f"INTERNAL mean: ROC={perf['roc_auc'].mean():.4f} AP={perf['average_precision'].mean():.4f} "
        f"P@{cfg.topk}={perf[f'precision_at_{cfg.topk}'].mean():.4f}")

    # -------- SINGLE external held-out test of the ensemble --------
    log("===== FINAL external held-out test (ensemble) =====")
    dfe = load_input(heldout_dir, iter_num=11, tail=cfg.file_tail,
                     query_col=cfg.query_col, lib_col=cfg.lib_col, fdr_col=cfg.fdr_col)
    feat = feature_columns(dfe, cfg.non_feature_cols)
    dfe = dfe.dropna(subset=feat).reset_index(drop=True)
    Xv = dfe[feat].values

    ens = np.zeros(len(dfe), dtype=np.float64)
    for model, scaler in models:
        ens += score(model, torch.tensor(scaler.transform(Xv), dtype=torch.float32))
    ens /= len(models)
    dfe["predict_proba"] = ens

    ext = {"n_pairs": int(len(dfe)), "n_models": len(models)}
    for lab in [cfg.label_col, "GI_standard_Type2"]:
        y = dfe[lab].values.astype(int)
        mm = compute_metrics(y, ens, k=cfg.topk)
        mm["auroc_sklearn"] = safe_auroc(y, ens)
        mm["n_positives"] = int(y.sum())
        ext[lab] = mm
        log(f"[{lab}] ROC={mm['roc_auc']:.4f} PR-AUC={mm['pr_auc']:.4f} AP={mm['average_precision']:.4f} "
            f"P@{cfg.topk}={mm[f'precision_at_{cfg.topk}']:.4f} R@{cfg.topk}={mm[f'recall_at_{cfg.topk}']:.4f} "
            f"pos={mm['n_positives']}")

    keep = [c for c in cfg.non_feature_cols if c in dfe.columns] + ["predict_proba"]
    dfe[keep].to_csv(os.path.join(run_dir, "heldout_ensemble_predictions.tsv"), sep="\t", index=False)
    with open(os.path.join(run_dir, "heldout_ensemble_metrics.json"), "w") as fh:
        json.dump(ext, fh, indent=2)
    log("===== DONE =====")
    log(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
