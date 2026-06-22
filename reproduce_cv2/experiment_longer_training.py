"""Experiment: does training longer (less early stopping) help EXTERNAL transfer?

Trains a single CV2 fold (seed 1843) for the full 50 epochs with NO early
stopping, and after every epoch records:
  * internal validation AUPR,
  * internal CV2 test ROC,
  * EXTERNAL held-out ROC (stringent and standard).

If external ROC rises with epochs (even transiently) the "train longer / mild
overfit helps external" hypothesis is supported; if it stays ~0.5 throughout,
training duration is not the cause.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset

_PKG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PKG)
from config import TrainConfig, NON_FEATURE_COLS  # noqa: E402
from data import load_input, split_cv2_query_holdout  # noqa: E402
from model import NeuralNetwork, FocalLoss  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1843


def score(model, Xt, bs=16384):
    model.eval()
    out = np.empty(Xt.shape[0], np.float32)
    with torch.no_grad():
        for i in range(0, Xt.shape[0], bs):
            b = Xt[i:i + bs].to(DEV)
            out[i:i + b.shape[0]] = torch.sigmoid(model(b)).cpu().numpy()
    return out


def val_loss_fn(model, Xt, y, crit, bs=16384):
    model.eval()
    yt = torch.tensor(y, dtype=torch.float32)
    total = 0.0
    with torch.no_grad():
        for i in range(0, Xt.shape[0], bs):
            b = Xt[i:i + bs].to(DEV)
            logits = model(b).cpu()
            total += crit(logits, yt[i:i + b.shape[0]]).item() * b.shape[0]
    return total / Xt.shape[0]


def main():
    cfg = TrainConfig()
    print(f"Device {DEV}; seed {SEED}; full {cfg.num_epochs} epochs, NO early stopping")

    df = load_input(cfg.input_dir, iter_num=cfg.input_glob_iter, tail=cfg.file_tail,
                    query_col=cfg.query_col, lib_col=cfg.lib_col, fdr_col=cfg.fdr_col)
    split = split_cv2_query_holdout(df, label_col=cfg.label_col, non_feature_cols=cfg.non_feature_cols,
                                    query_col=cfg.query_col, test_ratio=cfg.test_ratio,
                                    val_ratio=cfg.val_ratio, rand_seed=SEED)
    feat = [c for c in df.columns if c not in cfg.non_feature_cols]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(split["X_train"])
    Xval = scaler.transform(split["X_val"])
    Xte = scaler.transform(split["X_test"])
    ytr = split["y_train"].values
    yval = split["y_val"].values
    yte = split["y_test"].values

    # External held-out
    heldout_dir = os.path.join(os.path.dirname(os.path.dirname(_PKG)), "data", "input", "GIV_24Q4_heldout") + os.sep
    heldout_dir = os.path.expanduser("~/DeepSLP/data/input/GIV_24Q4_heldout/")
    dfe = load_input(heldout_dir, iter_num=11, tail=".tsv")
    dfe = dfe.dropna(subset=feat).reset_index(drop=True)
    Xext = torch.tensor(scaler.transform(dfe[feat].values), dtype=torch.float32)
    ye_str = dfe["GI_stringent_Type2"].values.astype(int)
    ye_std = dfe["GI_standard_Type2"].values.astype(int)
    print(f"Train={len(ytr)} Val={len(yval)} Test={len(yte)} | External={len(ye_str)} "
          f"(str pos={ye_str.sum()}, std pos={ye_std.sum()})")

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xval_t = torch.tensor(Xval, dtype=torch.float32).to(DEV)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)

    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    model = NeuralNetwork(Xtr.shape[1], cfg.hidden_size1, cfg.hidden_size2, cfg.hidden_size3,
                          cfg.output_size, dropout_p=cfg.dropout_p).to(DEV)
    crit = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    opt = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=cfg.scheduler_factor,
                                                 patience=cfg.scheduler_patience)

    print(f"\n{'ep':>3} {'trLoss':>8} {'valAUPR':>8} {'testROC':>8} {'extROC_str':>11} {'extROC_std':>11}")
    rows = []
    for ep in range(cfg.num_epochs):
        model.train()
        tl = 0.0
        for xb, yb in loader:
            xb = xb.to(DEV); yb = yb.to(DEV).float()
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward(); opt.step()
            tl += loss.item() * xb.size(0)
        tl /= len(loader.dataset)

        # val metrics + scheduler step on true val loss
        pv = score(model, Xval_t)
        val_aupr = average_precision_score(yval, pv)
        vl = val_loss_fn(model, Xval_t, yval, crit)
        sched.step(vl)

        pte = score(model, Xte_t)
        test_roc = roc_auc_score(yte, pte)
        pe = score(model, Xext)
        ext_str = roc_auc_score(ye_str, pe)
        ext_std = roc_auc_score(ye_std, pe)
        print(f"{ep+1:>3} {tl:>8.4f} {val_aupr:>8.4f} {test_roc:>8.4f} {ext_str:>11.4f} {ext_std:>11.4f}")
        rows.append(dict(epoch=ep + 1, train_loss=tl, val_aupr=val_aupr, test_roc=test_roc,
                         ext_roc_stringent=ext_str, ext_roc_standard=ext_std))

    out_dir = os.path.expanduser("~/DeepSLP/data/interim/experiment_longer_training")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"epoch_curve_seed{SEED}.tsv"), sep="\t", index=False)
    print(f"\nSaved curve to {out_dir}/epoch_curve_seed{SEED}.tsv")
    best = max(rows, key=lambda r: r["ext_roc_stringent"])
    print(f"BEST external stringent ROC = {best['ext_roc_stringent']:.4f} at epoch {best['epoch']} "
          f"(test ROC {best['test_roc']:.4f})")


if __name__ == "__main__":
    main()
