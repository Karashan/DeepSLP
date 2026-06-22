"""Phase 1: loss / sampling screening for positive-class learning.

Trains a fresh model on ONE CV2 fold (seed 1843) under several loss/sampling
strategies and compares INTERNAL metrics only. The external held-out set is
NOT touched here (held out for the final test of the ultimate model).

Key diagnostic beyond ROC/AP: positive-vs-negative score separation on the
internal test set — the baseline model underfits positives, so we want to see
whether proper positive weighting actually pushes positive scores up.

Fast settings (batch 512, ~25 epochs, no early stopping) for screening.
"""

from __future__ import annotations

import os
import sys
import time

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
from config import TrainConfig  # noqa: E402
from data import load_input, split_cv2_query_holdout  # noqa: E402
from model import NeuralNetwork, FocalLoss  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1843
EPOCHS = 25
BATCH = 512
OUT = os.path.expanduser("~/DeepSLP/data/interim/experiment_loss_screening")


class BalancedFocalLoss(nn.Module):
    """Focal loss with proper per-class alpha weighting (alpha for positives)."""

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()


def score(model, Xt, bs=16384):
    model.eval()
    out = np.empty(Xt.shape[0], np.float32)
    with torch.no_grad():
        for i in range(0, Xt.shape[0], bs):
            b = Xt[i:i + bs].to(DEV)
            out[i:i + b.shape[0]] = torch.sigmoid(model(b)).cpu().numpy()
    return out


def make_loader(Xtr_t, ytr_t, sampler_kind, npos, nneg):
    if sampler_kind == "balanced":
        # weight each sample inversely to its class frequency
        w_pos = 1.0 / npos
        w_neg = 1.0 / nneg
        weights = torch.where(ytr_t == 1, torch.tensor(w_pos), torch.tensor(w_neg)).double()
        sampler = WeightedRandomSampler(weights, num_samples=len(ytr_t), replacement=True)
        return DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=BATCH, sampler=sampler, drop_last=True)
    return DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=BATCH, shuffle=True, drop_last=True)


def build_criterion(kind, npos, nneg):
    if kind == "focal_global":      # baseline (reproduces original)
        return FocalLoss(alpha=0.75, gamma=1.5)
    if kind == "focal_balanced":    # proper per-class alpha
        return BalancedFocalLoss(alpha=0.75, gamma=2.0)
    if kind == "bce_pw_auto":       # full inverse-frequency weighting
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(nneg / npos), device=DEV))
    if kind == "bce_pw_100":
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(100.0, device=DEV))
    if kind == "bce_balanced_sampler":
        return nn.BCEWithLogitsLoss()
    raise ValueError(kind)


CONFIGS = [
    ("focal_global", "shuffle"),
    ("focal_balanced", "shuffle"),
    ("bce_pw_auto", "shuffle"),
    ("bce_pw_100", "shuffle"),
    ("bce_balanced_sampler", "balanced"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = TrainConfig()
    print(f"Device {DEV}; seed {SEED}; {EPOCHS} epochs; batch {BATCH}; configs: {[c[0] for c in CONFIGS]}")

    df = load_input(cfg.input_dir, iter_num=cfg.input_glob_iter, tail=cfg.file_tail,
                    query_col=cfg.query_col, lib_col=cfg.lib_col, fdr_col=cfg.fdr_col)
    split = split_cv2_query_holdout(df, label_col=cfg.label_col, non_feature_cols=cfg.non_feature_cols,
                                    query_col=cfg.query_col, test_ratio=cfg.test_ratio,
                                    val_ratio=cfg.val_ratio, rand_seed=SEED)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(split["X_train"]); ytr = split["y_train"].values
    Xval = scaler.transform(split["X_val"]); yval = split["y_val"].values
    Xte = scaler.transform(split["X_test"]); yte = split["y_test"].values
    npos, nneg = int(ytr.sum()), int((ytr == 0).sum())
    print(f"Train={len(ytr)} (pos={npos}, neg={nneg}, ratio 1:{nneg/npos:.0f}) "
          f"Val={len(yval)} Test={len(yte)}")

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32); ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xval_t = torch.tensor(Xval, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    input_size = Xtr.shape[1]

    summary = []
    for kind, sampler_kind in CONFIGS:
        torch.manual_seed(0)  # same init across configs for fair comparison
        np.random.seed(0)
        model = NeuralNetwork(input_size, cfg.hidden_size1, cfg.hidden_size2, cfg.hidden_size3,
                              cfg.output_size, dropout_p=cfg.dropout_p).to(DEV)
        crit = build_criterion(kind, npos, nneg)
        opt = optim.Adam(model.parameters(), lr=cfg.learning_rate)
        # BCEWithLogitsLoss with float targets; FocalLoss variants too
        loader = make_loader(Xtr_t, ytr_t.float(), sampler_kind, npos, nneg)

        print(f"\n===== CONFIG {kind} (sampler={sampler_kind}) =====")
        best = {"val_aupr": -1}
        rows = []
        t0 = time.time()
        for ep in range(EPOCHS):
            model.train()
            tl = 0.0
            for xb, yb in loader:
                xb = xb.to(DEV); yb = yb.to(DEV).float()
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward(); opt.step()
                tl += loss.item() * xb.size(0)
            tl /= len(loader.dataset)
            pv = score(model, Xval_t); val_aupr = average_precision_score(yval, pv)
            pte = score(model, Xte_t)
            test_roc = roc_auc_score(yte, pte); test_ap = average_precision_score(yte, pte)
            sep = float(pte[yte == 1].mean() - pte[yte == 0].mean())
            rows.append(dict(config=kind, epoch=ep + 1, train_loss=tl, val_aupr=val_aupr,
                             test_roc=test_roc, test_ap=test_ap, pos_minus_neg=sep))
            if val_aupr > best["val_aupr"]:
                best = dict(epoch=ep + 1, val_aupr=val_aupr, test_roc=test_roc, test_ap=test_ap, sep=sep)
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"  ep{ep+1:>2} trLoss={tl:.4f} valAUPR={val_aupr:.4f} "
                      f"testROC={test_roc:.4f} testAP={test_ap:.4f} pos-neg={sep:+.4f}")
        dt = time.time() - t0
        print(f"  BEST(by valAUPR): epoch {best['epoch']} valAUPR={best['val_aupr']:.4f} "
              f"testROC={best['test_roc']:.4f} testAP={best['test_ap']:.4f} sep={best['sep']:+.4f} ({dt:.0f}s)")
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"curve_{kind}.tsv"), sep="\t", index=False)
        summary.append(dict(config=kind, **best))

    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(OUT, "screening_summary.tsv"), sep="\t", index=False)
    print("\n================ INTERNAL SCREENING SUMMARY (best epoch by val AUPR) ================")
    print(s.to_string(index=False))
    print(f"\nSaved to {OUT}/")


if __name__ == "__main__":
    main()
