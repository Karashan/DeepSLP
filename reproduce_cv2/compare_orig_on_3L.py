"""Apples-to-apples: score the ORIGINAL (previously-trained) .pth models on the
new pipeline's 3L held-out test split, alongside the new 3L model on the same split.

Both the original 10 models and our new models were trained on the SAME 3L
feature background (data/input/GIV_24Q4_3L / AE_3L). The 5L variant was a wrong
turn and is abandoned. So this is purely "old training vs new training" on the
identical 3L representation -- not a comparison of feature spaces.

NOTE: on this internal split the original models look better, but that is QUERY
LEAKAGE -- each original fold held out a different set of query genes, so most of
this split's test queries were in the original models' training data. The
leakage-free comparison is in compare_orig_on_external.py.

It reuses the 3L 10-seed run artifacts (processed data + trained seed-1843 model
+ scaler) and re-derives the deterministic seed-1843 CV2 split.
"""

from __future__ import annotations

import glob
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
import torch

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

import model as model_module  # noqa: E402
# Allow un-pickling the whole-model checkpoints saved from the notebook __main__.
sys.modules["__main__"].NeuralNetwork = model_module.NeuralNetwork
sys.modules["__main__"].FocalLoss = model_module.FocalLoss

from config import TrainConfig  # noqa: E402
from data import split_cv2_query_holdout  # noqa: E402
from metrics import compute_metrics, safe_auroc  # noqa: E402

REPO = os.path.dirname(_PKG_DIR)
RUN3L = os.path.join(REPO, "data", "interim", "repro_10seeds_20260626_042346")
ORIG = os.path.join(REPO, "data", "interim", "ReLU128_f_a075_g15_10folds_pt10")
SEED = 1843
TOPK = 100


def score(model, X, device, bs=8192):
    Xt = torch.tensor(X, dtype=torch.float32)
    out = np.empty(Xt.shape[0], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(0, Xt.shape[0], bs):
            b = Xt[s : s + bs].to(device)
            out[s : s + b.shape[0]] = torch.sigmoid(model(b).squeeze(-1)).cpu().numpy()
    return out


def report(tag, y, p):
    m = compute_metrics(y, p, k=TOPK)
    print(f"{tag:<42} ROC-AUC={m['roc_auc']:.4f}  AP={m['average_precision']:.4f}  "
          f"P@{TOPK}={m[f'precision_at_{TOPK}']:.4f}")
    return m


def main():
    cfg = TrainConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("Loading 3L processed data + re-deriving seed-1843 test split ...")
    df = joblib.load(os.path.join(RUN3L, "processed_train_data.joblib"))
    split = split_cv2_query_holdout(
        df, label_col=cfg.label_col, non_feature_cols=cfg.non_feature_cols,
        query_col=cfg.query_col, test_ratio=cfg.test_ratio, val_ratio=cfg.val_ratio,
        rand_seed=SEED,
    )
    X_test_raw = split["X_test"].values.astype(np.float32)
    y_test = split["y_test"].values.astype(int)
    print(f"3L test set: {X_test_raw.shape[0]} pairs, {int(y_test.sum())} positives, "
          f"{split['X_test'].shape[1]} features\n")

    # The 3L scaler that the new model was trained with (fit on 3L train features)
    scaler_3l = joblib.load(os.path.join(RUN3L, f"seed_{SEED}", "scaler.joblib"))
    X_test_3l = scaler_3l.transform(X_test_raw)

    print("=== BASELINE: new 3L model (seed 1843) on 3L test set ===")
    new_model = torch.load(os.path.join(RUN3L, f"seed_{SEED}", "model.pth"),
                           map_location=device, weights_only=False).to(device)
    report("new 3L model / 3L-scaled features", y_test, score(new_model, X_test_3l, device))

    print("\n=== ORIGINAL (3L-trained) models scored on the new 3L test split ===")
    print("(same 3L representation; NOTE most test queries leaked into original training)\n")
    orig_paths = sorted(
        glob.glob(os.path.join(ORIG, "CV2_811_GIV_NN_LR1e2_50e_p10_d01_*.pth")),
        key=lambda x: int(re.search(r"_(\d+)\.pth$", x).group(1)),
    )
    rocs, aps, p100s = [], [], []
    for pth in orig_paths:
        fold = int(re.search(r"_(\d+)\.pth$", pth).group(1))
        om = torch.load(pth, map_location=device, weights_only=False).to(device)
        m = report(f"orig fold {fold} (3L-trained) / 3L test", y_test, score(om, X_test_3l, device))
        rocs.append(m["roc_auc"]); aps.append(m["average_precision"])
        p100s.append(m[f"precision_at_{TOPK}"])

    print("\n--- Original models on 3L test set: mean +/- std ---")
    print(f"ROC-AUC = {np.mean(rocs):.4f} +/- {np.std(rocs):.4f}")
    print(f"AP      = {np.mean(aps):.4f} +/- {np.std(aps):.4f}")
    print(f"P@{TOPK}   = {np.mean(p100s):.4f} +/- {np.std(p100s):.4f}")

    # Also: original fold-1 with its OWN training scaler (also fit on 3L features)
    print("\n=== (extra) original fold-1 with its own training (3L) scaler ===")
    cand = os.path.join(ORIG, "CV2_811_seed1.joblib")
    if os.path.exists(cand):
        sc_orig = joblib.load(cand)
        X_test_origscale = sc_orig.transform(X_test_raw)
        om1 = torch.load(orig_paths[0], map_location=device, weights_only=False).to(device)
        report("orig fold 1 / own 3L scaler", y_test, score(om1, X_test_origscale, device))


if __name__ == "__main__":
    main()
