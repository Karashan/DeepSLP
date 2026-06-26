"""Decisive leakage-free check: score the ORIGINAL (previously-trained) .pth
models on the EXTERNAL held-out set (queries unseen by every model), and compare
to the new 3L models (~0.80 AUROC there).

Both the original and new models share the SAME 3L feature background, and the
external held-out set is in that same 3L space. This isolates "old training vs
new training" with no representation or leakage confound. If the original models
score ~0.80 here too (they do), then old and new training generalize equally,
and the original models' strong 0.89 on the 3L *internal* test was just query
leakage (their per-fold splits held out different query genes).
"""

from __future__ import annotations

import glob
import os
import re
import sys

import joblib
import numpy as np
import torch

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

import model as model_module  # noqa: E402
sys.modules["__main__"].NeuralNetwork = model_module.NeuralNetwork
sys.modules["__main__"].FocalLoss = model_module.FocalLoss

from config import TrainConfig  # noqa: E402
from data import load_input, feature_columns  # noqa: E402
from metrics import compute_metrics, safe_auroc  # noqa: E402

REPO = os.path.dirname(_PKG_DIR)
RUN3L = os.path.join(REPO, "data", "interim", "repro_10seeds_20260626_042346")
ORIG = os.path.join(REPO, "data", "interim", "ReLU128_f_a075_g15_10folds_pt10")
HELDOUT = os.path.join(REPO, "data", "input", "GIV_24Q4_heldout") + os.sep
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


def main():
    cfg = TrainConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("Loading external held-out set ...")
    df_ext = load_input(HELDOUT, iter_num=11, tail=cfg.file_tail,
                        query_col=cfg.query_col, lib_col=cfg.lib_col, fdr_col=cfg.fdr_col)
    feat = feature_columns(df_ext, cfg.non_feature_cols)
    valid = df_ext[feat].notna().all(axis=1)
    df_ext = df_ext.loc[valid].reset_index(drop=True)
    X_raw = df_ext[feat].values.astype(np.float32)
    y_str = df_ext["GI_stringent_Type2"].values.astype(int)
    print(f"External: {X_raw.shape[0]} pairs, {int(y_str.sum())} stringent positives, "
          f"{df_ext[cfg.query_col].nunique()} unseen queries\n")

    # Use the 3L seed-1843 scaler to standardise (same as the new 3L models got)
    scaler_3l = joblib.load(os.path.join(RUN3L, "seed_1843", "scaler.joblib"))
    X3 = scaler_3l.transform(X_raw)

    print("=== ORIGINAL (3L-trained) models on EXTERNAL held-out ===")
    paths = sorted(glob.glob(os.path.join(ORIG, "CV2_811_GIV_NN_LR1e2_50e_p10_d01_*.pth")),
                   key=lambda x: int(re.search(r"_(\d+)\.pth$", x).group(1)))
    rocs, aps, p100 = [], [], []
    for pth in paths:
        fold = int(re.search(r"_(\d+)\.pth$", pth).group(1))
        m = torch.load(pth, map_location=device, weights_only=False).to(device)
        em = compute_metrics(y_str, score(m, X3, device), k=TOPK)
        rocs.append(em["roc_auc"]); aps.append(em["average_precision"]); p100.append(em[f"precision_at_{TOPK}"])
        print(f"orig fold {fold:>2}: ROC-AUC={em['roc_auc']:.4f}  AP={em['average_precision']:.4f}  P@{TOPK}={em[f'precision_at_{TOPK}']:.4f}")
    print(f"\nORIGINAL on external: ROC-AUC={np.mean(rocs):.4f}+/-{np.std(rocs):.4f}  "
          f"AP={np.mean(aps):.4f}+/-{np.std(aps):.4f}  P@{TOPK}={np.mean(p100):.4f}+/-{np.std(p100):.4f}")
    print("\n(reference) new 3L models on external: ROC-AUC=0.800+/-0.006  P@100=0.19")

    # Also score original fold-1 with its own training scaler (also a 3L scaler)
    cand = os.path.join(ORIG, "CV2_811_seed1.joblib")
    if os.path.exists(cand):
        sc_orig = joblib.load(cand)
        m1 = torch.load(paths[0], map_location=device, weights_only=False).to(device)
        em = compute_metrics(y_str, score(m1, sc_orig.transform(X_raw), device), k=TOPK)
        print(f"\n(extra) orig fold 1 with its own training scaler: ROC-AUC={em['roc_auc']:.4f}  "
              f"AP={em['average_precision']:.4f}  P@{TOPK}={em[f'precision_at_{TOPK}']:.4f}")


if __name__ == "__main__":
    main()
