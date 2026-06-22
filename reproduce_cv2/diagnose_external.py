"""Diagnostic: why is external held-out performance poor for the reproduced model?

Scores data/input/GIV_24Q4_heldout with:
  (a) the original fold-1 checkpoint + its scaler,
  (b) the ensemble (mean prob) of all 10 original checkpoints (+ per-fold scalers),
  (c) the newly reproduced single model + its scaler,
and compares against the notebook's reported numbers.

Also checks feature-distribution alignment between the training scaler and the
held-out features (a proxy for embedding-space mismatch).
"""

import glob
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

_PKG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PKG)
import model as _m  # noqa: E402
sys.modules["__main__"].NeuralNetwork = _m.NeuralNetwork
sys.modules["__main__"].FocalLoss = _m.FocalLoss
from data import load_input  # noqa: E402
from config import NON_FEATURE_COLS  # noqa: E402

ORIG_DIR = os.path.expanduser("~/DeepSLP/data/interim/ReLU128_f_a075_g15_10folds_pt10")
REPRO_DIR = os.path.expanduser("~/DeepSLP/data/interim/repro_single_seed1843_20260621_211446")
HELDOUT = os.path.expanduser("~/DeepSLP/data/input/GIV_24Q4_heldout/")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def patk(y, p, k=100):
    idx = np.argsort(p)[::-1][:k]
    return float(np.sum(np.asarray(y)[idx]) / k)


def metrics(y, p, k=100):
    return dict(roc=roc_auc_score(y, p), ap=average_precision_score(y, p), p100=patk(y, p, k))


def score(model, scaler, Xdf):
    feat = [c for c in Xdf.columns if c not in NON_FEATURE_COLS]
    X = scaler.transform(Xdf[feat].values)
    Xt = torch.tensor(X, dtype=torch.float32)
    model.eval().to(DEVICE)
    out = np.empty(Xt.shape[0], dtype=np.float32)
    with torch.no_grad():
        for s in range(0, Xt.shape[0], 8192):
            b = Xt[s:s + 8192].to(DEVICE)
            out[s:s + b.shape[0]] = torch.sigmoid(model(b).squeeze(-1)).cpu().numpy()
    return out


def main():
    print(f"Device: {DEVICE}")
    df = load_input(HELDOUT, iter_num=11, tail=".tsv")
    feat = [c for c in df.columns if c not in NON_FEATURE_COLS]
    df = df.dropna(subset=feat).reset_index(drop=True)
    print(f"Held-out pairs after dedup+dropna: {len(df)}, queries={df['Query'].nunique()}")

    for lab in ["GI_stringent_Type2", "GI_standard_Type2"]:
        print(f"  positives[{lab}] = {int(df[lab].sum())}")

    fold_paths = sorted(glob.glob(os.path.join(ORIG_DIR, "CV2_811_GIV_NN_LR1e2_50e_p10_d01_*.pth")),
                        key=lambda x: int(re.search(r"_(\d+)\.pth", x).group(1)))

    # (a) original fold-1
    m1 = torch.load(fold_paths[0], map_location=DEVICE, weights_only=False)
    s1 = joblib.load(os.path.join(ORIG_DIR, "CV2_811_seed1.joblib"))
    p_f1 = score(m1, s1, df)

    # feature alignment check
    feat_mean = df[feat].values.mean(axis=0)
    print("\n[Feature alignment] train-scaler mean vs heldout mean (first 5 dims):")
    print("  scaler.mean_[:5] =", np.round(s1.mean_[:5], 3))
    print("  heldout.mean[:5] =", np.round(feat_mean[:5], 3))
    corr = np.corrcoef(s1.mean_, feat_mean)[0, 1]
    print(f"  corr(scaler.mean_, heldout per-dim mean) over 256 dims = {corr:.3f}")

    # (b) ensemble of all 10 original
    ens = np.zeros(len(df), dtype=np.float64)
    for fp in fold_paths:
        k = int(re.search(r"_(\d+)\.pth", fp).group(1))
        mk = torch.load(fp, map_location=DEVICE, weights_only=False)
        sk = joblib.load(os.path.join(ORIG_DIR, f"CV2_811_seed{k}.joblib"))
        ens += score(mk, sk, df)
    ens /= len(fold_paths)

    # (c) reproduced model
    mr = torch.load(os.path.join(REPRO_DIR, "model.pth"), map_location=DEVICE, weights_only=False)
    sr = joblib.load(os.path.join(REPRO_DIR, "scaler.joblib"))
    p_rep = score(mr, sr, df)

    print("\n================ EXTERNAL HELD-OUT COMPARISON ================")
    for lab in ["GI_stringent_Type2", "GI_standard_Type2"]:
        y = df[lab].values.astype(int)
        print(f"\n--- {lab} (pos={y.sum()}/{len(y)}) ---")
        for name, p in [("orig fold-1", p_f1), ("orig 10-model ENSEMBLE", ens), ("reproduced single", p_rep)]:
            m = metrics(y, p)
            print(f"  {name:<24} ROC={m['roc']:.4f}  AP={m['ap']:.4f}  P@100={m['p100']:.4f}")


if __name__ == "__main__":
    main()
