"""Characterize why the reproduced ensemble fails externally.

Compares, on the external held-out set:
  * spread (std) of reproduced-ensemble predictions vs original-ensemble,
  * correlation between reproduced and original ensemble predictions,
  * ROC for each.
Also reports the reproduced models' prediction spread on their own internal
test sets (to confirm they are non-degenerate internally).
"""

import glob
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

_PKG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PKG)
import model as _m  # noqa: E402
sys.modules["__main__"].NeuralNetwork = _m.NeuralNetwork
sys.modules["__main__"].FocalLoss = _m.FocalLoss
from data import load_input  # noqa: E402
from config import NON_FEATURE_COLS  # noqa: E402

ORIG = os.path.expanduser("~/DeepSLP/data/interim/ReLU128_f_a075_g15_10folds_pt10")
REPRO = os.path.expanduser("~/DeepSLP/data/interim/repro_ensemble_10folds_20260622_022346")
HELDOUT = os.path.expanduser("~/DeepSLP/data/input/GIV_24Q4_heldout/")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def score(m, s, Xv, bs=8192):
    Xt = torch.tensor(s.transform(Xv), dtype=torch.float32)
    m.eval().to(DEV)
    out = np.empty(Xt.shape[0], np.float32)
    with torch.no_grad():
        for i in range(0, Xt.shape[0], bs):
            b = Xt[i:i + bs].to(DEV)
            out[i:i + b.shape[0]] = torch.sigmoid(m(b)).cpu().numpy()
    return out


def ens(pairs, Xv):
    """pairs: list of (model_path, scaler_path)."""
    acc = np.zeros(len(Xv), np.float64)
    per = []
    for mp, sp in pairs:
        m = torch.load(mp, map_location=DEV, weights_only=False)
        s = joblib.load(sp)
        p = score(m, s, Xv)
        per.append(p)
        acc += p
    return acc / len(pairs), np.array(per)


def orig_pairs():
    out = []
    for k in range(1, 11):
        mp = os.path.join(ORIG, f"CV2_811_GIV_NN_LR1e2_50e_p10_d01_{k}.pth")
        sp = os.path.join(ORIG, f"CV2_811_seed{k}.joblib")
        out.append((mp, sp))
    return out


def repro_pairs():
    out = []
    for k in range(1, 11):
        mp = os.path.join(REPRO, f"fold{k}_model.pth")
        sp = os.path.join(REPRO, f"fold{k}_scaler.joblib")
        out.append((mp, sp))
    return out



def main():
    df = load_input(HELDOUT, iter_num=11, tail=".tsv")
    feat = [c for c in df.columns if c not in NON_FEATURE_COLS]
    df = df.dropna(subset=feat).reset_index(drop=True)
    Xv = df[feat].values
    y_str = df["GI_stringent_Type2"].values.astype(int)

    orig_ens, _ = ens(orig_pairs(), Xv)
    rep_ens, rep_per = ens(repro_pairs(), Xv)

    print("\n================ EXTERNAL PREDICTION CHARACTERISATION ================")
    print(f"{'':22} {'std':>10} {'min':>8} {'max':>8} {'ROC(stringent)':>16}")
    for name, p in [("ORIGINAL ensemble", orig_ens), ("REPRODUCED ensemble", rep_ens)]:
        print(f"{name:22} {p.std():10.5f} {p.min():8.4f} {p.max():8.4f} {roc_auc_score(y_str, p):16.4f}")

    r = np.corrcoef(orig_ens, rep_ens)[0, 1]
    print(f"\nPearson corr(original ens, reproduced ens) on held-out = {r:.4f}")
    print(f"Spearman-ish (rank corr via argsort overlap top-1000): ", end="")
    o_top = set(np.argsort(orig_ens)[::-1][:1000])
    r_top = set(np.argsort(rep_ens)[::-1][:1000])
    print(f"{len(o_top & r_top)}/1000 shared top predictions")

    # reproduced per-fold spread on external
    print("\nReproduced per-fold external prediction std:")
    print("  ", np.round(rep_per.std(axis=1), 5))

    # internal spread sanity for reproduced fold 1
    t = pd.read_csv(os.path.join(REPRO, "fold1_test_predictions.tsv"), sep="\t", low_memory=False)
    print(f"\nReproduced fold1 INTERNAL test predict_proba: std={t['predict_proba'].std():.5f} "
          f"min={t['predict_proba'].min():.4f} max={t['predict_proba'].max():.4f}")


if __name__ == "__main__":
    main()
