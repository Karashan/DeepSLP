"""Stratify external held-out performance by whether genes were seen in training.

Tests the hypothesis: models predict better on held-out pairs whose query/library
genes were seen during training (memorisation/overfitting helps "familiar" genes).

Reports ROC/AP on the external set, stratified by:
  * query_seen  : held-out Query gene appears as a Query in training data
  * lib_seen    : held-out library Gene appears as a Gene in training data
for BOTH the original 10-model ensemble and the reproduced 10-model ensemble.
"""

import os
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
from config import NON_FEATURE_COLS, TrainConfig  # noqa: E402

ORIG = os.path.expanduser("~/DeepSLP/data/interim/ReLU128_f_a075_g15_10folds_pt10")
REPRO = os.path.expanduser("~/DeepSLP/data/interim/repro_ensemble_10folds_20260622_022346")
HELDOUT = os.path.expanduser("~/DeepSLP/data/input/GIV_24Q4_heldout/")
TRAIN_DIR = os.path.expanduser("~/DeepSLP/data/input/GIV_24Q4_3L/")
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
    acc = np.zeros(len(Xv), np.float64)
    for mp, sp in pairs:
        acc += score(torch.load(mp, map_location=DEV, weights_only=False), joblib.load(sp), Xv)
    return acc / len(pairs)


def strat_metrics(y, p, mask, name):
    n = int(mask.sum()); pos = int(y[mask].sum())
    if pos < 1 or pos == n:
        return f"  {name:<26} n={n:>8} pos={pos:>5}  ROC=   n/a   AP=  n/a"
    return (f"  {name:<26} n={n:>8} pos={pos:>5}  "
            f"ROC={roc_auc_score(y[mask], p[mask]):.4f}  AP={average_precision_score(y[mask], p[mask]):.4f}")


def main():
    cfg = TrainConfig()
    # Training gene sets (only need Query + Gene columns -> fast)
    train_q, train_g = set(), set()
    for i in range(20):
        fp = os.path.join(TRAIN_DIR, f"qGI_24Q4_GIV_ReLU128_{i}.tsv")
        if not os.path.exists(fp):
            continue
        t = pd.read_csv(fp, sep="\t", usecols=["Query", "Gene"], low_memory=False)
        train_q |= set(t["Query"].dropna().unique())
        train_g |= set(t["Gene"].dropna().unique())
    train_genes = train_q | train_g
    print(f"Training: {len(train_q)} query genes, {len(train_g)} library genes, {len(train_genes)} union")

    dfe = load_input(HELDOUT, iter_num=11, tail=".tsv")
    feat = [c for c in dfe.columns if c not in NON_FEATURE_COLS]
    dfe = dfe.dropna(subset=feat).reset_index(drop=True)
    Xv = dfe[feat].values
    print(f"External: {len(dfe)} pairs, {dfe['Query'].nunique()} queries, {dfe['Gene'].nunique()} library genes")

    q_seen = dfe["Query"].isin(train_genes).values   # query gene seen anywhere in training
    g_seen = dfe["Gene"].isin(train_genes).values     # library gene seen anywhere in training
    q_as_query = dfe["Query"].isin(train_q).values
    print(f"  held-out Query genes seen in training (any role): {q_seen.mean()*100:.1f}% of pairs; "
          f"as a training Query: {q_as_query.mean()*100:.1f}%")
    print(f"  held-out library Gene seen in training (any role): {g_seen.mean()*100:.1f}% of pairs")
    print(f"  unique held-out queries also seen in training: "
          f"{len(set(dfe['Query'].unique()) & train_genes)}/{dfe['Query'].nunique()}")

    orig = ens([(os.path.join(ORIG, f"CV2_811_GIV_NN_LR1e2_50e_p10_d01_{k}.pth"),
                 os.path.join(ORIG, f"CV2_811_seed{k}.joblib")) for k in range(1, 11)], Xv)
    rep = ens([(os.path.join(REPRO, f"fold{k}_model.pth"),
                os.path.join(REPRO, f"fold{k}_scaler.joblib")) for k in range(1, 11)], Xv)

    for lab in ["GI_stringent_Type2", "GI_standard_Type2"]:
        y = dfe[lab].values.astype(int)
        print(f"\n================ {lab} ================")
        for ename, p in [("ORIGINAL ensemble", orig), ("REPRODUCED ensemble", rep)]:
            print(f"-- {ename} --")
            print(strat_metrics(y, p, np.ones(len(y), bool), "ALL"))
            print(strat_metrics(y, p, q_seen, "query gene seen"))
            print(strat_metrics(y, p, ~q_seen, "query gene UNSEEN"))
            print(strat_metrics(y, p, g_seen, "library gene seen"))
            print(strat_metrics(y, p, g_seen & q_seen, "both seen"))


if __name__ == "__main__":
    main()
