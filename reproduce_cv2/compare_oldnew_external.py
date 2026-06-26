"""Sanity check: does the NEW on-the-fly inference reproduce the OLD handwritten
external-validation predictions on the same ~2M external pairs?

Old: outputs/external_val/external_val_old_models/ensemble_predictions.tsv
     (predict_proba = mean over 10 models, scored from CURATED heldout features).
New: build GIV features on the fly from AE_3L for the same (Gene, Query) pairs,
     score the same 10 models + per-fold scalers, take the ensemble mean.
"""
import os, sys, numpy as np, pandas as pd, joblib, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

_PKG = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _PKG)
import model as _m
sys.modules["__main__"].NeuralNetwork = _m.NeuralNetwork
sys.modules["__main__"].FocalLoss = _m.FocalLoss

REPO = os.path.expanduser("~/DeepSLP")
OLD = f"{REPO}/outputs/external_val/external_val_old_models/ensemble_predictions.tsv"
AE_KO = f"{REPO}/data/input/AE_3L/AE_std100_CRISPRGeneEffect_24Q4_imputed_gene_wise_mean_align_qGI2021.txt"
AE_EX = f"{REPO}/data/input/AE_3L/AE_std100_Expression_BC_24Q4_align_qGI2021.txt"
MD = f"{REPO}/data/interim/ReLU128_f_a075_g15_10folds_pt10"
OUT = f"{REPO}/data/interim/screened_recovery_3L"  # reuse dir for the figure
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(p):
    d = pd.read_csv(p, sep="\t", index_col=0); return d[~d.index.duplicated()]


def main():
    os.makedirs(OUT, exist_ok=True)
    old = pd.read_csv(OLD, sep="\t", usecols=["Gene", "Query", "GI_stringent_Type2", "predict_proba"],
                      low_memory=False)
    print(f"old predictions: {len(old):,} pairs")

    ko = load(AE_KO); ex = load(AE_EX)
    genes = np.intersect1d(ko.index, ex.index)
    gi = {g: i for i, g in enumerate(genes)}
    Cko = ko.loc[genes].values.astype(np.float32); Cko -= Cko.mean(1, keepdims=True)
    Cex = ex.loc[genes].values.astype(np.float32); Cex -= Cex.mean(1, keepdims=True)
    Cko_t = torch.tensor(Cko, device=DEV); Cex_t = torch.tensor(Cex, device=DEV)

    a = old["Gene"].map(gi).to_numpy(); b = old["Query"].map(gi).to_numpy()
    valid = ~(pd.isna(a) | pd.isna(b))
    print(f"pairs with both genes in AE_3L universe: {int(valid.sum()):,} "
          f"(missing {int((~valid).sum()):,})")
    ai = a[valid].astype(np.int64); bi = b[valid].astype(np.int64)

    # load models + scalers
    nets, mus, sds = [], [], []
    for k in range(1, 11):
        nets.append(torch.load(os.path.join(MD, f"CV2_811_GIV_NN_LR1e2_50e_p10_d01_{k}.pth"),
                               map_location=DEV, weights_only=False).eval())
        sc = joblib.load(os.path.join(MD, f"CV2_811_seed{k}.joblib"))
        mus.append(torch.tensor(sc.mean_, dtype=torch.float32, device=DEV))
        sds.append(torch.tensor(sc.scale_, dtype=torch.float32, device=DEV))

    n = len(ai); new = np.empty(n, np.float32); bs = 500_000
    with torch.no_grad():
        for s in range(0, n, bs):
            it = torch.as_tensor(ai[s:s+bs], device=DEV); jt = torch.as_tensor(bi[s:s+bs], device=DEV)
            feat = torch.cat([Cko_t[it] + Cko_t[jt], Cex_t[it] + Cex_t[jt]], dim=1)
            acc = torch.zeros(feat.shape[0], device=DEV)
            for m in range(10):
                acc += torch.sigmoid(nets[m]((feat - mus[m]) / sds[m]))
            new[s:s+feat.shape[0]] = (acc / 10).cpu().numpy()

    old_p = old.loc[valid, "predict_proba"].to_numpy(np.float64)
    diff = new.astype(np.float64) - old_p
    pr = pearsonr(old_p, new)[0]; sr = spearmanr(old_p, new)[0]
    lines = ["=== OLD (curated/handwritten) vs NEW (on-the-fly) external predictions ===",
             f"pairs compared = {n:,}",
             f"Pearson r  = {pr:.6f}",
             f"Spearman r = {sr:.6f}",
             f"max |diff|    = {np.abs(diff).max():.6f}",
             f"mean |diff|   = {np.abs(diff).mean():.6e}",
             f"median |diff| = {np.median(np.abs(diff)):.6e}",
             f"frac |diff|<1e-3 = {(np.abs(diff)<1e-3).mean():.4%}",
             f"frac |diff|<1e-2 = {(np.abs(diff)<1e-2).mean():.4%}",
             f"old range [{old_p.min():.4f},{old_p.max():.4f}]  new range [{new.min():.4f},{new.max():.4f}]"]
    txt = "\n".join(lines); print(txt)
    with open(os.path.join(OUT, "old_vs_new_external_compare.txt"), "w") as fh:
        fh.write(txt + "\n")

    # scatter + diff hist (subsample for plotting)
    idx = np.random.default_rng(0).choice(n, size=min(n, 200000), replace=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(old_p[idx], new[idx], s=2, alpha=0.2)
    lim = [0, max(old_p.max(), new.max())]
    ax[0].plot(lim, lim, "r--", lw=1)
    ax[0].set(xlabel="OLD predict_proba", ylabel="NEW on-the-fly mean",
              title=f"Old vs New (r={pr:.5f})")
    ax[1].hist(diff, bins=100, color="purple")
    ax[1].set(xlabel="new - old", ylabel="count", title="Prediction difference"); ax[1].set_yscale("log")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "old_vs_new_external_compare.png"), dpi=150); plt.close(fig)
    print(f"\nsaved -> {OUT}/old_vs_new_external_compare.{{txt,png}}")


if __name__ == "__main__":
    main()
