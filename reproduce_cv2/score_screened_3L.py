"""Score the ~4M screened training pairs (3L features) with the 10-model ensemble
and compare the ensemble mean probability to the known labels (GI_stringent_Type2).

Reads the pre-built GIV features + labels from data/input/GIV_24Q4_3L, dedups
A-B/B-A pairs (lowest FDR), scores each fold model with its own scaler, and writes
full + trimmed prediction tables alongside the true label, plus recovery metrics.

NOTE: these are the TRAINING pairs, so the ensemble mean is largely in-sample
(each pair's query was a train query in most folds) -> recovery is optimistic.
"""
from __future__ import annotations
import glob, os, sys, time, json
import numpy as np, pandas as pd, joblib, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

_PKG = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _PKG)
import model as _m
sys.modules["__main__"].NeuralNetwork = _m.NeuralNetwork
sys.modules["__main__"].FocalLoss = _m.FocalLoss
from data import filter_unique_pairs_by_lowest_fdr
from metrics import compute_metrics, safe_auroc

REPO = os.path.expanduser("~/DeepSLP")
DIR = f"{REPO}/data/input/GIV_24Q4_3L"
MD = f"{REPO}/data/interim/ReLU128_f_a075_g15_10folds_pt10"
OUT = f"{REPO}/data/interim/screened_recovery_3L"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
KO = [f"ko_{i}" for i in range(128)]; EXP = [f"exp_{i}" for i in range(128)]
FEAT = KO + EXP
LABELS = ["GI_stringent_Type2", "GI_standard_Type2"]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    usecols = ["Gene", "Query", "FDR"] + LABELS + FEAT
    shards = sorted(glob.glob(os.path.join(DIR, "*.tsv")))
    log(f"loading {len(shards)} shards from {DIR}")
    df = pd.concat([pd.read_csv(f, sep="\t", usecols=usecols, low_memory=False) for f in shards],
                   ignore_index=True)
    log(f"raw rows = {len(df):,}")
    df = filter_unique_pairs_by_lowest_fdr(df, col1="Gene", col2="Query", col_fdr="FDR")
    log(f"unique pairs after dedup = {len(df):,}")
    valid = df[FEAT].notna().all(axis=1)
    n_drop = int((~valid).sum())
    if n_drop:
        log(f"dropping {n_drop:,} pairs with missing features")
    df = df[valid].reset_index(drop=True)
    X = df[FEAT].values.astype(np.float32)
    log(f"scoring {len(df):,} pairs x {X.shape[1]} features")

    # load 10 models + scalers
    probs = np.empty((len(df), 10), dtype=np.float32)
    Xt = torch.tensor(X, device=DEV)
    for k in range(1, 11):
        net = torch.load(os.path.join(MD, f"CV2_811_GIV_NN_LR1e2_50e_p10_d01_{k}.pth"),
                         map_location=DEV, weights_only=False).eval()
        sc = joblib.load(os.path.join(MD, f"CV2_811_seed{k}.joblib"))
        mu = torch.tensor(sc.mean_, dtype=torch.float32, device=DEV)
        sd = torch.tensor(sc.scale_, dtype=torch.float32, device=DEV)
        with torch.no_grad():
            for s in range(0, len(df), 1_000_000):
                b = Xt[s:s + 1_000_000]
                probs[s:s + b.shape[0], k - 1] = torch.sigmoid(net((b - mu) / sd)).cpu().numpy()
    mean = probs.mean(axis=1)

    # ---- outputs ----
    full = df[["Gene", "Query"] + LABELS].copy()
    for k in range(10):
        full[f"prob_fold{k+1}"] = probs[:, k]
    full["mean"] = mean
    full = full.rename(columns={"Gene": "gene1", "Query": "gene2"})
    full.to_csv(os.path.join(OUT, "screened_3L_predictions.tsv"), sep="\t", index=False, float_format="%.4f")
    full[["gene1", "gene2", "GI_stringent_Type2", "mean"]].rename(
        columns={"mean": "mean_pred_proba"}).to_csv(
        os.path.join(OUT, "screened_3L_predictions_mean_only.tsv"), sep="\t", index=False, float_format="%.4f")

    # ---- recovery metrics (mean vs labels) ----
    lines = ["=== Recovery of known SLs on the ~4M screened training set (3L) ===",
             f"pairs scored = {len(df):,}", ""]
    perfold = {}
    for lab in LABELS:
        y = df[lab].values.astype(int)
        npos = int(y.sum())
        m = compute_metrics(y, mean, k=100)
        m["auroc"] = safe_auroc(y, mean)
        lines += [f"[{lab}] positives = {npos:,} ({npos/len(y):.3%})",
                  f"   ensemble mean: ROC-AUC={m['auroc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
                  f"AP={m['average_precision']:.4f}  P@100={m['precision_at_100']:.4f}  "
                  f"R@100={m['recall_at_100']:.4f}"]
        # per-fold AUROC spread
        aus = [safe_auroc(y, probs[:, k]) for k in range(10)]
        lines.append(f"   per-fold ROC-AUC: min={min(aus):.4f} max={max(aus):.4f} mean={np.mean(aus):.4f}")
        # score separation
        lines.append(f"   mean score: positives={mean[y==1].mean():.4f} vs negatives={mean[y==0].mean():.4f}")
        lines.append("")
        perfold[lab] = {"auroc": m["auroc"], "pr_auc": m["pr_auc"], "ap": m["average_precision"],
                        "p100": m["precision_at_100"], "r100": m["recall_at_100"], "n_pos": npos}
    lines.append("NOTE: training pairs => ensemble mean is largely IN-SAMPLE (optimistic).")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "screened_3L_recovery_metrics.txt"), "w") as fh:
        fh.write(txt + "\n")
    json.dump(perfold, open(os.path.join(OUT, "screened_3L_recovery_metrics.json"), "w"), indent=2)

    # ---- plots ----
    from sklearn.metrics import precision_recall_curve
    y = df["GI_stringent_Type2"].values.astype(int)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].hist(mean[y == 0], bins=80, density=True, alpha=0.6, label="non-SL", color="gray")
    ax[0].hist(mean[y == 1], bins=80, density=True, alpha=0.6, label="SL (stringent T2)", color="crimson")
    ax[0].set(xlabel="ensemble mean probability", ylabel="density", title="Score by true label")
    ax[0].legend(); ax[0].set_yscale("log")
    pr, rc, _ = precision_recall_curve(y, mean)
    ax[1].plot(rc, pr, color="navy")
    ax[1].axhline(y.mean(), ls="--", color="gray", label=f"baseline={y.mean():.4f}")
    ax[1].set(xlabel="recall", ylabel="precision", title="PR curve (GI_stringent_Type2)")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "screened_3L_recovery.png"), dpi=150); plt.close(fig)

    log(f"DONE in {time.time()-t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
