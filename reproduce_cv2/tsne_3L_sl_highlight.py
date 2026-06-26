"""t-SNE of the 3L feature space with synthetic-lethal (SL) pairs highlighted.

Three plots, each colored by GI_stringent_Type2 (SL=1 highlighted, all SL pairs
kept; negatives subsampled):
  1. training (GIV_24Q4_3L, ~4M pairs) alone
  2. external validation (GIV_24Q4_heldout, ~2M pairs) alone
  3. combined (train + heldout) on one map  [+ a dataset-colored panel]

Both datasets are loaded by reading all shards -> concat -> unique-pair / lowest-
FDR dedup (data.load_input), then non-finite-feature rows are dropped.

Per plot: StandardScaler + PCA(50) fit on a large random subset of that plot's
data, then openTSNE on a subsample that KEEPS ALL SL pairs plus random negatives.

Outputs -> data/interim/tsne_3L_sl_<timestamp>/
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)
REPO_ROOT = os.path.dirname(_PKG_DIR)

from data import load_input  # noqa: E402

KO = [f"ko_{i}" for i in range(128)]
EXP = [f"exp_{i}" for i in range(128)]
FEATS = KO + EXP
LABEL = "GI_stringent_Type2"


def log(m: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def load_dataset(input_dir: str, iter_num: int, name: str) -> pd.DataFrame:
    log(f"Loading {name}: {input_dir} (iter_num={iter_num}) ...")
    df = load_input(input_dir, iter_num=iter_num)
    keep = ["Gene", "Query", "qGI_score", LABEL] + FEATS
    df = df[[c for c in keep if c in df.columns]].copy()
    finite = np.isfinite(df[FEATS].to_numpy(np.float32)).all(axis=1)
    if (~finite).sum():
        log(f"  {name}: dropping {int((~finite).sum())} non-finite-feature rows")
    df = df.loc[finite].reset_index(drop=True)
    df["dataset"] = name
    n_pos = int((df[LABEL] == 1).sum())
    log(f"  {name}: {len(df)} pairs | SL positives={n_pos} ({100*n_pos/len(df):.3f}%)")
    return df


def subsample_keep_positives(df, target, seed):
    """All SL=1 rows + random negatives up to `target` total."""
    rng = np.random.default_rng(seed)
    pos = np.where(df[LABEL].to_numpy() == 1)[0]
    neg = np.where(df[LABEL].to_numpy() != 1)[0]
    n_neg = max(0, min(len(neg), target - len(pos)))
    neg_s = rng.choice(neg, size=n_neg, replace=False)
    idx = np.concatenate([pos, neg_s])
    rng.shuffle(idx)
    return idx


def embed(X, sub_idx, pca_dims, perplexity, seed, fit_cap=1_000_000):
    from openTSNE import TSNE
    rng = np.random.default_rng(seed)
    fit_idx = rng.choice(X.shape[0], size=min(fit_cap, X.shape[0]), replace=False)
    sc = StandardScaler().fit(X[fit_idx])
    pca = PCA(n_components=pca_dims, svd_solver="randomized", random_state=seed)
    pca.fit(sc.transform(X[fit_idx]))
    Xp = pca.transform(sc.transform(X[sub_idx])).astype(np.float32)
    log(f"  PCA cumulative EVR({pca_dims})={pca.explained_variance_ratio_.sum():.3f}")
    t0 = time.time()
    tsne = TSNE(n_components=2, perplexity=perplexity, metric="euclidean",
                initialization="pca", n_jobs=-1, random_state=seed, verbose=True)
    emb = np.asarray(tsne.fit(Xp))
    log(f"  t-SNE done in {time.time()-t0:.1f}s")
    return emb


def plot_sl(ax, emb, y, title):
    pos = y == 1
    ax.scatter(emb[~pos, 0], emb[~pos, 1], c="lightgray", s=2, linewidths=0, alpha=0.35,
               label=f"non-SL (n={int((~pos).sum())})")
    ax.scatter(emb[pos, 0], emb[pos, 1], c="crimson", s=10, linewidths=0, alpha=0.9,
               label=f"SL (GI_stringent_Type2=1, n={int(pos.sum())})")
    ax.set_title(title); ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    ax.legend(markerscale=3, loc="best")


def plot_dataset(ax, emb, ds, title):
    for name, col in [("train", "#1f77b4"), ("heldout", "#d62728")]:
        m = ds == name
        ax.scatter(emb[m, 0], emb[m, 1], c=col, s=3, linewidths=0, alpha=0.4,
                   label=f"{name} (n={int(m.sum())})")
    ax.set_title(title); ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    ax.legend(markerscale=4, loc="best")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dir", default=os.path.join(REPO_ROOT, "data", "input", "GIV_24Q4_3L"))
    p.add_argument("--train-iter", type=int, default=20)
    p.add_argument("--heldout-dir", default=os.path.join(REPO_ROOT, "data", "input", "GIV_24Q4_heldout"))
    p.add_argument("--heldout-iter", type=int, default=11)
    p.add_argument("--target-single", type=int, default=80000, help="points per individual plot")
    p.add_argument("--target-combined", type=int, default=120000)
    p.add_argument("--pca", type=int, default=50)
    p.add_argument("--perplexity", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(REPO_ROOT, "data", "interim", f"tsne_3L_sl_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    log(f"Output dir: {out_dir}")

    df_tr = load_dataset(args.train_dir, args.train_iter, "train")
    df_he = load_dataset(args.heldout_dir, args.heldout_iter, "heldout")

    # ---- Plot 1: training alone ----
    log("===== PLOT 1: TRAINING (3L, ~4M) =====")
    Xtr = df_tr[FEATS].to_numpy(np.float32)
    idx = subsample_keep_positives(df_tr, args.target_single, args.seed)
    emb = embed(Xtr, idx, args.pca, args.perplexity, args.seed)
    y = df_tr[LABEL].to_numpy()[idx]
    np.save(os.path.join(out_dir, "tsne_train_2d.npy"), emb)
    df_tr.iloc[idx][["Gene", "Query", "qGI_score", LABEL]].to_csv(
        os.path.join(out_dir, "sample_meta_train.tsv"), sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(11, 9))
    plot_sl(ax, emb, y, f"t-SNE — TRAINING 3L (~4M pairs), SL highlighted\n"
                        f"(all {int((y==1).sum())} SL kept + {int((y!=1).sum())} sampled non-SL)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "tsne_train_SL.png"), dpi=200); plt.close(fig)
    log("Saved tsne_train_SL.png")

    # ---- Plot 2: heldout alone ----
    log("===== PLOT 2: HELDOUT (~2M) =====")
    Xhe = df_he[FEATS].to_numpy(np.float32)
    idx = subsample_keep_positives(df_he, args.target_single, args.seed)
    emb = embed(Xhe, idx, args.pca, args.perplexity, args.seed)
    y = df_he[LABEL].to_numpy()[idx]
    np.save(os.path.join(out_dir, "tsne_heldout_2d.npy"), emb)
    df_he.iloc[idx][["Gene", "Query", "qGI_score", LABEL]].to_csv(
        os.path.join(out_dir, "sample_meta_heldout.tsv"), sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(11, 9))
    plot_sl(ax, emb, y, f"t-SNE — VALIDATION 3L (~2M pairs), SL highlighted\n"
                        f"(all {int((y==1).sum())} SL kept + {int((y!=1).sum())} sampled non-SL)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "tsne_heldout_SL.png"), dpi=200); plt.close(fig)
    log("Saved tsne_heldout_SL.png")

    # ---- Plot 3: combined ----
    log("===== PLOT 3: COMBINED (train + heldout) =====")
    df = pd.concat([df_tr, df_he], ignore_index=True)
    del df_tr, df_he
    Xc = df[FEATS].to_numpy(np.float32)
    idx = subsample_keep_positives(df, args.target_combined, args.seed)  # keeps ALL SL from both
    emb = embed(Xc, idx, args.pca, args.perplexity, args.seed)
    y = df[LABEL].to_numpy()[idx]
    ds = df["dataset"].to_numpy()[idx]
    np.save(os.path.join(out_dir, "tsne_combined_2d.npy"), emb)
    df.iloc[idx][["Gene", "Query", "qGI_score", LABEL, "dataset"]].to_csv(
        os.path.join(out_dir, "sample_meta_combined.tsv"), sep="\t", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    plot_sl(axes[0], emb, y, f"t-SNE — COMBINED 3L (train+heldout), SL highlighted\n"
                             f"(all {int((y==1).sum())} SL kept)")
    plot_dataset(axes[1], emb, ds, "t-SNE — COMBINED, colored by dataset")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "tsne_combined_SL_and_dataset.png"), dpi=200); plt.close(fig)
    log("Saved tsne_combined_SL_and_dataset.png")

    log("===== DONE =====")
    log(f"All artifacts saved under: {out_dir}")


if __name__ == "__main__":
    main()
