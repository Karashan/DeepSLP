"""Tasks 2 & 3: t-SNE of TRAINING (~4M) + EXTERNAL held-out (~2M) pairs combined.

Both datasets are loaded by reading every shard, concatenating, and applying the
same unique-pair / lowest-FDR deduplication used for training
(``data.load_input`` -> ``filter_unique_pairs_by_lowest_fdr``).

Three t-SNE maps are produced on the SAME sampled rows, each colored by dataset
(train vs heldout) so we can see how the two distributions overlap in latent
space:
  * task 2 : all 256 features
  * task 3 : KO-derived features only   (ko_0  .. ko_127)
  * task 3 : Expression features only   (exp_0 .. exp_127)

Per view: StandardScaler + PCA(50) fit on a large random subset of the combined
data, then openTSNE on a balanced subsample (equal #points per dataset).

Outputs go to a fresh data/interim/tsne_train_vs_heldout_<timestamp>/ dir.
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

from data import load_input  # noqa: E402  (does concat + dedup + drop self-GIs)

KO = [f"ko_{i}" for i in range(128)]
EXP = [f"exp_{i}" for i in range(128)]
FEATS = KO + EXP


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_dataset(input_dir: str, iter_num: int, name: str) -> pd.DataFrame:
    log(f"Loading {name}: {input_dir} (iter_num={iter_num}) ...")
    df = load_input(input_dir, iter_num=iter_num)  # concat + dedup unique pairs by lowest FDR
    # keep only what we need
    keep = ["Gene", "Query", "qGI_score", "GI_stringent_Type2"] + FEATS
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    # drop rows with any non-finite feature
    finite = np.isfinite(df[FEATS].to_numpy(dtype=np.float32)).all(axis=1)
    n_drop = int((~finite).sum())
    if n_drop:
        log(f"  {name}: dropping {n_drop} rows with non-finite features")
    df = df.loc[finite].reset_index(drop=True)
    df["dataset"] = name
    log(f"  {name}: {len(df)} unique pairs, {df['Query'].nunique()} query genes")
    return df


def embed_view(X_view, sub_idx, pca_dims, perplexity, seed, fit_cap=1_000_000):
    """StandardScaler+PCA fit on a large random subset; t-SNE on sub_idx."""
    from openTSNE import TSNE
    rng = np.random.default_rng(seed)
    n = X_view.shape[0]
    fit_idx = rng.choice(n, size=min(fit_cap, n), replace=False)
    scaler = StandardScaler().fit(X_view[fit_idx])
    pca = PCA(n_components=pca_dims, svd_solver="randomized", random_state=seed)
    pca.fit(scaler.transform(X_view[fit_idx]))
    Xp_sub = pca.transform(scaler.transform(X_view[sub_idx])).astype(np.float32)
    log(f"  PCA cumulative EVR({pca_dims}) = {pca.explained_variance_ratio_.sum():.3f}")
    t0 = time.time()
    tsne = TSNE(n_components=2, perplexity=perplexity, metric="euclidean",
                initialization="pca", n_jobs=-1, random_state=seed, verbose=True)
    emb = np.asarray(tsne.fit(Xp_sub))
    log(f"  t-SNE done in {time.time()-t0:.1f}s")
    return emb


def plot_by_dataset(emb, ds, title, out_png):
    fig, ax = plt.subplots(figsize=(11, 9))
    order = ["train", "heldout"]
    colors = {"train": "#1f77b4", "heldout": "#d62728"}
    # draw the larger group first
    for name in sorted(order, key=lambda s: -(ds == s).sum()):
        m = ds == name
        ax.scatter(emb[m, 0], emb[m, 1], c=colors[name], s=3, linewidths=0, alpha=0.45,
                   label=f"{name} (n={int(m.sum())})")
    ax.set_title(title)
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    ax.legend(markerscale=4, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    log(f"Saved {out_png}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dir", default=os.path.join(REPO_ROOT, "data", "input", "GIV_24Q4_3L"))
    p.add_argument("--train-iter", type=int, default=20)
    p.add_argument("--heldout-dir", default=os.path.join(REPO_ROOT, "data", "input", "GIV_24Q4_heldout"))
    p.add_argument("--heldout-iter", type=int, default=11)
    p.add_argument("--per-dataset", type=int, default=60000, help="t-SNE points sampled per dataset")
    p.add_argument("--pca", type=int, default=50)
    p.add_argument("--perplexity", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(REPO_ROOT, "data", "interim", f"tsne_train_vs_heldout_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    log(f"Output dir: {out_dir}")

    # ---- Load both datasets (concat + dedup) ----
    df_tr = load_dataset(args.train_dir, args.train_iter, "train")
    df_he = load_dataset(args.heldout_dir, args.heldout_iter, "heldout")
    df = pd.concat([df_tr, df_he], ignore_index=True)
    del df_tr, df_he
    log(f"Combined: {len(df)} pairs (train+heldout)")

    X = df[FEATS].to_numpy(dtype=np.float32)
    ds_all = df["dataset"].to_numpy()

    # ---- Balanced subsample (equal per dataset) ----
    rng = np.random.default_rng(args.seed)
    idx_parts = []
    for name in ["train", "heldout"]:
        pool = np.where(ds_all == name)[0]
        k = min(args.per_dataset, len(pool))
        idx_parts.append(rng.choice(pool, size=k, replace=False))
        log(f"sampled {k} from {name}")
    sub_idx = np.concatenate(idx_parts)
    rng.shuffle(sub_idx)
    ds_sub = ds_all[sub_idx]

    # save sampled metadata
    df.iloc[sub_idx][["Gene", "Query", "qGI_score", "GI_stringent_Type2", "dataset"]] \
        .to_csv(os.path.join(out_dir, "sample_meta.tsv"), sep="\t", index=False)

    views = {
        "all256": (X, "t-SNE (all 256 features) — train vs heldout"),
        "ko128":  (X[:, :128], "t-SNE (KO features ko_0..127) — train vs heldout"),
        "exp128": (X[:, 128:], "t-SNE (Expression features exp_0..127) — train vs heldout"),
    }
    for key, (Xv, title) in views.items():
        log(f"===== VIEW: {key} (dim={Xv.shape[1]}) =====")
        emb = embed_view(Xv, sub_idx, args.pca, args.perplexity, args.seed)
        np.save(os.path.join(out_dir, f"tsne_{key}_2d.npy"), emb)
        plot_by_dataset(emb, ds_sub, title, os.path.join(out_dir, f"tsne_{key}_by_dataset.png"))

    log("===== DONE =====")
    log(f"All artifacts saved under: {out_dir}")


if __name__ == "__main__":
    main()
