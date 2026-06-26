"""Visualize the 256-dim GIV latent feature space of the ~4M gene pairs.

Reads all latent features from data/input/GIV_24Q4_3L/ (20 shards,
~200k pairs each, 256 features = ko_0..ko_127 + exp_0..exp_127), then projects
representative subsamples to 2-D with UMAP and t-SNE (openTSNE) to reveal global
structure.

Why subsample? UMAP/t-SNE on 4M points is intractable. We still *read* every
pair (so scaling/PCA reflect the full distribution), fit StandardScaler + PCA(50)
on the full set, and embed:
  * a uniform random sample (default 300k) with UMAP,
  * a uniform random sample (default 80k) with t-SNE,
  * with ALL positive pairs (GI_stringent_Type2 == 1) force-included so the rare
    positive class is visible.

Outputs (a fresh, timestamped dir under data/interim/):
  * umap_2d.npy / tsne_2d.npy  -- embeddings
  * sample_meta_umap.tsv / sample_meta_tsne.tsv -- Gene, Query, qGI, labels
  * umap.png / tsne.png  -- colored by qGI_score and by GI label
  * pca_explained_variance.txt

Usage:
    python reproduce_cv2/visualize_feature_space.py
    python reproduce_cv2/visualize_feature_space.py --umap-sample 300000 --tsne-sample 80000
"""

from __future__ import annotations

import argparse
import glob
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
REPO_ROOT = os.path.dirname(_PKG_DIR)

FEATS = [f"ko_{i}" for i in range(128)] + [f"exp_{i}" for i in range(128)]
META = ["Gene", "Query", "qGI_score", "FDR", "GI_stringent_Type2", "GI_standard_Type2"]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_all(input_dir: str):
    """Load all shards: feature matrix (float32) + metadata frame."""
    files = sorted(
        glob.glob(os.path.join(input_dir, "qGI_24Q4_GIV_ReLU128_*.tsv")),
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )
    log(f"Found {len(files)} shards")
    X_parts, meta_parts = [], []
    dtypes = {c: np.float32 for c in FEATS}
    for i, f in enumerate(files):
        df = pd.read_csv(f, sep="\t", usecols=META + FEATS, dtype=dtypes, low_memory=False)
        X_parts.append(df[FEATS].to_numpy(dtype=np.float32))
        meta_parts.append(df[META])
        log(f"  shard {i}: {len(df)} rows (cum {sum(len(m) for m in meta_parts)})")
    X = np.concatenate(X_parts, axis=0)
    meta = pd.concat(meta_parts, ignore_index=True)
    del X_parts, meta_parts
    log(f"Loaded full feature matrix: {X.shape}")
    return X, meta


def choose_sample(n_total: int, n_sample: int, pos_idx: np.ndarray, rng) -> np.ndarray:
    """Uniform random sample of size n_sample, with all positives force-included."""
    if n_sample >= n_total:
        return np.arange(n_total)
    base = rng.choice(n_total, size=n_sample, replace=False)
    idx = np.union1d(base, pos_idx)
    return idx


def scatter_panels(emb, meta_s, title, out_png, topk_label="GI_stringent_Type2"):
    """Two-panel figure: colored by qGI_score and by binary GI label."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # Panel 1: qGI_score (robust clip)
    q = meta_s["qGI_score"].to_numpy()
    lo, hi = np.nanpercentile(q, [1, 99])
    m = np.nanmax(np.abs([lo, hi]))
    sc = axes[0].scatter(emb[:, 0], emb[:, 1], c=np.clip(q, -m, m), s=2, cmap="coolwarm",
                         vmin=-m, vmax=m, linewidths=0, alpha=0.6)
    axes[0].set_title(f"{title} — colored by qGI_score")
    axes[0].set_xlabel("dim 1"); axes[0].set_ylabel("dim 2")
    fig.colorbar(sc, ax=axes[0], shrink=0.7, label="qGI_score (clipped 1–99%)")

    # Panel 2: binary label (negatives gray, positives on top)
    y = meta_s[topk_label].to_numpy()
    pos = y == 1
    neg = ~pos
    axes[1].scatter(emb[neg, 0], emb[neg, 1], c="lightgray", s=2, linewidths=0, alpha=0.4,
                    label=f"negative (n={int(neg.sum())})")
    axes[1].scatter(emb[pos, 0], emb[pos, 1], c="crimson", s=8, linewidths=0, alpha=0.9,
                    label=f"{topk_label}=1 (n={int(pos.sum())})")
    axes[1].set_title(f"{title} — colored by {topk_label}")
    axes[1].set_xlabel("dim 1"); axes[1].set_ylabel("dim 2")
    axes[1].legend(markerscale=3, loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    log(f"Saved {out_png}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default=os.path.join(REPO_ROOT, "data", "input", "GIV_24Q4_3L"))
    p.add_argument("--out-dir", default=None)
    p.add_argument("--pca", type=int, default=50, help="PCA dims before UMAP/t-SNE")
    p.add_argument("--umap-sample", type=int, default=300000)
    p.add_argument("--tsne-sample", type=int, default=80000)
    p.add_argument("--umap-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--tsne-perplexity", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(REPO_ROOT, "data", "interim", f"feature_viz_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    log(f"Output dir: {out_dir}")

    # ---- Load all features ----
    X, meta = load_all(args.input_dir)
    # Drop rows with any non-finite feature (cannot be scaled / projected)
    finite = np.isfinite(X).all(axis=1)
    n_drop = int((~finite).sum())
    if n_drop:
        log(f"Dropping {n_drop} rows with non-finite features "
            f"({100*n_drop/X.shape[0]:.2f}%)")
        X = X[finite]
        meta = meta.loc[finite].reset_index(drop=True)
    n_total = X.shape[0]
    pos_idx = np.where(meta["GI_stringent_Type2"].to_numpy() == 1)[0]
    log(f"Total pairs: {n_total} | positives (stringent): {len(pos_idx)} "
        f"({100*len(pos_idx)/n_total:.3f}%)")

    # ---- Global scaling + PCA (fit on all) ----
    log("Standardizing (fit on full set)...")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    del X
    log(f"PCA -> {args.pca} dims (randomized, fit on full set)...")
    t0 = time.time()
    pca = PCA(n_components=args.pca, svd_solver="randomized", random_state=args.seed)
    Xp = pca.fit_transform(Xs).astype(np.float32)
    del Xs
    ev = pca.explained_variance_ratio_
    log(f"PCA done in {time.time()-t0:.1f}s | top-10 EVR: {np.round(ev[:10], 4)} | "
        f"cumulative({args.pca}) = {ev.sum():.3f}")
    np.savetxt(os.path.join(out_dir, "pca_explained_variance.txt"),
               np.c_[np.arange(1, len(ev)+1), ev, np.cumsum(ev)],
               header="component explained_var_ratio cumulative", fmt=["%d", "%.6f", "%.6f"])

    # ---- UMAP ----
    import umap  # local import so the script still loads if only t-SNE wanted
    u_idx = choose_sample(n_total, args.umap_sample, pos_idx, rng)
    log(f"UMAP on {len(u_idx)} points (neighbors={args.umap_neighbors}, min_dist={args.umap_min_dist})...")
    t0 = time.time()
    reducer = umap.UMAP(n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist,
                        n_components=2, metric="euclidean", random_state=args.seed, verbose=True)
    emb_u = reducer.fit_transform(Xp[u_idx])
    log(f"UMAP done in {time.time()-t0:.1f}s")
    np.save(os.path.join(out_dir, "umap_2d.npy"), emb_u)
    meta_u = meta.iloc[u_idx].reset_index(drop=True)
    meta_u.to_csv(os.path.join(out_dir, "sample_meta_umap.tsv"), sep="\t", index=False)
    scatter_panels(emb_u, meta_u, "UMAP", os.path.join(out_dir, "umap.png"))

    # ---- t-SNE (openTSNE) ----
    from openTSNE import TSNE
    t_idx = choose_sample(n_total, args.tsne_sample, pos_idx, rng)
    log(f"t-SNE on {len(t_idx)} points (perplexity={args.tsne_perplexity})...")
    t0 = time.time()
    tsne = TSNE(n_components=2, perplexity=args.tsne_perplexity, metric="euclidean",
                initialization="pca", n_jobs=-1, random_state=args.seed, verbose=True)
    emb_t = np.asarray(tsne.fit(Xp[t_idx]))
    log(f"t-SNE done in {time.time()-t0:.1f}s")
    np.save(os.path.join(out_dir, "tsne_2d.npy"), emb_t)
    meta_t = meta.iloc[t_idx].reset_index(drop=True)
    meta_t.to_csv(os.path.join(out_dir, "sample_meta_tsne.tsv"), sep="\t", index=False)
    scatter_panels(emb_t, meta_t, "t-SNE", os.path.join(out_dir, "tsne.png"))

    log("===== DONE =====")
    log(f"All artifacts saved under: {out_dir}")


if __name__ == "__main__":
    main()
