"""Degree-corrected scoring + de-hubbed SL network construction.

Problem: the raw ensemble top list is dominated by a few hub genes whose
embeddings make them look SL-prone with almost everything (per-gene marginal).

Fix (two parts):
  1. Degree-corrected score:
        subtract : c(i,j) = mean(i,j) - 0.5*(m_i + m_j)
        zscore   : c(i,j) = 0.5*((mean-m_i)/sd_i + (mean-m_j)/sd_j)
     where m_g / sd_g are gene g's mean/std of ensemble probability over all of
     its (unscreened) partners. This removes each gene's baseline propensity, so
     universal hubs are penalised.
  2. Edge filtering: keep only MUTUAL top-k edges (j in top-k of i AND i in
     top-k of j) by corrected score. This caps every gene's degree at k and
     requires reciprocity, breaking hub stars.

All ensemble scores are recomputed on GPU into an in-memory (N x N) matrix
(~1.3 GB) so no 17 GB TSV is re-read.

Output: a de-hubbed edge list  gene1, gene2, score(=corrected), raw_mean,
m_g1, m_g2  (sorted by corrected score), plus a summary of the de-hubbing effect.

Usage:
    python build_dehubbed_network.py \
      --models  ../data/interim/ReLU128_f_a075_g15_10folds_pt10/CV2_811_GIV_NN_LR1e2_50e_p10_d01_{1..10}.pth \
      --scalers ../data/interim/ReLU128_f_a075_g15_10folds_pt10/CV2_811_seed{1..10}.joblib \
      --mask-cache ../data/interim/all_pairs_pred/screened_mask_keys.npy \
      --output ../data/interim/all_pairs_pred/dehubbed_network.tsv \
      --topk 25 --corr-method subtract
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch

_PKG = os.path.dirname(os.path.abspath(__file__))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)
import model as _m  # noqa: E402
sys.modules["__main__"].NeuralNetwork = _m.NeuralNetwork
sys.modules["__main__"].FocalLoss = _m.FocalLoss

REPO = os.path.expanduser("~/DeepSLP")
DEF_KO = os.path.join(REPO, "data/input/AE/AE128_5L_std100_CRISPRGeneEffect_24Q4_imputed_gene_wise_mean_align_qGI2021.txt")
DEF_EXP = os.path.join(REPO, "data/input/AE/AE128_5L_std100_Expression_BC_24Q4_align_qGI2021.txt")
DEF_SCREEN = os.path.join(REPO, "data/input/GIV_24Q4/ReLU128_5L")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--scalers", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--ae-ko", default=DEF_KO)
    ap.add_argument("--ae-exp", default=DEF_EXP)
    ap.add_argument("--screened-dir", default=DEF_SCREEN)
    ap.add_argument("--mask-cache", default=None)
    ap.add_argument("--topk", type=int, default=25, help="top partners kept per gene")
    ap.add_argument("--corr-method", choices=["subtract", "zscore"], default="subtract")
    ap.add_argument("--mutual", dest="mutual", action="store_true", default=True,
                    help="keep only reciprocal top-k edges (default)")
    ap.add_argument("--no-mutual", dest="mutual", action="store_false",
                    help="keep union of top-k (looser)")
    ap.add_argument("--block", type=int, default=64, help="query-gene block size for GPU scoring")
    ap.add_argument("--limit-genes", type=int, default=None, help="restrict universe (for quick tests)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if len(args.models) != len(args.scalers):
        ap.error("models and scalers must match in length")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    t_all = time.time()

    # ---- AE -> centered per-gene tensors ----
    df_ko = pd.read_csv(args.ae_ko, sep="\t", index_col=0)
    df_exp = pd.read_csv(args.ae_exp, sep="\t", index_col=0)
    genes = np.intersect1d(df_ko.index, df_exp.index)
    if args.limit_genes:
        genes = genes[: args.limit_genes]
    gene_idx = {g: i for i, g in enumerate(genes)}
    N = len(genes)
    Cko = df_ko.loc[genes].values.astype(np.float32); Cko -= Cko.mean(1, keepdims=True)
    Cexp = df_exp.loc[genes].values.astype(np.float32); Cexp -= Cexp.mean(1, keepdims=True)
    Cko_t = torch.tensor(Cko, device=device); Cexp_t = torch.tensor(Cexp, device=device)
    log(f"AE: N={N} genes | device={device} | models={len(args.models)} | k={args.topk} | corr={args.corr_method}")

    # ---- models + scalers ----
    nets, means, scales = [], [], []
    for mp, sp in zip(args.models, args.scalers):
        nets.append(torch.load(mp, map_location=device, weights_only=False).eval())
        sc = joblib.load(sp)
        means.append(torch.tensor(sc.mean_, dtype=torch.float32, device=device))
        scales.append(torch.tensor(sc.scale_, dtype=torch.float32, device=device))
    Mn = len(nets)

    # ---- screened mask keys ----
    cache = args.mask_cache or os.path.join(os.path.dirname(os.path.abspath(args.output)), f"screened_mask_keys_N{N}.npy")
    if os.path.exists(cache) and not args.limit_genes:
        screened_keys = np.load(cache)
        log(f"Screened mask loaded ({len(screened_keys):,} keys)")
    else:
        shards = sorted(glob.glob(os.path.join(args.screened_dir, "*.tsv")))
        parts = []
        for sh in shards:
            gq = pd.read_csv(sh, sep="\t", usecols=["Gene", "Query"], low_memory=False)
            a = gq["Gene"].map(gene_idx).to_numpy(); b = gq["Query"].map(gene_idx).to_numpy()
            ok = ~(pd.isna(a) | pd.isna(b)); a = a[ok].astype(np.int64); b = b[ok].astype(np.int64)
            lo = np.minimum(a, b); hi = np.maximum(a, b); neq = lo != hi
            parts.append(lo[neq] * N + hi[neq])
        screened_keys = np.unique(np.concatenate(parts)) if parts else np.array([], np.int64)
        if not args.limit_genes:
            np.save(cache, screened_keys)
        log(f"Screened mask built ({len(screened_keys):,} keys)")

    # ---- score sweep: fill mean matrix (N x N) ----
    t = time.time()
    meanmat = np.empty((N, N), dtype=np.float32)

    @torch.no_grad()
    def score_block(qs):
        q = torch.as_tensor(qs, device=device)
        ko = Cko_t[q][:, None, :] + Cko_t[None, :, :]      # (B,N,128)
        exp = Cexp_t[q][:, None, :] + Cexp_t[None, :, :]
        feat = torch.cat([ko, exp], dim=2).reshape(-1, 256)
        acc = torch.zeros(feat.shape[0], device=device)
        for m in range(Mn):
            acc += torch.sigmoid(nets[m]((feat - means[m]) / scales[m]))
        return (acc / Mn).reshape(len(qs), N)

    for s in range(0, N, args.block):
        qs = np.arange(s, min(s + args.block, N))
        meanmat[qs] = score_block(qs).cpu().numpy()
    log(f"Score sweep done in {time.time()-t:.1f}s  (meanmat {meanmat.nbytes/1e9:.2f} GB)")

    # ---- screened + self mask matrix ----
    mask = np.zeros((N, N), dtype=bool)
    np.fill_diagonal(mask, True)
    if len(screened_keys):
        lo = (screened_keys // N).astype(np.int64); hi = (screened_keys % N).astype(np.int64)
        mask[lo, hi] = True; mask[hi, lo] = True

    # ---- per-gene marginal m_g / sd_g over unscreened partners ----
    masked = np.where(mask, np.nan, meanmat)
    m_g = np.nanmean(masked, axis=1).astype(np.float32)
    sd_g = np.nanstd(masked, axis=1).astype(np.float32)
    sd_g[sd_g == 0] = 1.0
    del masked

    # ---- corrected score matrix ----
    if args.corr_method == "subtract":
        corr = meanmat - 0.5 * (m_g[:, None] + m_g[None, :])
    else:  # zscore
        corr = 0.5 * ((meanmat - m_g[:, None]) / sd_g[:, None] + (meanmat - m_g[None, :]) / sd_g[None, :])
    corr[mask] = -np.inf  # never select self/screened

    # ---- per-gene top-k by corrected score ----
    k = min(args.topk, N - 1)
    corr_t = torch.tensor(corr, device=device)
    topv, topi = torch.topk(corr_t, k, dim=1)
    topi = topi.cpu().numpy(); topv = topv.cpu().numpy()
    del corr_t

    # ---- build edges (mutual reciprocal top-k, or union) ----
    rows = np.repeat(np.arange(N), k)
    cols = topi.reshape(-1)
    valid = np.isfinite(topv.reshape(-1))
    rows, cols = rows[valid], cols[valid]
    dir_key = rows.astype(np.int64) * N + cols.astype(np.int64)
    if args.mutual:
        rev_key = cols.astype(np.int64) * N + rows.astype(np.int64)
        keep = np.isin(rev_key, dir_key)
        rows, cols = rows[keep], cols[keep]
    lo = np.minimum(rows, cols); hi = np.maximum(rows, cols)
    canon = np.unique(lo.astype(np.int64) * N + hi.astype(np.int64))
    ei = (canon // N).astype(np.int64); ej = (canon % N).astype(np.int64)
    log(f"Edges after {'mutual' if args.mutual else 'union'} top-{k}: {len(ei):,}")

    # ---- assemble + write ----
    raw = meanmat[ei, ej]
    cscore = (raw - 0.5 * (m_g[ei] + m_g[ej])) if args.corr_method == "subtract" else \
             0.5 * ((raw - m_g[ei]) / sd_g[ei] + (raw - m_g[ej]) / sd_g[ej])
    out = pd.DataFrame({
        "gene1": genes[ei], "gene2": genes[ej],
        "score": cscore.astype(np.float32),
        "raw_mean": raw.astype(np.float32),
        "m_g1": m_g[ei], "m_g2": m_g[ej],
    }).sort_values("score", ascending=False)
    out.to_csv(args.output, sep="\t", index=False, float_format="%.6f")

    # ---- de-hubbing summary (compare degree distributions) ----
    deg = pd.concat([out["gene1"], out["gene2"]]).value_counts()
    summary = [
        "=== De-hubbed SL network ===",
        f"method: corrected={args.corr_method}, filter={'mutual' if args.mutual else 'union'} top-{k}",
        f"edges = {len(out):,}",
        f"unique genes = {deg.size:,}",
        f"mean degree = {2*len(out)/max(deg.size,1):.2f}",
        f"max degree = {int(deg.max())} ({deg.index[0]})   [raw top-list hub was HCLS1=10931]",
        f"degree Gini = {gini(deg.values):.3f}   [raw top-list was 0.828]",
        f"top hub genes: {', '.join(f'{g}({d})' for g, d in deg.head(10).items())}",
        f"corrected score: min={out['score'].min():.4f} median={out['score'].median():.4f} max={out['score'].max():.4f}",
        f"output -> {args.output}",
        f"total time = {time.time()-t_all:.1f}s",
    ]
    txt = "\n".join(summary)
    with open(args.output + ".summary.txt", "w") as fh:
        fh.write(txt + "\n")
    deg.rename_axis("gene").reset_index(name="degree").to_csv(
        args.output.replace(".tsv", "_gene_degree.tsv"), sep="\t", index=False)
    log("===== DONE =====")
    print("\n" + txt)


if __name__ == "__main__":
    main()
