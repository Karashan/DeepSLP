"""Plan-2 integrated inference: predict SL for ALL unscreened gene pairs.

Supports a single model or an ENSEMBLE of models (e.g. the 10 CV2 fold models).
GIV features are built on the fly (no big feature store): for a pair (g1, g2),
    ko_*  = center(AE_ko[g1])  + center(AE_ko[g2])
    exp_* = center(AE_exp[g1]) + center(AE_exp[g2])      center(v) = v - mean(v)
so per-gene centered embeddings are precomputed once and just gathered + added.
Each model applies its OWN StandardScaler before the forward pass.

Outputs
-------
Single model : gene1, gene2, score
Ensemble     : gene1, gene2, prob_fold1 ... prob_foldM, mean
Plus a top-fraction file (default top 0.3% by mean / score).

The screened-pair mask is cached to a .npy for reuse.
All stage timings are written to <output>.timing.txt.

Examples
--------
# Ensemble of the 10 fold models, full universe, with top-0.3% file:
    python infer_all_pairs.py \
      --models  ../data/interim/ReLU128_f_a075_g15_10folds_pt10/CV2_811_GIV_NN_LR1e2_50e_p10_d01_{1..10}.pth \
      --scalers ../data/interim/ReLU128_f_a075_g15_10folds_pt10/CV2_811_seed{1..10}.joblib \
      --output  ../data/interim/all_pairs_pred/all_pairs_ensemble10.tsv \
      --mask-cache ../data/interim/all_pairs_pred/screened_mask_keys.npy
"""

from __future__ import annotations

import argparse
import glob
import math
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
DEF_KO = os.path.join(REPO, "data/input/AE_3L/AE_std100_CRISPRGeneEffect_24Q4_imputed_gene_wise_mean_align_qGI2021.txt")
DEF_EXP = os.path.join(REPO, "data/input/AE_3L/AE_std100_Expression_BC_24Q4_align_qGI2021.txt")
DEF_SCREEN = os.path.join(REPO, "data/input/GIV_24Q4_3L")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _model_tag(path, i):
    m = re.search(r"_(\d+)\.pth$", os.path.basename(path))
    return f"fold{m.group(1)}" if m else f"model{i + 1}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", help="one or more model .pth files")
    p.add_argument("--scalers", nargs="+", help="matching scaler .joblib files (same order/length)")
    p.add_argument("--model", help="legacy single-model alias")
    p.add_argument("--scaler", help="legacy single-scaler alias")
    p.add_argument("--output", required=True)
    p.add_argument("--top-output", default=None, help="path for the top-fraction file (auto if unset)")
    p.add_argument("--top-frac", type=float, default=0.003, help="fraction of top-ranked pairs to also save")
    p.add_argument("--ae-ko", default=DEF_KO)
    p.add_argument("--ae-exp", default=DEF_EXP)
    p.add_argument("--screened-dir", default=DEF_SCREEN)
    p.add_argument("--screened-glob", default="*.tsv")
    p.add_argument("--mask-cache", default=None, help=".npy cache of screened-pair keys (built+saved if missing)")
    p.add_argument("--batch-pairs", type=int, default=4_000_000)
    p.add_argument("--max-query-genes", type=int, default=None)
    p.add_argument("--score-decimals", type=int, default=6)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    models = list(args.models) if args.models else ([args.model] if args.model else None)
    scalers = list(args.scalers) if args.scalers else ([args.scaler] if args.scaler else None)
    if not models or not scalers or len(models) != len(scalers):
        p.error("Provide matching --models and --scalers (or single --model/--scaler).")
    M = len(models)
    is_ensemble = M > 1
    dec = args.score_decimals
    fmt = f"%.{dec}f"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    timings = {}
    t_all = time.time()

    # ---- AE embeddings -> centered per-gene tensors ----
    t = time.time()
    df_ko = pd.read_csv(args.ae_ko, sep="\t", index_col=0)
    df_exp = pd.read_csv(args.ae_exp, sep="\t", index_col=0)
    genes = np.intersect1d(df_ko.index, df_exp.index)
    gene_idx = {g: i for i, g in enumerate(genes)}
    N = len(genes)
    Cko = df_ko.loc[genes].values.astype(np.float32)
    Cexp = df_exp.loc[genes].values.astype(np.float32)
    Cko -= Cko.mean(axis=1, keepdims=True)
    Cexp -= Cexp.mean(axis=1, keepdims=True)
    Cko_t = torch.tensor(Cko, device=device)
    Cexp_t = torch.tensor(Cexp, device=device)
    timings["load_ae_center"] = time.time() - t
    log(f"AE loaded: N={N} genes | device={device} | models={M} ({'ensemble' if is_ensemble else 'single'})")

    # ---- models + per-model scalers ----
    t = time.time()
    nets, means, scales, tags = [], [], [], []
    for i, (mp, sp) in enumerate(zip(models, scalers)):
        net = torch.load(mp, map_location=device, weights_only=False).eval()
        sc = joblib.load(sp)
        nets.append(net)
        means.append(torch.tensor(sc.mean_, dtype=torch.float32, device=device))
        scales.append(torch.tensor(sc.scale_, dtype=torch.float32, device=device))
        tags.append(_model_tag(mp, i))
    timings["load_models"] = time.time() - t
    prob_cols = [f"prob_{tg}" for tg in tags]

    # ---- screened mask (cached) ----
    t = time.time()
    cache = args.mask_cache or os.path.join(os.path.dirname(os.path.abspath(args.output)),
                                            f"screened_mask_keys_N{N}.npy")
    if os.path.exists(cache):
        screened_keys = np.load(cache)
        log(f"Screened mask loaded from cache: {cache} ({len(screened_keys):,} keys)")
    else:
        shards = sorted(glob.glob(os.path.join(args.screened_dir, args.screened_glob)))
        parts, n_raw = [], 0
        for sh in shards:
            gq = pd.read_csv(sh, sep="\t", usecols=["Gene", "Query"], low_memory=False)
            n_raw += len(gq)
            a = gq["Gene"].map(gene_idx).to_numpy()
            b = gq["Query"].map(gene_idx).to_numpy()
            ok = ~(pd.isna(a) | pd.isna(b))
            a = a[ok].astype(np.int64); b = b[ok].astype(np.int64)
            lo = np.minimum(a, b); hi = np.maximum(a, b)
            neq = lo != hi
            parts.append(lo[neq] * N + hi[neq])
        screened_keys = np.unique(np.concatenate(parts)) if parts else np.array([], dtype=np.int64)
        np.save(cache, screened_keys)
        log(f"Screened mask built from {n_raw:,} rows -> {len(screened_keys):,} keys; saved to {cache}")
    timings["mask"] = time.time() - t

    # ---- scoring helper ----
    @torch.no_grad()
    def score_pairs(i_arr, j_arr):
        """Return (n, M) probability matrix for the given index pairs."""
        it = torch.as_tensor(i_arr, device=device)
        jt = torch.as_tensor(j_arr, device=device)
        feat = torch.cat([Cko_t[it] + Cko_t[jt], Cexp_t[it] + Cexp_t[jt]], dim=1)
        out = np.empty((len(i_arr), M), dtype=np.float32)
        for m in range(M):
            fm = (feat - means[m]) / scales[m]
            out[:, m] = torch.sigmoid(nets[m](fm)).cpu().numpy()
        return out

    # candidate count (for preallocating ranking arrays)
    n_query = N if args.max_query_genes is None else min(args.max_query_genes, N)
    cand_total = sum(N - 1 - i for i in range(n_query))
    mean_all = np.empty(cand_total, dtype=np.float32)
    i_all = np.empty(cand_total, dtype=np.int32)
    j_all = np.empty(cand_total, dtype=np.int32)
    off = 0

    # ---- stream + write full output ----
    cols = (["gene1", "gene2"] + prob_cols + ["mean"]) if is_ensemble else ["gene1", "gene2", "score"]
    n_masked = 0
    t_loop = time.time()
    first = True
    buf_i, buf_j, buf_n = [], [], 0

    def flush(i_arr, j_arr, fh):
        nonlocal off, n_masked, first
        keys = i_arr.astype(np.int64) * N + j_arr.astype(np.int64)
        keep = ~_isin_sorted(keys, screened_keys)
        n_masked += int((~keep).sum())
        i_k = i_arr[keep]; j_k = j_arr[keep]
        if len(i_k) == 0:
            return
        probs = score_pairs(i_k, j_k)
        mean = probs.mean(axis=1)
        data = {"gene1": genes[i_k], "gene2": genes[j_k]}
        if is_ensemble:
            for c in range(M):
                data[prob_cols[c]] = probs[:, c]
            data["mean"] = mean
        else:
            data["score"] = probs[:, 0]
        df = pd.DataFrame(data, columns=cols)
        df.to_csv(fh, sep="\t", header=first, index=False, float_format=fmt)
        first = False
        k = len(i_k)
        mean_all[off:off + k] = mean
        i_all[off:off + k] = i_k
        j_all[off:off + k] = j_k
        off += k

    with open(args.output, "w") as fh:
        for i in range(n_query):
            m = N - (i + 1)
            if m <= 0:
                continue
            buf_i.append(np.full(m, i)); buf_j.append(np.arange(i + 1, N)); buf_n += m
            if buf_n >= args.batch_pairs:
                flush(np.concatenate(buf_i), np.concatenate(buf_j), fh)
                buf_i, buf_j, buf_n = [], [], 0
        if buf_n:
            flush(np.concatenate(buf_i), np.concatenate(buf_j), fh)
    timings["full_infer_write"] = time.time() - t_loop
    n_written = off

    # ---- top fraction by mean (ensemble) / score (single) ----
    t = time.time()
    rank = mean_all[:n_written]
    K = max(1, math.ceil(n_written * args.top_frac))
    K = min(K, n_written)
    top_idx = np.argpartition(rank, -K)[-K:]
    top_idx = top_idx[np.argsort(rank[top_idx])[::-1]]   # sort desc by rank
    ti = i_all[:n_written][top_idx]; tj = j_all[:n_written][top_idx]
    top_probs = score_pairs(ti, tj)
    top_mean = top_probs.mean(axis=1)
    top_data = {"gene1": genes[ti], "gene2": genes[tj]}
    if is_ensemble:
        for c in range(M):
            top_data[prob_cols[c]] = top_probs[:, c]
        top_data["mean"] = top_mean
    else:
        top_data["score"] = top_probs[:, 0]
    top_out = args.top_output or args.output.replace(".tsv", f"_top{args.top_frac*100:g}pct.tsv")
    pd.DataFrame(top_data, columns=cols).to_csv(top_out, sep="\t", index=False, float_format=fmt)
    timings["top_fraction"] = time.time() - t
    timings["total"] = time.time() - t_all

    rate = n_written / timings["full_infer_write"] if timings["full_infer_write"] > 0 else 0
    summary = [
        f"models            : {M} ({'ensemble' if is_ensemble else 'single'})",
        "  " + "\n  ".join(f"{tags[i]}: {os.path.basename(models[i])}" for i in range(M)),
        f"output (full)     : {args.output}",
        f"output (top {args.top_frac*100:g}%): {top_out}",
        f"N genes           : {N}",
        f"query genes used  : {n_query}" + ("  (FULL)" if args.max_query_genes is None else "  (SUBSET)"),
        f"candidate pairs   : {cand_total:,}",
        f"masked (screened) : {n_masked:,}",
        f"written pairs     : {n_written:,}",
        f"top-fraction pairs: {K:,}",
        "",
        f"load_ae_center    : {timings['load_ae_center']:.2f} s",
        f"load_models       : {timings['load_models']:.2f} s",
        f"mask              : {timings['mask']:.2f} s",
        f"full_infer_write  : {timings['full_infer_write']:.2f} s",
        f"top_fraction      : {timings['top_fraction']:.2f} s",
        f"TOTAL             : {timings['total']:.2f} s",
        f"throughput        : {rate/1e6:.2f} M written pairs/sec",
    ]
    with open(args.output + ".timing.txt", "w") as fh:
        fh.write("\n".join(summary) + "\n")
    log("===== DONE =====")
    print("\n".join(summary))


def _isin_sorted(keys, sorted_ref):
    if len(sorted_ref) == 0:
        return np.zeros(len(keys), dtype=bool)
    pos = np.searchsorted(sorted_ref, keys)
    pos = np.clip(pos, 0, len(sorted_ref) - 1)
    return sorted_ref[pos] == keys


if __name__ == "__main__":
    main()
