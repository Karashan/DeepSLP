"""Plan-2 integrated inference: predict SL for ALL unscreened gene pairs.

Given a trained model (.pth) + its StandardScaler (.joblib), this builds GIV
features on the fly (no 160 GB feature store) and scores every unique unordered
gene pair C(N,2) that is NOT already in the screened 4M set.

GIV feature for a pair (g1, g2) is additive per gene:
    ko_*  = center(AE_ko[g1])  + center(AE_ko[g2])
    exp_* = center(AE_exp[g1]) + center(AE_exp[g2])      center(v) = v - mean(v)
so we precompute each gene's centered embedding once and just gather + add.

Output: a TSV with columns gene1, gene2, score (predicted SL probability).
Screened pairs (read from the GIV_24Q4 shards) are masked out.

Timing for every stage is recorded and written to <output>.timing.txt.

Example (small test, first model, 50 query genes):
    python infer_all_pairs.py \
        --model  ../data/interim/ReLU128_f_a075_g15_10folds_pt10/CV2_811_GIV_NN_LR1e2_50e_p10_d01_1.pth \
        --scaler ../data/interim/ReLU128_f_a075_g15_10folds_pt10/CV2_811_seed1.joblib \
        --output ../data/interim/all_pairs_pred/test_fold1.tsv \
        --max-query-genes 50
"""

from __future__ import annotations

import argparse
import glob
import os
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--scaler", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--ae-ko", default=DEF_KO)
    p.add_argument("--ae-exp", default=DEF_EXP)
    p.add_argument("--screened-dir", default=DEF_SCREEN,
                   help="dir of GIV TSV shards used to build the screened-pair mask")
    p.add_argument("--screened-glob", default="*.tsv")
    p.add_argument("--batch-pairs", type=int, default=4_000_000)
    p.add_argument("--max-query-genes", type=int, default=None,
                   help="limit number of lower-index query genes (for quick tests)")
    p.add_argument("--score-decimals", type=int, default=6)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    timings = {}
    t_all = time.time()

    # ---- Load AE embeddings, build centered per-gene tensors ----
    t = time.time()
    df_ko = pd.read_csv(args.ae_ko, sep="\t", index_col=0)
    df_exp = pd.read_csv(args.ae_exp, sep="\t", index_col=0)
    genes = np.intersect1d(df_ko.index, df_exp.index)          # sorted unique
    gene_idx = {g: i for i, g in enumerate(genes)}
    N = len(genes)
    Cko = df_ko.loc[genes].values.astype(np.float32)
    Cexp = df_exp.loc[genes].values.astype(np.float32)
    Cko -= Cko.mean(axis=1, keepdims=True)
    Cexp -= Cexp.mean(axis=1, keepdims=True)
    Cko_t = torch.tensor(Cko, device=device)
    Cexp_t = torch.tensor(Cexp, device=device)
    timings["load_ae_center"] = time.time() - t
    log(f"AE loaded: N={N} genes | device={device}")

    # ---- Load model + scaler ----
    t = time.time()
    model = torch.load(args.model, map_location=device, weights_only=False).eval()
    scaler = joblib.load(args.scaler)
    mean_t = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    scale_t = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)
    timings["load_model_scaler"] = time.time() - t

    # ---- Build screened-pair mask (canonical key = a*N + b, a<b by gene index) ----
    t = time.time()
    shards = sorted(glob.glob(os.path.join(args.screened_dir, args.screened_glob)))
    screened_keys = []
    n_screened_raw = 0
    for sh in shards:
        gq = pd.read_csv(sh, sep="\t", usecols=["Gene", "Query"], low_memory=False)
        n_screened_raw += len(gq)
        a = gq["Gene"].map(gene_idx).to_numpy()
        b = gq["Query"].map(gene_idx).to_numpy()
        ok = ~(pd.isna(a) | pd.isna(b))
        a = a[ok].astype(np.int64)
        b = b[ok].astype(np.int64)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        neq = lo != hi
        screened_keys.append(lo[neq] * N + hi[neq])
    screened_keys = np.unique(np.concatenate(screened_keys)) if screened_keys else np.array([], dtype=np.int64)
    timings["build_mask"] = time.time() - t
    log(f"Screened mask: {n_screened_raw:,} rows -> {len(screened_keys):,} unique in-universe pairs")

    # ---- Stream all i<j pairs in batches, masking screened ----
    n_query = N if args.max_query_genes is None else min(args.max_query_genes, N)
    header = "gene1\tgene2\tscore\n"
    timing_path = args.output + ".timing.txt"

    buf_i, buf_j, buf_n = [], [], 0
    n_written = 0
    n_candidate = 0
    n_masked = 0
    t_loop = time.time()

    @torch.no_grad()
    def flush(i_arr, j_arr, fh):
        nonlocal n_written, n_masked
        keys = i_arr.astype(np.int64) * N + j_arr.astype(np.int64)
        keep = ~_isin_sorted(keys, screened_keys)
        n_masked += int((~keep).sum())
        i_k = i_arr[keep]; j_k = j_arr[keep]
        if len(i_k) == 0:
            return
        it = torch.as_tensor(i_k, device=device)
        jt = torch.as_tensor(j_k, device=device)
        feat = torch.cat([Cko_t[it] + Cko_t[jt], Cexp_t[it] + Cexp_t[jt]], dim=1)
        feat = (feat - mean_t) / scale_t
        prob = torch.sigmoid(model(feat)).cpu().numpy()
        g1 = genes[i_k]; g2 = genes[j_k]
        fmt = "%." + str(args.score_decimals) + "f"
        lines = [f"{a}\t{b}\t{fmt % s}" for a, b, s in zip(g1, g2, prob)]
        fh.write("\n".join(lines))
        fh.write("\n")
        n_written += len(i_k)

    with open(args.output, "w") as fh:
        fh.write(header)
        for i in range(n_query):
            m = N - (i + 1)
            if m <= 0:
                continue
            j = np.arange(i + 1, N)
            ii = np.full(m, i)
            buf_i.append(ii); buf_j.append(j); buf_n += m
            n_candidate += m
            if buf_n >= args.batch_pairs:
                flush(np.concatenate(buf_i), np.concatenate(buf_j), fh)
                buf_i, buf_j, buf_n = [], [], 0
        if buf_n:
            flush(np.concatenate(buf_i), np.concatenate(buf_j), fh)
    timings["inference_and_write"] = time.time() - t_loop
    timings["total"] = time.time() - t_all

    rate = n_candidate / timings["inference_and_write"] if timings["inference_and_write"] > 0 else 0
    summary = [
        f"model            : {args.model}",
        f"output           : {args.output}",
        f"N genes          : {N}",
        f"query genes used : {n_query}" + ("  (FULL)" if args.max_query_genes is None else "  (TEST SUBSET)"),
        f"candidate pairs  : {n_candidate:,}",
        f"masked (screened): {n_masked:,}",
        f"written pairs    : {n_written:,}",
        "",
        f"load_ae_center      : {timings['load_ae_center']:.2f} s",
        f"load_model_scaler   : {timings['load_model_scaler']:.2f} s",
        f"build_mask          : {timings['build_mask']:.2f} s",
        f"inference_and_write : {timings['inference_and_write']:.2f} s",
        f"TOTAL               : {timings['total']:.2f} s",
        f"throughput          : {rate/1e6:.2f} M candidate pairs/sec",
        "",
        f"Extrapolated FULL run (all {N} query genes, ~{N*(N-1)//2:,} pairs):",
        f"  est inference+write : {(N*(N-1)//2)/rate/60:.1f} min" if rate > 0 else "  n/a",
    ]
    with open(timing_path, "w") as fh:
        fh.write("\n".join(summary) + "\n")
    log("===== DONE =====")
    print("\n".join(summary))
    log(f"Predictions -> {args.output}")
    log(f"Timing      -> {timing_path}")


def _isin_sorted(keys, sorted_ref):
    """Vectorized membership test against a sorted reference array."""
    if len(sorted_ref) == 0:
        return np.zeros(len(keys), dtype=bool)
    pos = np.searchsorted(sorted_ref, keys)
    pos = np.clip(pos, 0, len(sorted_ref) - 1)
    return sorted_ref[pos] == keys


if __name__ == "__main__":
    main()
