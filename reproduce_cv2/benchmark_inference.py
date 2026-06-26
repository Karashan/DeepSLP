"""Micro-benchmark: on-the-fly GIV feature build + model inference throughput.

Precomputes per-gene centered AE embeddings, then for random gene-pair batches
builds the 256-dim GIV feature on the GPU, applies the StandardScaler, and runs
one trained model. Reports pairs/sec and extrapolates to the full C(N,2) universe.
"""
import os, sys, time, glob, numpy as np, joblib, torch

_PKG = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _PKG)
import model as _m
sys.modules["__main__"].NeuralNetwork = _m.NeuralNetwork
sys.modules["__main__"].FocalLoss = _m.FocalLoss
import pandas as pd

AE = os.path.expanduser("~/DeepSLP/data/input/AE_3L")
KO = os.path.join(AE, "AE_std100_CRISPRGeneEffect_24Q4_imputed_gene_wise_mean_align_qGI2021.txt")
EXP = os.path.join(AE, "AE_std100_Expression_BC_24Q4_align_qGI2021.txt")
ORIG = os.path.expanduser("~/DeepSLP/data/interim/ReLU128_f_a075_g15_10folds_pt10")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

df_ko = pd.read_csv(KO, sep="\t", index_col=0)
df_exp = pd.read_csv(EXP, sep="\t", index_col=0)
overlap = np.intersect1d(df_ko.index, df_exp.index)
N = len(overlap)
Cko = (df_ko.loc[overlap].values - df_ko.loc[overlap].values.mean(axis=1, keepdims=True)).astype(np.float32)
Cexp = (df_exp.loc[overlap].values - df_exp.loc[overlap].values.mean(axis=1, keepdims=True)).astype(np.float32)
Cko_t = torch.tensor(Cko, device=DEV); Cexp_t = torch.tensor(Cexp, device=DEV)
print(f"Device={DEV} | N={N} genes | C(N,2)={N*(N-1)//2:,} pairs")

model = torch.load(os.path.join(ORIG, "CV2_811_GIV_NN_LR1e2_50e_p10_d01_1.pth"), map_location=DEV, weights_only=False).eval()
scaler = joblib.load(os.path.join(ORIG, "CV2_811_seed1.joblib"))
mean_t = torch.tensor(scaler.mean_, dtype=torch.float32, device=DEV)
scale_t = torch.tensor(scaler.scale_, dtype=torch.float32, device=DEV)

@torch.no_grad()
def run_batch(i_idx, j_idx):
    ko = Cko_t[i_idx] + Cko_t[j_idx]
    exp = Cexp_t[i_idx] + Cexp_t[j_idx]
    feat = torch.cat([ko, exp], dim=1)
    feat = (feat - mean_t) / scale_t
    return torch.sigmoid(model(feat))

# warmup
rng = np.random.default_rng(0)
bs = 1_000_000
for _ in range(2):
    i = torch.tensor(rng.integers(0, N, bs), device=DEV)
    j = torch.tensor(rng.integers(0, N, bs), device=DEV)
    _ = run_batch(i, j)
if DEV == "cuda": torch.cuda.synchronize()

# timed: 10M pairs
total = 10_000_000
t0 = time.time()
done = 0
while done < total:
    n = min(bs, total - done)
    i = torch.tensor(rng.integers(0, N, n), device=DEV)
    j = torch.tensor(rng.integers(0, N, n), device=DEV)
    p = run_batch(i, j).cpu().numpy()
    done += n
if DEV == "cuda": torch.cuda.synchronize()
dt = time.time() - t0
rate = total / dt
print(f"\nProcessed {total:,} pairs in {dt:.2f}s  ->  {rate/1e6:.2f} M pairs/sec (incl. GPU->CPU copy)")

full = N * (N - 1) // 2
print(f"Extrapolated full C(N,2)={full:,} pairs:")
print(f"  compute-only (on-the-fly, GPU): {full/rate/60:.1f} min")
print(f"  + writing all {full:,} predictions to TSV (~24 B/row, ~50 MB/s disk): {full*24/50e6/60:.1f} min")
