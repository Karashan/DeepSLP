"""Tier-1 experiment driver: interaction features (Step 1) and/or imbalance-aware
loss (Step 2), evaluated against the current ~0.80 external baseline.

Self-contained; does NOT modify or overwrite the existing pipeline. Reuses the
recovered config (CV2 query-holdout, MLP 256->128->64->32->1 shape scaled to the
feature width, Adam lr=1e-2, ReduceLROnPlateau, early stop patience 10 on val
AUPR, 50 epochs). Everything is written to a fresh timestamped run dir.

Knobs:
  --feature-mode {sum, interaction, interaction_only}
        sum             = current additive baseline (256 dims; control)
        interaction     = [sum, product, absdiff] for ko & exp (768 dims)   [Step 1]
        interaction_only= [product, absdiff] only (512 dims)
  --loss {focal, bce_posweight, balanced}
        focal           = FocalLoss(0.75,1.5) — current (global alpha; pos NOT upweighted)
        bce_posweight   = BCEWithLogitsLoss(pos_weight=n_neg/n_pos)          [Step 2]
        balanced        = BCE + WeightedRandomSampler (balanced batches)     [Step 2]

Examples (small-scale):
  # Step 1 smoke (1 seed)
  python reproduce_cv2/train_tier1.py --feature-mode interaction --loss focal --n-seeds 1 --tag expA_smoke
  # Step 1 5-seed
  python reproduce_cv2/train_tier1.py --feature-mode interaction --loss focal --n-seeds 5 --tag expA
  # Step 2 (on top of Step 1)
  python reproduce_cv2/train_tier1.py --feature-mode interaction --loss balanced --n-seeds 5 --tag expB
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from config import TrainConfig, REPO_ROOT  # noqa: E402
from data import filter_unique_pairs_by_lowest_fdr  # noqa: E402
from metrics import compute_metrics, safe_auroc  # noqa: E402
from model import NeuralNetwork, FocalLoss  # noqa: E402
from features_interaction import load_centered_embeddings, build_pair_features  # noqa: E402


META_COLS = ["Gene", "Query", "FDR", "GI_stringent_Type2", "GI_standard_Type2"]


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def load_meta(input_dir, iter_num, tail=".tsv"):
    """Load only metadata/label columns from the GIV shards, dedup ABBA pairs."""
    files = os.listdir(input_dir)
    frames = []
    for i in range(iter_num):
        want = f"_{i}{tail}"
        for f in files:
            if f.endswith(want):
                frames.append(pd.read_csv(os.path.join(input_dir, f), sep="\t",
                                          usecols=META_COLS, low_memory=False))
    df = pd.concat(frames).reset_index(drop=True)
    df = filter_unique_pairs_by_lowest_fdr(df, col1="Gene", col2="Query", col_fdr="FDR")
    df = df[df["Query"] != df["Gene"]].reset_index(drop=True)
    return df


def cv2_split_indices(df, query_col, test_ratio, val_ratio, seed):
    """Row indices for CV2 query-holdout split (mirrors data.split_cv2_query_holdout RNG)."""
    qg = df[query_col].values
    queries = pd.unique(qg)
    n = len(queries)
    test_len = int(n * test_ratio); val_len = int(n * val_ratio)
    np.random.seed(seed)
    test_q = np.random.choice(queries, size=test_len, replace=False)
    train_q = list(set(queries) - set(test_q))
    val_q = np.random.choice(train_q, size=val_len, replace=False)
    train_q = list(set(train_q) - set(val_q))
    in_ = lambda qs: np.where(np.isin(qg, list(qs)))[0]
    return in_(train_q), in_(val_q), in_(test_q)


def make_loss(kind, y_train, device):
    if kind == "focal":
        return FocalLoss(alpha=0.75, gamma=1.5), None
    if kind == "bce_posweight":
        n_pos = float(y_train.sum()); n_neg = float(len(y_train) - n_pos)
        pw = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
        log(f"  pos_weight = {pw.item():.1f}")
        return nn.BCEWithLogitsLoss(pos_weight=pw), None
    if kind == "balanced":
        n_pos = float(y_train.sum()); n_neg = float(len(y_train) - n_pos)
        w_pos = 0.5 / max(n_pos, 1.0); w_neg = 0.5 / max(n_neg, 1.0)
        sample_w = np.where(y_train == 1, w_pos, w_neg).astype(np.float64)
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
        return nn.BCEWithLogitsLoss(), sampler
    raise ValueError(kind)


def eval_aupr(model, loader, device):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            ps.append(torch.sigmoid(model(xb.to(device))).cpu())
            ys.append(yb.float())
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(torch.cat(ys).numpy(), torch.cat(ps).numpy()))


def train_fold(Xtr, ytr, Xva, yva, cfg, loss_kind, device, epochs):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)

    model = NeuralNetwork(Xtr.shape[1], cfg.hidden_size1, cfg.hidden_size2,
                          cfg.hidden_size3, cfg.output_size, dropout_p=cfg.dropout_p).to(device)
    criterion, sampler = make_loss(loss_kind, ytr, device)

    tr_ds = TensorDataset(torch.tensor(Xtr_s, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long))
    va_ds = TensorDataset(torch.tensor(Xva_s, dtype=torch.float32), torch.tensor(yva, dtype=torch.long))
    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size,
                           shuffle=(sampler is None), sampler=sampler, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False)

    opt = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                 factor=cfg.scheduler_factor, patience=cfg.scheduler_patience)
    best_aupr, best_state, no_imp = 0.0, None, 0
    t0 = time.time()
    for ep in range(epochs):
        model.train(); tl = 0.0
        for xb, yb in tr_loader:
            xb = xb.to(device); yb = yb.to(device).float()
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); opt.step()
            tl += loss.item() * xb.size(0)
        tl /= len(tr_loader.dataset)

        model.eval(); vl = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device); yb = yb.to(device).float()
                vl += criterion(model(xb), yb).item() * xb.size(0)
        vl /= len(va_loader.dataset)
        sched.step(vl)
        va = eval_aupr(model, va_loader, device)
        log(f"  epoch {ep+1}/{epochs}: train_loss={tl:.4f} val_loss={vl:.4f} val_AUPR={va:.4f}")
        if va > best_aupr:
            best_aupr, best_state, no_imp = va, model.state_dict(), 0
        else:
            no_imp += 1
        if no_imp >= cfg.early_stop_patience:
            log("  early stop"); break
    if best_state is not None:
        model.load_state_dict(best_state)
    log(f"  trained in {time.time()-t0:.0f}s")
    return model, scaler


def predict(model, scaler, X, device, bs=8192):
    Xs = scaler.transform(X)
    Xt = torch.tensor(Xs, dtype=torch.float32)
    out = np.empty(Xt.shape[0], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(0, Xt.shape[0], bs):
            b = Xt[s:s+bs].to(device)
            out[s:s+b.shape[0]] = torch.sigmoid(model(b)).cpu().numpy()
    return out


def main():
    cfg = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feature-mode", default="interaction", choices=["sum", "interaction", "interaction_only"])
    p.add_argument("--loss", default="focal", choices=["focal", "bce_posweight", "balanced"])
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--start-seed", type=int, default=1843)
    p.add_argument("--epochs", type=int, default=cfg.num_epochs)
    p.add_argument("--input-dir", default=cfg.input_dir)
    p.add_argument("--iter-num", type=int, default=cfg.input_glob_iter)
    p.add_argument("--heldout-dir", default=os.path.join(REPO_ROOT, "data", "input", "GIV_24Q4_heldout") + os.sep)
    p.add_argument("--heldout-iter", type=int, default=11)
    p.add_argument("--tag", default="tier1")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(REPO_ROOT, "data", "interim", f"tier1_{args.tag}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    log(f"=== Tier-1 experiment: feature-mode={args.feature_mode} loss={args.loss} ===")
    log(f"Run dir: {run_dir} | device={device} | seeds {args.start_seed}..{args.start_seed+args.n_seeds-1}")
    json.dump(vars(args) | {"device": device, "run_dir": run_dir},
              open(os.path.join(run_dir, "run_config.json"), "w"), indent=2)

    # ---- Build features once (train + external) ----
    emb = load_centered_embeddings()
    log(f"AE embeddings loaded (dim={emb.dim})")

    log("Loading training metadata + dedup ...")
    df = load_meta(args.input_dir, args.iter_num)
    Xall, valid = build_pair_features(df, emb, mode=args.feature_mode)
    df = df[valid].reset_index(drop=True)
    log(f"Train pairs (valid): {df.shape[0]} | feature dim={Xall.shape[1]} | "
        f"stringent pos={int(df['GI_stringent_Type2'].sum())}")

    log("Loading external held-out metadata + building features ...")
    df_ext = load_meta(args.heldout_dir, args.heldout_iter)
    Xext, vext = build_pair_features(df_ext, emb, mode=args.feature_mode)
    df_ext = df_ext[vext].reset_index(drop=True)
    y_ext_str = df_ext["GI_stringent_Type2"].values.astype(int)
    y_ext_std = df_ext["GI_standard_Type2"].values.astype(int)
    log(f"External pairs (valid): {df_ext.shape[0]} | stringent pos={int(y_ext_str.sum())}")

    y_all = df[cfg.label_col].values.astype(int)
    internal_rows, ext_rows = [], []

    for k in range(args.n_seeds):
        seed = args.start_seed + k
        sd = os.path.join(run_dir, f"seed_{seed}"); os.makedirs(sd, exist_ok=True)
        log(f"===== seed {seed} ({k+1}/{args.n_seeds}) =====")
        tr, va, te = cv2_split_indices(df, cfg.query_col, cfg.test_ratio, cfg.val_ratio, seed)
        log(f"  split: train={len(tr)} val={len(va)} test={len(te)} | "
            f"test pos={int(y_all[te].sum())}")

        model, scaler = train_fold(Xall[tr], y_all[tr], Xall[va], y_all[va],
                                   cfg, args.loss, device, args.epochs)
        torch.save(model, os.path.join(sd, "model.pth"))
        joblib.dump(scaler, os.path.join(sd, "scaler.joblib"))

        # internal test
        te_scores = predict(model, scaler, Xall[te], device)
        tm = compute_metrics(y_all[te], te_scores, k=cfg.topk)
        tm["auroc_sklearn"] = safe_auroc(y_all[te], te_scores)
        tm["n_test_pairs"] = int(len(te)); tm["n_test_pos"] = int(y_all[te].sum())
        internal_rows.append({"seed": seed, **tm})
        json.dump(tm, open(os.path.join(sd, "test_metrics.json"), "w"), indent=2)
        log(f"  INTERNAL test: ROC={tm['roc_auc']:.4f} AP={tm['average_precision']:.4f} "
            f"P@{cfg.topk}={tm[f'precision_at_{cfg.topk}']:.4f}")

        # external
        ext_scores = predict(model, scaler, Xext, device)
        seed_ext = {}
        for lc, y in [("GI_stringent_Type2", y_ext_str), ("GI_standard_Type2", y_ext_std)]:
            em = compute_metrics(y, ext_scores, k=cfg.topk)
            em["auroc_sklearn"] = safe_auroc(y, ext_scores); em["n_positives"] = int(y.sum())
            seed_ext[lc] = em
            ext_rows.append({"seed": seed, "label": lc, **em})
            log(f"  EXTERNAL [{lc}]: ROC={em['roc_auc']:.4f} AP={em['average_precision']:.4f} "
                f"P@{cfg.topk}={em[f'precision_at_{cfg.topk}']:.4f}")
        json.dump(seed_ext, open(os.path.join(sd, "heldout_metrics.json"), "w"), indent=2)

        if k == 0:  # save first-seed predictions for inspection
            dt = df.iloc[te][["Gene", "Query", cfg.label_col]].copy(); dt["predict_proba"] = te_scores
            dt.to_csv(os.path.join(sd, "test_predictions.tsv"), sep="\t", index=False)
            de = df_ext[["Gene", "Query", "GI_stringent_Type2", "GI_standard_Type2"]].copy()
            de["predict_proba"] = ext_scores
            de.to_csv(os.path.join(sd, "heldout_predictions.tsv"), sep="\t", index=False)

    # ---- aggregate ----
    di = pd.DataFrame(internal_rows); de = pd.DataFrame(ext_rows)
    di.to_csv(os.path.join(run_dir, "internal_test_per_seed.tsv"), sep="\t", index=False)
    de.to_csv(os.path.join(run_dir, "external_per_seed.tsv"), sep="\t", index=False)
    mcols = ["roc_auc", "average_precision", f"precision_at_{cfg.topk}", f"recall_at_{cfg.topk}"]
    summary = {"feature_mode": args.feature_mode, "loss": args.loss, "n_seeds": args.n_seeds,
               "feature_dim": int(Xall.shape[1])}
    summary["internal_test"] = {m: {"mean": float(di[m].mean()), "std": float(di[m].std())} for m in mcols}
    summary["external"] = {}
    for lc in ["GI_stringent_Type2", "GI_standard_Type2"]:
        sub = de[de["label"] == lc]
        summary["external"][lc] = {m: {"mean": float(sub[m].mean()), "std": float(sub[m].std())} for m in mcols}
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)

    log("================= SUMMARY (mean +/- std) =================")
    log(f"feature-mode={args.feature_mode} loss={args.loss} dim={Xall.shape[1]} seeds={args.n_seeds}")
    for m in mcols:
        log(f"  internal {m:<18} {di[m].mean():.4f} +/- {di[m].std():.4f}")
    for lc in ["GI_stringent_Type2", "GI_standard_Type2"]:
        sub = de[de["label"] == lc]
        for m in mcols:
            log(f"  external[{lc}] {m:<18} {sub[m].mean():.4f} +/- {sub[m].std():.4f}")
    log(f"DONE -> {run_dir}")


if __name__ == "__main__":
    main()
