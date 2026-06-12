"""
Inspect the 10 saved DeepSLP CV2 models to recover their training configuration.

The models in
    ~/DeepSLP/data/interim/ReLU128_f_a075_g15_10folds_pt10
were saved with `torch.save(model, ...)` (whole pickled module), so we can load
them back and read off the true architecture (layer sizes, dropout, batchnorm),
the learned parameters, and the companion StandardScaler (.joblib) + prediction
tables (.tsv).

Usage:
    python scripts/inspect_saved_models.py
"""

import os
import re
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib


# ---------------------------------------------------------------------------
# Re-declare the classes exactly as defined in notebooks/cv2_tuning.ipynb so
# that the pickled module objects can be un-pickled.
# ---------------------------------------------------------------------------
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size=1):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = self.dropout1(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(torch.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(torch.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x).squeeze(-1)
        return x


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')


MODEL_DIR = os.path.expanduser(
    "~/DeepSLP/data/interim/ReLU128_f_a075_g15_10folds_pt10"
)


def natural_fold(path):
    m = re.search(r"_(\d+)\.pth$", os.path.basename(path))
    return int(m.group(1)) if m else 0


def describe_model(model):
    """Pull out architecture facts from a loaded NeuralNetwork."""
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    dropouts = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Dropout)]
    bns = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.BatchNorm1d)]

    layer_dims = []
    for name, lin in linears:
        layer_dims.append((name, lin.in_features, lin.out_features, lin.bias is not None))

    info = {
        "input_size": linears[0][1].in_features,
        "hidden_sizes": [lin.out_features for _, lin in linears[:-1]],
        "output_size": linears[-1][1].out_features,
        "n_linear_layers": len(linears),
        "dropout_p": sorted({d.p for _, d in dropouts}),
        "n_batchnorm": len(bns),
        "layer_dims": layer_dims,
        "training_mode_flag": model.training,
        "total_params": sum(p.numel() for p in model.parameters()),
    }
    return info


def main():
    pth_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pth")), key=natural_fold)
    print(f"Found {len(pth_files)} .pth files in:\n  {MODEL_DIR}\n")

    all_info = []
    weight_sigs = []
    for pth in pth_files:
        fold = natural_fold(pth)
        # weights_only=False because these are whole pickled modules
        obj = torch.load(pth, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            print(f"[fold {fold}] NOTE: file is a state_dict, not a full model.")
            continue
        model = obj
        model.eval()
        info = describe_model(model)
        info["fold"] = fold
        info["type"] = type(model).__name__
        all_info.append(info)

        # signature of first-layer weights to confirm folds differ
        w = model.fc1.weight.detach().numpy()
        weight_sigs.append((fold, float(w.mean()), float(w.std())))

        print(f"[fold {fold:>2}] type={info['type']:<13} "
              f"input={info['input_size']:>4} "
              f"hidden={info['hidden_sizes']} "
              f"out={info['output_size']} "
              f"dropout_p={info['dropout_p']} "
              f"bn={info['n_batchnorm']} "
              f"params={info['total_params']:,}")

    print("\n--- Architecture consistency across folds ---")
    df = pd.DataFrame(all_info)
    for col in ["input_size", "hidden_sizes", "output_size", "dropout_p", "n_batchnorm", "n_linear_layers"]:
        vals = df[col].astype(str).unique()
        status = "IDENTICAL" if len(vals) == 1 else "DIFFERS"
        print(f"  {col:<16}: {status:<10} {vals}")

    print("\n--- fc1 weight stats per fold (should differ => independently trained) ---")
    for fold, mean, std in weight_sigs:
        print(f"  fold {fold:>2}: mean={mean:+.5f}  std={std:.5f}")

    # Companion scaler files
    print("\n--- StandardScaler (.joblib) companions ---")
    scalers = sorted(glob.glob(os.path.join(MODEL_DIR, "*.joblib")), key=lambda p: natural_fold(p.replace("seed", "")) if "seed" in p else 0)
    for sc_path in sorted(glob.glob(os.path.join(MODEL_DIR, "*.joblib"))):
        try:
            sc = joblib.load(sc_path)
            n_feat = getattr(sc, "n_features_in_", "?")
            print(f"  {os.path.basename(sc_path):<24} n_features_in_={n_feat} "
                  f"mean[:3]={np.round(sc.mean_[:3], 3) if hasattr(sc, 'mean_') else 'NA'}")
        except Exception as e:
            print(f"  {os.path.basename(sc_path)}: failed to load ({e})")

    # Companion prediction tables
    print("\n--- Prediction tables (.tsv) companions ---")
    for tsv in sorted(glob.glob(os.path.join(MODEL_DIR, "*_*.tsv")), key=lambda p: natural_fold(p.replace('.tsv', '.pth'))):
        try:
            t = pd.read_csv(tsv, sep="\t")
            n_pos = int(t["GI_stringent_Type2"].sum()) if "GI_stringent_Type2" in t.columns else "?"
            print(f"  {os.path.basename(tsv):<48} rows={len(t):>5} cols={list(t.columns)[:6]}... "
                  f"test_pos={n_pos}")
        except Exception as e:
            print(f"  {os.path.basename(tsv)}: failed to read ({e})")


if __name__ == "__main__":
    main()
