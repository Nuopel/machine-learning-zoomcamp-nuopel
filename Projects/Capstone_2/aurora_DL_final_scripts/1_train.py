#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aurora DL Training
==================
- Loads frozen features/targets
- Walk-forward split (60/20/20)
- Impute + scale (fit on train only)
- Build fixed lookback sequences
- Train LSTM and TCN per horizon with early stopping
- Select best model type by mean ROC-AUC across horizons
- Save best model per horizon (for thgoe selected model type)
- Save scaler/imputer parameters
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.onnx

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
FEATURES_PATH = DATA_DIR / "processed" / "features.parquet"
TARGETS_PATH = DATA_DIR / "processed" / "targets.parquet"

OUT_DIR = Path(__file__).resolve().parent
MODELS_DIR = OUT_DIR / "models"
RESULTS_DIR = OUT_DIR / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
SEED = 42
HORIZONS = ["3h", "6h", "12h", "24h", "48h", "72h", "96h"]

TRAIN_FRAC = 0.60
VAL_FRAC = 0.20

LOOKBACK_STEPS = 8  # 24h window

BATCH_SIZE = 256
MAX_EPOCHS = 30
PATIENCE = 5
LR = 1e-3
DROPOUT = 0.2
HIDDEN = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# MODEL DEFINITIONS
# ============================================================
class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_size=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size-1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size-1)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.dropout(y)
        res = x if self.downsample is None else self.downsample(x)
        if y.size(-1) > res.size(-1):
            y = y[..., -res.size(-1):]
        return y + res


class TCNClassifier(nn.Module):
    def __init__(self, n_features, hidden_size=64, dropout=0.2):
        super().__init__()
        self.block1 = TCNBlock(n_features, hidden_size, dropout=dropout)
        self.block2 = TCNBlock(hidden_size, hidden_size, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.block1(x)
        y = self.block2(y)
        y = y[:, :, -1]
        return self.fc(y).squeeze(-1)

# ============================================================
# HELPERS
# ============================================================

def make_sequences(X, y, lookback):
    X_seq, y_seq = [], []
    for i in range(lookback - 1, len(X)):
        X_seq.append(X[i - lookback + 1 : i + 1])
        y_seq.append(y[i])
    return np.stack(X_seq), np.asarray(y_seq)


def eval_metrics(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float().to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()
    return {
        "roc_auc": roc_auc_score(y, probs),
        "pr_auc": average_precision_score(y, probs),
        "brier": brier_score_loss(y, probs),
    }


def train_model(model, loaders, ckpt_path):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_val = None
    best_state = None
    epochs_no_improve = 0

    for _ in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in loaders["val"]:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses))

        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
            torch.save(best_state, ckpt_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# ============================================================
# MAIN
# ============================================================



if __name__ == "__main__":
    print("=" * 80)
    print("Aurora DL Training (LSTM + TCN)")
    print("=" * 80)

    torch.manual_seed(SEED)

    features_df = pd.read_parquet(FEATURES_PATH).sort_index()
    targets_df = pd.read_parquet(TARGETS_PATH).sort_index()
    features_df = features_df.loc[targets_df.index]

    n = len(targets_df)
    train_end = int(TRAIN_FRAC * n)
    val_end = int((TRAIN_FRAC + VAL_FRAC) * n)

    idx = targets_df.index
    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]

    # Impute + scale
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_raw = features_df.loc[train_idx]
    X_val_raw = features_df.loc[val_idx]
    X_test_raw = features_df.loc[test_idx]

    X_train_imp = imputer.fit_transform(X_train_raw)
    X_val_imp = imputer.transform(X_val_raw)
    X_test_imp = imputer.transform(X_test_raw)

    X_train = scaler.fit_transform(X_train_imp)
    X_val = scaler.transform(X_val_imp)
    X_test = scaler.transform(X_test_imp)

    # Save scaler/imputer params
    medians = pd.Series(imputer.statistics_, index=features_df.columns)
    medians.to_json(RESULTS_DIR / "train_medians.json")

    scaler_meta = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "features": list(features_df.columns),
        "lookback_steps": LOOKBACK_STEPS,
    }
    (RESULTS_DIR / "scaler.json").write_text(json.dumps(scaler_meta, indent=2))

    # Train per horizon
    val_scores = {"lstm": [], "tcn": []}
    best_models = {"lstm": {}, "tcn": {}}

    for h in HORIZONS:
        y_train = targets_df.loc[train_idx, f"target_{h}"].to_numpy()
        y_val = targets_df.loc[val_idx, f"target_{h}"].to_numpy()
        y_test = targets_df.loc[test_idx, f"target_{h}"].to_numpy()

        X_train_seq, y_train_seq = make_sequences(X_train, y_train, LOOKBACK_STEPS)
        X_val_seq, y_val_seq = make_sequences(X_val, y_val, LOOKBACK_STEPS)
        X_test_seq, y_test_seq = make_sequences(X_test, y_test, LOOKBACK_STEPS)

        train_ds = TensorDataset(torch.from_numpy(X_train_seq).float(), torch.from_numpy(y_train_seq).float())
        val_ds = TensorDataset(torch.from_numpy(X_val_seq).float(), torch.from_numpy(y_val_seq).float())

        loaders = {
            "train": DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
            "val": DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False),
        }

        n_features = X_train_seq.shape[-1]

        # LSTM
        lstm = LSTMClassifier(n_features, hidden_size=HIDDEN, dropout=DROPOUT)
        lstm = train_model(lstm, loaders, MODELS_DIR / f"lstm_{h}.pt")
        lstm_val = eval_metrics(lstm, X_val_seq, y_val_seq)
        val_scores["lstm"].append(lstm_val["roc_auc"])
        best_models["lstm"][h] = lstm

        # TCN
        tcn = TCNClassifier(n_features, hidden_size=HIDDEN, dropout=DROPOUT)
        tcn = train_model(tcn, loaders, MODELS_DIR / f"tcn_{h}.pt")
        tcn_val = eval_metrics(tcn, X_val_seq, y_val_seq)
        val_scores["tcn"].append(tcn_val["roc_auc"])
        best_models["tcn"][h] = tcn

        print(f"{h}: LSTM val ROC-AUC={lstm_val['roc_auc']:.4f} | TCN val ROC-AUC={tcn_val['roc_auc']:.4f}")

    # Select best model type by mean ROC-AUC
    lstm_mean = float(np.mean(val_scores["lstm"]))
    tcn_mean = float(np.mean(val_scores["tcn"]))

    best_model_type = "tcn" if tcn_mean >= lstm_mean else "lstm"

    metadata = {
        "best_model_type": best_model_type,
        "horizons": HORIZONS,
        "features": list(features_df.columns),
        "lookback_steps": LOOKBACK_STEPS,
        "train_range": [str(train_idx[0]), str(train_idx[-1])],
        "val_range": [str(val_idx[0]), str(val_idx[-1])],
        "test_range": [str(test_idx[0]), str(test_idx[-1])],
        "lstm_mean_roc_auc": lstm_mean,
        "tcn_mean_roc_auc": tcn_mean,
    }
    (RESULTS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print("\nBest model type:", best_model_type)
    print("Saved:")
    print(f"- Models: {MODELS_DIR}")
    print(f"- Metadata: {RESULTS_DIR / 'metadata.json'}")
    print(f"- Scaler: {RESULTS_DIR / 'scaler.json'}")
    print(f"- Medians: {RESULTS_DIR / 'train_medians.json'}")

    # Export ONNX for all horizons using the best model type
    n_features = len(features_df.columns)
    dummy = torch.randn(1, LOOKBACK_STEPS, n_features).to(DEVICE)
    for h in HORIZONS:
        onnx_path = RESULTS_DIR / f"{best_model_type}_{h}.onnx"
        if best_model_type == "tcn":
            model = TCNClassifier(n_features, hidden_size=HIDDEN, dropout=DROPOUT)
            model.load_state_dict(torch.load(MODELS_DIR / f"tcn_{h}.pt", map_location=DEVICE))
        else:
            model = LSTMClassifier(n_features, hidden_size=HIDDEN, dropout=DROPOUT)
            model.load_state_dict(torch.load(MODELS_DIR / f"lstm_{h}.pt", map_location=DEVICE))
        model.eval()
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["input"],
            output_names=["logit"],
            dynamic_axes={"input": {0: "batch"}},
        )
        print(f"- ONNX: {onnx_path}")

