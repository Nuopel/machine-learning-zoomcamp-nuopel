#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2_predict.py
===========
Offline prediction for Aurora DL models.
- Loads best model type from metadata
- Uses random recent night samples (last year, 21:00–03:00)
- Applies imputer+scaler, builds sequences, prints probs vs true labels
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
FEATURES_PATH = DATA_DIR / "processed" / "features.parquet"
TARGETS_PATH = DATA_DIR / "processed" / "targets.parquet"

MODELS_DIR = Path(__file__).resolve().parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

METADATA_PATH = RESULTS_DIR / "metadata.json"
MEDIANS_PATH = RESULTS_DIR / "train_medians.json"
SCALER_PATH = RESULTS_DIR / "scaler.json"

# ============================================================
# CONFIG
# ============================================================
HORIZON = "24h"
N_EXAMPLES = 5
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# MODELS (same as train)
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

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def make_sequences(X, y, lookback):
    X_seq, y_seq = [], []
    for i in range(lookback - 1, len(X)):
        X_seq.append(X[i - lookback + 1 : i + 1])
        y_seq.append(y[i])
    return np.stack(X_seq), np.asarray(y_seq)


def pick_random_recent_night_rows(targets_df, n):
    last_year_start = targets_df.index.max() - pd.Timedelta(days=365)
    recent_idx = targets_df.loc[targets_df.index >= last_year_start].index

    night_hours = {21, 22, 23, 0, 1, 2, 3}
    night_idx = [ts for ts in recent_idx if ts.hour in night_hours]

    if len(night_idx) < n:
        sample_pool = recent_idx
    else:
        sample_pool = night_idx

    rng = np.random.default_rng(SEED)
    idx = rng.choice(sample_pool, size=n, replace=False)
    return pd.DatetimeIndex(idx).sort_values()


def main():
    metadata = load_json(METADATA_PATH)
    best_model_type = metadata.get("best_model_type")
    features = metadata.get("features", [])
    lookback = metadata.get("lookback_steps", 8)

    scaler_meta = load_json(SCALER_PATH)
    mean = np.array(scaler_meta["mean"])
    scale = np.array(scaler_meta["scale"])

    medians = pd.Series(load_json(MEDIANS_PATH))

    # Load data
    features_df = pd.read_parquet(FEATURES_PATH).sort_index()
    targets_df = pd.read_parquet(TARGETS_PATH).sort_index()

    idx = pick_random_recent_night_rows(targets_df, N_EXAMPLES)
    X_raw = features_df.loc[idx]
    y_true = targets_df.loc[idx, f"target_{HORIZON}"]

    # Align + impute
    for c in features:
        if c not in X_raw.columns:
            X_raw[c] = np.nan
    X_raw = X_raw[features]
    X_raw = X_raw.fillna(medians)

    # Scale
    X_scaled = (X_raw.values - mean) / scale

    # Build sequences (need lookback padding from earlier rows)
    # For simplicity: use last (lookback-1) rows preceding each timestamp from full series
    # Create a small window from full scaled dataset
    full_X = features_df.copy()
    for c in features:
        if c not in full_X.columns:
            full_X[c] = np.nan
    full_X = full_X[features].fillna(medians)
    full_X = (full_X.values - mean) / scale

    # Map index to position
    pos_map = {t: i for i, t in enumerate(features_df.index)}

    X_seq_list = []
    for ts in idx:
        i = pos_map[ts]
        if i < lookback - 1:
            raise ValueError("Not enough history for sequence window")
        X_seq_list.append(full_X[i - lookback + 1 : i + 1])

    X_seq = np.stack(X_seq_list)

    # Load model
    n_features = X_seq.shape[-1]
    if best_model_type == "tcn":
        model = TCNClassifier(n_features)
        model.load_state_dict(torch.load(MODELS_DIR / f"tcn_{HORIZON}.pt", map_location=DEVICE))
    else:
        model = LSTMClassifier(n_features)
        model.load_state_dict(torch.load(MODELS_DIR / f"lstm_{HORIZON}.pt", map_location=DEVICE))

    model = model.to(DEVICE).eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_seq).float().to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()

    out = pd.DataFrame({
        "timestamp": idx,
        "y_true": y_true.values,
        "y_pred": probs,
    })

    print(f"Model type: {best_model_type} | Horizon: {HORIZON}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
