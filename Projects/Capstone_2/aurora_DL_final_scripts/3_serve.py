"""
Aurora DL Prediction API
========================
FastAPI service for predicting probability of Kp >= 5 at multiple horizons
using the best DL model type selected during training (LSTM or TCN).
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import uvicorn

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"

MODELS_DIR = Path(__file__).resolve().parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

METADATA_PATH = RESULTS_DIR / "metadata.json"
MEDIANS_PATH = RESULTS_DIR / "train_medians.json"
SCALER_PATH = RESULTS_DIR / "scaler.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# MODELS
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
# Load metadata + preprocessing
# ============================================================
if not METADATA_PATH.exists():
    raise RuntimeError(f"Missing metadata: {METADATA_PATH}")

metadata = json.loads(METADATA_PATH.read_text())
BEST_MODEL_TYPE = metadata.get("best_model_type")
FEATURES = metadata.get("features", [])
HORIZONS = metadata.get("horizons", [])
LOOKBACK = metadata.get("lookback_steps", 8)

if not FEATURES:
    raise RuntimeError("metadata.json missing 'features'")

medians = pd.Series(json.loads(MEDIANS_PATH.read_text()))
scaler_meta = json.loads(SCALER_PATH.read_text())
MEAN = np.array(scaler_meta["mean"])
SCALE = np.array(scaler_meta["scale"])

# Load models per horizon
models: Dict[str, nn.Module] = {}
for h in HORIZONS:
    if BEST_MODEL_TYPE == "tcn":
        model = TCNClassifier(len(FEATURES))
        model.load_state_dict(torch.load(MODELS_DIR / f"tcn_{h}.pt", map_location=DEVICE))
    else:
        model = LSTMClassifier(len(FEATURES))
        model.load_state_dict(torch.load(MODELS_DIR / f"lstm_{h}.pt", map_location=DEVICE))
    model = model.to(DEVICE).eval()
    models[h] = model

# ============================================================
# Helpers
# ============================================================

def preprocess_sequence(payload: Dict[str, Any], history: np.ndarray) -> np.ndarray:
    """
    Build a sequence of length LOOKBACK using the provided history array
    plus the current payload as the last step.
    history: shape (LOOKBACK-1, n_features)
    """
    df = pd.DataFrame([payload])
    for c in FEATURES:
        if c not in df.columns:
            df[c] = np.nan
    df = df[FEATURES].fillna(medians)

    x = df.to_numpy(dtype=float)
    x = (x - MEAN) / SCALE

    seq = np.vstack([history, x])
    return seq

# ============================================================
# API
# ============================================================
class AuroraFeatures(BaseModel):
    model_config = {"extra": "allow"}

app = FastAPI(
    title="Aurora DL Prediction API",
    description="Predict probability of Kp >= 5 using DL sequences",
    version="1.0.0",
)

@app.get("/")
def root():
    return {
        "status": "healthy",
        "best_model_type": BEST_MODEL_TYPE,
        "horizons": HORIZONS,
        "lookback_steps": LOOKBACK,
        "n_features": len(FEATURES),
    }

@app.get("/schema")
def schema():
    return {
        "expected_features": FEATURES,
        "horizons": HORIZONS,
        "lookback_steps": LOOKBACK,
    }

@app.post("/predict")
def predict_one(payload: AuroraFeatures, horizon: str = "24h"):
    if horizon not in models:
        raise HTTPException(status_code=400, detail=f"Horizon not available: {horizon}")

    # For API simplicity, require caller to provide a short history window
    # The client must pass a list of LOOKBACK-1 previous feature dicts.
    data = payload.model_dump()
    history = data.pop("history", None)
    if history is None:
        raise HTTPException(status_code=400, detail="Missing 'history' (list of LOOKBACK-1 past feature rows)")
    if len(history) != LOOKBACK - 1:
        raise HTTPException(status_code=400, detail=f"history must have length {LOOKBACK-1}")

    hist_df = pd.DataFrame(history)
    for c in FEATURES:
        if c not in hist_df.columns:
            hist_df[c] = np.nan
    hist_df = hist_df[FEATURES].fillna(medians)
    hist_x = hist_df.to_numpy(dtype=float)
    hist_x = (hist_x - MEAN) / SCALE

    seq = preprocess_sequence(data, hist_x)
    X = torch.from_numpy(seq[None, :, :]).float().to(DEVICE)

    model = models[horizon]
    with torch.no_grad():
        prob = torch.sigmoid(model(X)).cpu().numpy()[0]

    return {"horizon": horizon, "probability": float(prob), "model": BEST_MODEL_TYPE}

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    print("Aurora DL API running at http://0.0.0.0:7860")
    uvicorn.run(app, host="0.0.0.0", port=7860)
