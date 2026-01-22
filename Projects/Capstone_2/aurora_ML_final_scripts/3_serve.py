"""
Aurora Activity Prediction API
==============================
FastAPI service for predicting probability of Kp >= 5 at multiple horizons.
Loads the selected best model type and per-horizon models saved by 1_train.py.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import joblib
import uvicorn

# ============================================================
# Paths
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(__file__).resolve().parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

METADATA_PATH = RESULTS_DIR / "metadata.json"
MEDIANS_PATH = RESULTS_DIR / "train_medians.json"

# ============================================================
# Load metadata + models
# ============================================================
if not METADATA_PATH.exists():
    raise RuntimeError(f"Missing metadata: {METADATA_PATH}")

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

BEST_MODEL_TYPE = metadata.get("best_model_type")
FEATURES = metadata.get("features", [])
HORIZONS = metadata.get("horizons", [])

if not FEATURES:
    raise RuntimeError("metadata.json missing 'features'")

# Medians (needed for RF)
train_medians: Optional[Dict[str, float]] = None
if MEDIANS_PATH.exists():
    with open(MEDIANS_PATH, "r") as f:
        train_medians = json.load(f)

# Load per-horizon models
models = {}
for h in HORIZONS:
    if BEST_MODEL_TYPE == "xgboost":
        model_path = MODELS_DIR / f"xgb_{h}.joblib"
    else:
        model_path = MODELS_DIR / f"rf_{h}.joblib"

    if model_path.exists():
        models[h] = joblib.load(model_path)

if not models:
    raise RuntimeError("No models loaded. Run 1_train.py first.")

# ============================================================
# Helpers
# ============================================================

def align_and_impute(payload: Dict[str, Any]) -> np.ndarray:
    df = pd.DataFrame([payload])

    # Ensure all expected columns exist
    for c in FEATURES:
        if c not in df.columns:
            df[c] = np.nan

    df = df[FEATURES]

    # Impute if RF
    if BEST_MODEL_TYPE == "random_forest":
        if train_medians is None:
            raise HTTPException(status_code=503, detail="Train medians missing for RF imputation")
        med = pd.Series(train_medians)
        df = df.fillna(med)

    return df.to_numpy(dtype=float)

# ============================================================
# API
# ============================================================
class AuroraFeatures(BaseModel):
    """Flexible schema for engineered feature inputs."""
    model_config = {"extra": "allow"}

app = FastAPI(
    title="Aurora Activity Prediction API",
    description="Predict probability of Kp >= 5 using frozen L1 features",
    version="1.0.0",
)

@app.get("/")
def root():
    return {
        "status": "healthy",
        "best_model_type": BEST_MODEL_TYPE,
        "n_models_loaded": len(models),
        "horizons": HORIZONS,
        "n_features": len(FEATURES),
    }

@app.get("/schema")
def schema():
    return {
        "expected_features": FEATURES,
        "horizons": HORIZONS,
        "best_model_type": BEST_MODEL_TYPE,
    }

@app.post("/predict")
def predict_one(payload: AuroraFeatures, horizon: str = "24h"):
    if horizon not in models:
        raise HTTPException(status_code=400, detail=f"Horizon not available: {horizon}")

    X = align_and_impute(payload.model_dump())
    model = models[horizon]

    proba = float(model.predict_proba(X)[:, 1][0])
    return {
        "horizon": horizon,
        "probability": proba,
        "model": BEST_MODEL_TYPE,
    }

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    print("Aurora API running at http://0.0.0.0:7860")
    uvicorn.run(app, host="0.0.0.0", port=7860)
