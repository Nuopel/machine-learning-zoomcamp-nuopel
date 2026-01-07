"""
Speaker Price Prediction API (LOT-003: LOG + HYBRID TE)
======================================================
FastAPI service for predicting speaker prices (€) using:
- A trained regression model on y = log1p(price)
- A smoothed target encoder fit on TRAIN ONLY using price in € (hybrid TE)

This API:
- loads ./models_log_hybrid/trained/best_model.pkl
- loads ./models_log_hybrid/encoders/target_encoder.pkl
- expects raw features (including categoricals) as input
- applies TE + numeric coercion + median imputation (train medians saved separately)
- outputs predicted price in euros

IMPORTANT:
- During training, you computed medians from X_train AFTER target encoding.
  To reproduce inference correctly, we need those medians at serving time.
  So: export them once (recommended) or compute from a saved "feature template".
"""

import os
import json
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import joblib
import uvicorn

from smoothed_target_encoder import SmoothedTargetEncoder  # needed for pickle.load
# ============================================================
# ✅ Paths (match your training outputs)
# ============================================================
MODEL_PATH = "./models_log_hybrid/trained/best_model.pkl"
ENCODER_PATH = "./models_log_hybrid/encoders/target_encoder.pkl"
METADATA_PATH = "./results_log_hybrid/metadata.json"

# Strongly recommended: save this during training (see note below)
MEDIANS_PATH = "./models_log_hybrid/encoders/train_medians.json"


# ============================================================
# ✅ Load model / encoder / metadata / medians
# ============================================================
def _safe_load_pickle(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def _safe_load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Speaker model loaded")
except FileNotFoundError:
    print(f"⚠️ Model not found: {MODEL_PATH}")
    model = None

try:
    encoder = _safe_load_pickle(ENCODER_PATH)
    print("✅ Target encoder loaded")
except FileNotFoundError:
    print(f"⚠️ Encoder not found: {ENCODER_PATH}")
    encoder = None

metadata = _safe_load_json(METADATA_PATH) or {}
feature_names: List[str] = metadata.get("features", [])

train_medians: Optional[Dict[str, float]] = _safe_load_json(MEDIANS_PATH)

if train_medians is None:
    # We can still run if there are no missing values, but it's unsafe.
    print(f"⚠️ Train medians not found: {MEDIANS_PATH} (imputation may fail)")


# ============================================================
# 🔧 Helpers (same logic as training)
# ============================================================
def predict_to_euros(model_obj, X_np: np.ndarray, y_is_log: bool = True) -> np.ndarray:
    pred = model_obj.predict(X_np)
    if y_is_log:
        # optional safety clip to avoid exp overflow
        pred = np.clip(pred, -20.0, 20.0)
        pred = np.expm1(pred)
    pred = np.clip(pred, 0.0, None)
    return pred


def preprocess_input(payload: Dict[str, Any]) -> np.ndarray:
    """
    Preprocess ONE sample:
    - create df with one row
    - apply encoder.transform (it expects the raw categoricals)
    - coerce numerics
    - align columns to training feature_names
    - fill missing with train medians
    """
    if encoder is None:
        raise HTTPException(status_code=503, detail="Encoder not loaded")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not feature_names:
        raise HTTPException(status_code=503, detail="metadata.json missing 'features' list")
    if train_medians is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Train medians not loaded. Save them during training to ensure consistent imputation "
                f"(expected at {MEDIANS_PATH})."
            ),
        )

    df = pd.DataFrame([payload])

    # Apply the exact same TE logic as training
    df_te = encoder.transform(df)

    # Coerce numeric
    for c in df_te.columns:
        df_te[c] = pd.to_numeric(df_te[c], errors="coerce")

    # Align to training features (add missing columns as NaN)
    for c in feature_names:
        if c not in df_te.columns:
            df_te[c] = np.nan

    df_te = df_te[feature_names]

    # Fill NA with train medians
    med = pd.Series(train_medians)
    df_te = df_te.fillna(med)

    return df_te.to_numpy(dtype=float)


def preprocess_batch(payloads: List[Dict[str, Any]]) -> np.ndarray:
    """
    Preprocess a batch:
    - df of N rows
    - encode once for speed
    - align + impute
    """
    if encoder is None:
        raise HTTPException(status_code=503, detail="Encoder not loaded")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not feature_names:
        raise HTTPException(status_code=503, detail="metadata.json missing 'features' list")
    if train_medians is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Train medians not loaded. Save them during training to ensure consistent imputation "
                f"(expected at {MEDIANS_PATH})."
            ),
        )

    df = pd.DataFrame(payloads)
    df_te = encoder.transform(df)

    for c in df_te.columns:
        df_te[c] = pd.to_numeric(df_te[c], errors="coerce")

    for c in feature_names:
        if c not in df_te.columns:
            df_te[c] = np.nan
    df_te = df_te[feature_names]

    med = pd.Series(train_medians)
    df_te = df_te.fillna(med)

    return df_te.to_numpy(dtype=float)


# ============================================================
# 🧩 Input & Output Models
# ============================================================
class SpeakerFeatures(BaseModel):
    """
    Flexible schema:
    - we accept ANY extra fields because your dataset has many columns.
    - you must provide at least the columns your trained model expects.
    """
    model_config = {"extra": "allow"}

class PredictResponse(BaseModel):
    predicted_price_eur: float = Field(..., description="Predicted speaker price in euros (€)")


# ============================================================
# 🚀 FastAPI App
# ============================================================
app = FastAPI(
    title="Speaker Price Prediction API",
    description="Predict speaker prices (€) from technical specs and encoded categorical features (LOT-003)",
    version="1.0.0",
)


# ============================================================
# 🌍 Endpoints
# ============================================================
@app.get("/")
def root():
    return {
        "status": "healthy",
        "service": "Speaker Price Prediction API",
        "model_loaded": model is not None,
        "encoder_loaded": encoder is not None,
        "metadata_loaded": bool(feature_names),
        "medians_loaded": train_medians is not None,
        "n_features_expected": len(feature_names),
        "best_model": metadata.get("best_model"),
    }


@app.get("/schema")
def schema():
    """
    Returns the list of expected input columns (raw + post-TE depends),
    but for serving we primarily need raw columns so encoder can do its job.
    Since your encoder drops categoricals and adds __te columns, the reliable
    thing to expose is the FINAL feature list used by the model.
    """
    if not feature_names:
        raise HTTPException(status_code=503, detail="metadata.json missing 'features'")
    return {"model_features_after_encoding": feature_names}


@app.post("/predict", response_model=PredictResponse)
def predict_one(speaker: SpeakerFeatures) -> PredictResponse:
    try:
        X = preprocess_input(speaker.model_dump())
        yhat = float(predict_to_euros(model, X, y_is_log=True)[0])
        return PredictResponse(predicted_price_eur=round(yhat, 2))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_batch", response_model=List[PredictResponse])
def predict_batch(speakers: List[SpeakerFeatures]) -> List[PredictResponse]:
    try:
        payloads = [s.model_dump() for s in speakers]
        X = preprocess_batch(payloads)
        preds = predict_to_euros(model, X, y_is_log=True)
        return [PredictResponse(predicted_price_eur=round(float(p), 2)) for p in preds]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


# ============================================================
# 🏁 Run server
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔊 Starting Speaker Price Prediction API (LOT-003)")
    print("=" * 60)
    print("\n📍 API running at: http://0.0.0.0:7860")
    print("📖 Docs available at: http://0.0.0.0:7860/docs")
    print("\n" + "=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=7860)
