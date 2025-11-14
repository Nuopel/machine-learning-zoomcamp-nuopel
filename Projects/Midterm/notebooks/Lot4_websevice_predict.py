"""
Wine Rating Prediction API
===========================
FastAPI service for predicting wine ratings using trained models
and an advanced target encoder for regional features.
"""

import pickle
import numpy as np
import pandas as pd
from typing import List
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
import uvicorn
import os
import joblib


from target_encoder import TargetEncoder  # must be imported before unpickling

# ============================================================
# ✅ Load trained model and encoder
# ============================================================
# using LinearRegression_te to avoid Xboost and randomforest 1G image for demo
try:
    with open('./models/trained/LinearRegression_te.pkl', 'rb') as f_in:
        model = joblib.load(f_in)

    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("⚠️ Warning: best_model_te.pkl not found.")
    model = None

try:
    with open('./models/encoders/target_encoder.pkl', 'rb') as f_enc:
        encoder = pickle.load(f_enc)
    print("✅ Target encoder loaded successfully")
except FileNotFoundError:
    print("⚠️ Warning: target_encoder.pkl not found.")
    encoder = None


# ============================================================
# 🧩 Input & Output Models
# ============================================================

class WineFeatures(BaseModel):
    """Input features for wine rating prediction"""
    vintage_year: float = Field(..., description="Year of the wine vintage")
    structure_acidity: float = Field(..., ge=0.0, le=5.0, description="Acidity level (0-5)")
    structure_tannin: float = Field(..., ge=0.0, le=5.0, description="Tannin level (0-5)")
    region: str = Field(..., description="Wine region name")


class PredictResponse(BaseModel):
    """Response model for wine rating prediction"""
    predicted_rating: float = Field(..., description="Predicted wine rating")
    rating_class: str = Field(..., description="Rating category (Excellent/Very Good/Good/Average)")


# ============================================================
# 🚀 FastAPI App
# ============================================================

app = FastAPI(
    title="Wine Rating Prediction API",
    description="Predict wine ratings based on structure, vintage, and encoded regional features",
    version="1.1.0"
)


# ============================================================
# 🔧 Helper functions
# ============================================================

def classify_rating(rating: float) -> str:
    """Classify wine rating into categories"""
    if rating >= 4.5:
        return "Excellent"
    elif rating >= 4.0:
        return "Very Good"
    elif rating >= 3.5:
        return "Good"
    else:
        return "Average"


def preprocess_input(wine_data: dict) -> np.ndarray:
    """
    Preprocess a single wine input:
    - Apply target encoder to region
    - Concatenate encoded regional features with numeric features
    """
    if encoder is None:
        raise HTTPException(status_code=503, detail="Encoder not loaded")

    df = pd.DataFrame([wine_data])
    numerical = ['vintage_year', 'structure_acidity', 'structure_tannin']

    X_encoded = encoder.transform(df, numerical)

    # Combine numerical and encoded columns
    feature_order = [
        'vintage_year', 'structure_acidity', 'structure_tannin',
        'region_mean_smoothed', 'region_median', 'region_count_log', 'region_std'
    ]

    return X_encoded[feature_order].to_numpy()


def predict_single(wine_features: dict) -> float:
    """Make prediction for a single wine"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = preprocess_input(wine_features)
    prediction = model.predict(X)[0]
    return float(prediction)


# ============================================================
# 🌍 Endpoints
# ============================================================

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Wine Rating Prediction API",
        "model_loaded": model is not None,
        "encoder_loaded": encoder is not None
    }


@app.post("/predict", response_model=PredictResponse)
def predict(wine: WineFeatures) -> PredictResponse:
    """
    Predict wine rating based on input features.

    Example:
    ```json
    {
        "vintage_year": 2018,
        "structure_acidity": 3.5,
        "structure_tannin": 3.0,
        "region": "Bordeaux"
    }
    ```
    """
    try:
        rating = predict_single(wine.model_dump())
        rating_category = classify_rating(rating)

        return PredictResponse(
            predicted_rating=round(rating, 2),
            rating_class=rating_category
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_batch", response_model=List[PredictResponse])
def predict_batch(wines: List[WineFeatures]) -> List[PredictResponse]:
    """
    Predict ratings for multiple wines at once.
    """
    try:
        predictions = []
        for wine in wines:
            rating = predict_single(wine.model_dump())
            predictions.append(PredictResponse(
                predicted_rating=round(rating, 2),
                rating_class=classify_rating(rating)
            ))
        return predictions

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


# ============================================================
# 🏁 Run server
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🍷 Starting Wine Rating Prediction API")
    print("=" * 60)
    print("\n📍 API running at: http://0.0.0.0:9696")
    print("📖 Docs available at: http://0.0.0.0:9696/docs")
    print("\n" + "=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=9696)
