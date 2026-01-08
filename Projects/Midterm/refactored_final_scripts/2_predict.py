"""
predict.py
==========
Standalone prediction script for wine ratings.
Usage: python predict.py
"""

import pickle
import numpy as np
import pandas as pd
import joblib
from target_encoder import TargetEncoder


def load_model_and_encoder():
    """Load trained model and encoder"""
    with open('./models/trained/LinearRegression_te.pkl', 'rb') as f:
        model = joblib.load(f)

    with open('./models/encoders/target_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)

    return model, encoder


def preprocess_input(wine_data: dict, encoder) -> np.ndarray:
    """Preprocess input features"""
    df = pd.DataFrame([wine_data])
    numerical = ['vintage_year', 'structure_acidity', 'structure_tannin']

    X_encoded = encoder.transform(df, numerical)

    feature_order = [
        'vintage_year', 'structure_acidity', 'structure_tannin',
        'region_mean_smoothed', 'region_median', 'region_count_log', 'region_std'
    ]

    return X_encoded[feature_order].to_numpy()


def predict(wine_features: dict, model, encoder) -> float:
    """Make a single prediction"""
    X = preprocess_input(wine_features, encoder)
    prediction = model.predict(X)[0]
    return float(prediction)


def classify_rating(rating: float) -> str:
    """Classify wine rating"""
    if rating >= 4.5:
        return "Excellent"
    elif rating >= 4.0:
        return "Very Good"
    elif rating >= 3.5:
        return "Good"
    else:
        return "Average"


if __name__ == "__main__":
    print("🍷 Wine Rating Prediction Script")
    print("=" * 60)

    # Load model and encoder
    print("\n📦 Loading model and encoder...")
    model, encoder = load_model_and_encoder()
    print("✅ Model loaded successfully")

    # Example predictions
    test_wines = [
        {
            "vintage_year": 2018,
            "structure_acidity": 3.5,
            "structure_tannin": 3.2,
            "region": "vin-de-pays-vignobles-de-france"
        },
        {
            "vintage_year": 2020,
            "structure_acidity": 1.2,
            "structure_tannin": 2.8,
            "region": "haut-medoc"
        },
        {
            "vintage_year": 1983,
            "structure_acidity": 1.2,
            "structure_tannin": 3.8,
            "region": "pauillac"
        }
    ]

    print("\n🔮 Making predictions...")
    for i, wine in enumerate(test_wines, 1):
        rating = predict(wine, model, encoder)
        category = classify_rating(rating)

        print(f"\nWine {i}: {wine['region']} {wine['vintage_year']}")
        print(f"  Predicted Rating: {rating:.2f}")
        print(f"  Category: {category}")

    print("\n" + "=" * 60)