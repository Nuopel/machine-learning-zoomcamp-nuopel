#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2_predict.py
===========
Offline prediction script for Aurora project.
- Loads trained model + metadata
- Uses recent rows from features/targets as examples
- Prints predicted probabilities vs true labels
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

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

# ============================================================
# CONFIG
# ============================================================
HORIZON = "24h"  #  change if needed "12h", "24h", "48h", "96h"
N_EXAMPLES = 5
SEED = 21

# ============================================================
# HELPERS
# ============================================================

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def align_features(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    for c in feature_names:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feature_names]
    return X


def main():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata: {METADATA_PATH}")

    metadata = load_json(METADATA_PATH)
    best_model_type = metadata.get("best_model_type")
    feature_names = metadata.get("features", [])

    if not feature_names:
        raise RuntimeError("metadata.json missing 'features'")

    model_path = MODELS_DIR / ("xgb_" + HORIZON + ".joblib" if best_model_type == "xgboost" else "rf_" + HORIZON + ".joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load data
    features_df = pd.read_parquet(FEATURES_PATH).sort_index()
    targets_df = pd.read_parquet(TARGETS_PATH).sort_index()

    # Random examples from last year, preferably between 21:00 and 03:00
    last_year_start = targets_df.index.max() - pd.Timedelta(days=365)
    recent_idx = targets_df.loc[targets_df.index >= last_year_start].index

    # Night hours: 21, 22, 23, 0, 1, 2, 3
    night_hours = {21, 22, 23, 0, 1, 2, 3}
    night_idx = [ts for ts in recent_idx if ts.hour in night_hours]

    if len(night_idx) < N_EXAMPLES:
        # fallback to any recent timestamps if not enough night hours
        sample_pool = recent_idx
    else:
        sample_pool = night_idx

    rng = np.random.default_rng(SEED)
    idx = rng.choice(sample_pool, size=N_EXAMPLES, replace=False)
    idx = pd.DatetimeIndex(idx).sort_values()

    X_raw = features_df.loc[idx]
    y_true = targets_df.loc[idx, f"target_{HORIZON}"]

    # Align columns
    X = align_features(X_raw.copy(), feature_names)

    # Optional imputation (required for RF)
    if best_model_type == "random_forest":
        if not MEDIANS_PATH.exists():
            raise FileNotFoundError(f"Missing medians: {MEDIANS_PATH}")
        med = pd.Series(load_json(MEDIANS_PATH))
        X = X.fillna(med)

    # Load model
    model = joblib.load(model_path)
    proba = model.predict_proba(X.values)[:, 1]

    out = pd.DataFrame({
        "timestamp": idx,
        "y_true": y_true.values,
        "y_pred": proba,
    })

    print(f"Model: {best_model_type} | Horizon: {HORIZON}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
