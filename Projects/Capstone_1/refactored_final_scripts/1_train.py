#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LOT-003 (LOG + HYBRID TE): Speaker Price Modelling
=================================================
- Grouped split by spec_general_reference (no leakage)
- Target encoding of categoricals computed on TRAIN ONLY using price in € (hybrid)
- Model trained on y = log1p(price)
- Metrics reported in € + SMAPE/MAPE + bin-wise errors
"""

import os
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from smoothed_target_encoder import fit_target_encoder, SmoothedTargetEncoder

import xgboost as xgb

# ============================================================================
# CONFIG
# ============================================================================
DATA_PATH = "../Datas/speaker_db_selected_refined.csv"
TARGET_COL = "price"
GROUP_COL = "spec_general_reference"

SEED = 42
TE_K = 20

TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
TEST_FRAC = 0.20

OUT_MODELS_DIR = "./models_log_hybrid/trained"
OUT_ENCODERS_DIR = "./models_log_hybrid/encoders"
OUT_RESULTS_DIR = "./results_log_hybrid"

os.makedirs(OUT_MODELS_DIR, exist_ok=True)
os.makedirs(OUT_ENCODERS_DIR, exist_ok=True)
os.makedirs(OUT_RESULTS_DIR, exist_ok=True)

# Price bins for diagnostics
PRICE_BINS = [0, 80, 200, 400, np.inf]
PRICE_BIN_LABELS = ["0-80", "80-200", "200-400", "400+"]

# ============================================================================
# METRICS
# ============================================================================
def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))

def mape(y_true, y_pred, eps=1e-8):
    # Guard against tiny prices
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_pred - y_true) / denom)))

def evaluate_euros(y_true_eur, y_pred_eur, name="Model"):
    rmse = float(np.sqrt(mean_squared_error(y_true_eur, y_pred_eur)))
    mae = float(mean_absolute_error(y_true_eur, y_pred_eur))
    r2 = float(r2_score(y_true_eur, y_pred_eur))
    s = smape(y_true_eur, y_pred_eur)
    m = mape(y_true_eur, y_pred_eur)

    print(f"\n{name}:")
    print(f"  RMSE (€): {rmse:.2f}")
    print(f"  MAE (€):  {mae:.2f}")
    print(f"  R²:       {r2:.4f}")
    print(f"  SMAPE:    {100*s:.2f}%")
    print(f"  MAPE:     {100*m:.2f}%")

    return {"rmse": rmse, "mae": mae, "r2": r2, "smape": s, "mape": m}

def binwise_report(y_true_eur, y_pred_eur, title="Bin-wise errors"):
    dfb = pd.DataFrame({"y": y_true_eur, "yhat": y_pred_eur})
    dfb["bin"] = pd.cut(dfb["y"], bins=PRICE_BINS, labels=PRICE_BIN_LABELS, right=False)
    rows = []
    for b in PRICE_BIN_LABELS:
        d = dfb[dfb["bin"] == b]
        if len(d) == 0:
            continue
        rows.append({
            "bin": b,
            "n": int(len(d)),
            "mae": float(mean_absolute_error(d["y"], d["yhat"])),
            "rmse": float(np.sqrt(mean_squared_error(d["y"], d["yhat"]))),
        })
    out = pd.DataFrame(rows)
    print(f"\n📦 {title}")
    print(out.to_string(index=False))
    return out


# ============================================================================
# HELPERS
# ============================================================================
def predict_to_euros(model, X, y_is_log=True):
    """
    Predict and return in € space.
    If y_is_log=True, model outputs log1p(price).
    """
    pred = model.predict(X)
    if y_is_log:
        pred = np.expm1(pred)
    # avoid negative price predictions
    pred = np.clip(pred, 0.0, None)
    return pred

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("🔊 LOT-003 (LOG + HYBRID TE): Speaker Price Modelling")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # Load & clean
    # ------------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET_COL, GROUP_COL]).copy()

    # ------------------------------------------------------------------------
    # Grouped split
    # ------------------------------------------------------------------------
    groups = df[GROUP_COL].astype(str)
    unique_groups = groups.unique()

    rng = np.random.default_rng(SEED)
    rng.shuffle(unique_groups)

    n = len(unique_groups)
    g_train = set(unique_groups[: int(TRAIN_FRAC * n)])
    g_val   = set(unique_groups[int(TRAIN_FRAC * n): int((TRAIN_FRAC + VAL_FRAC) * n)])
    g_test  = set(unique_groups[int((TRAIN_FRAC + VAL_FRAC) * n):])

    df_train = df[groups.isin(g_train)].copy()
    df_val   = df[groups.isin(g_val)].copy()
    df_test  = df[groups.isin(g_test)].copy()

    print(f"✅ Split rows: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
    print(f"✅ Split groups: train={len(g_train)}, val={len(g_val)}, test={len(g_test)}")

    # ------------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------------
    y_train_eur = df_train[TARGET_COL].values.astype(float)
    y_val_eur   = df_val[TARGET_COL].values.astype(float)
    y_test_eur  = df_test[TARGET_COL].values.astype(float)

    y_train_log = np.log1p(y_train_eur)
    y_val_log   = np.log1p(y_val_eur)
    y_test_log  = np.log1p(y_test_eur)

    X_train = df_train.drop(columns=[TARGET_COL, GROUP_COL])
    X_val   = df_val.drop(columns=[TARGET_COL, GROUP_COL])
    X_test  = df_test.drop(columns=[TARGET_COL, GROUP_COL])

    # ------------------------------------------------------------------------
    # Target encoding (fit on €)
    # ------------------------------------------------------------------------
    cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
    encoder = fit_target_encoder(X_train, y_train_eur, cat_cols, TE_K)

    with open(f"{OUT_ENCODERS_DIR}/target_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    X_train = encoder.transform(X_train)
    X_val   = encoder.transform(X_val)
    X_test  = encoder.transform(X_test)

    # numeric coercion + median imputation
    for X in [X_train, X_val, X_test]:
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    med = X_train.median()
    med.to_json(f"{OUT_ENCODERS_DIR}/train_medians.json")

    X_train_df = X_train.fillna(med)
    X_val_df   = X_val.fillna(med)
    X_test_df  = X_test.fillna(med)

    feature_names = list(X_train_df.columns)

    X_train_np = X_train_df.values
    X_val_np   = X_val_df.values
    X_test_np  = X_test_df.values

    # ------------------------------------------------------------------------
    # Train models on log target
    # ------------------------------------------------------------------------
    results = []
    models = {}

    # Linear
    lr = LinearRegression()
    lr.fit(X_train_np, y_train_log)
    pred_val = predict_to_euros(lr, X_val_np, y_is_log=True)
    results.append({"model": "Linear", **evaluate_euros(y_val_eur, pred_val)})
    models["Linear"] = lr

    # Ridge
    ridge = GridSearchCV(
        Ridge(),
        {"alpha": np.logspace(-4, 4, 20)},
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    ridge.fit(X_train_np, y_train_log)
    pred_val = predict_to_euros(ridge, X_val_np, y_is_log=True)
    results.append({"model": "Ridge", **evaluate_euros(y_val_eur, pred_val)})
    models["Ridge"] = ridge.best_estimator_

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_train_np, y_train_log)
    pred_val = predict_to_euros(rf, X_val_np, y_is_log=True)
    results.append({"model": "RF", **evaluate_euros(y_val_eur, pred_val)})
    models["RF"] = rf

    # XGBoost
    xgbm = xgb.XGBRegressor(
        n_estimators=800,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=10,
        min_child_weight=5,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )
    xgbm.fit(X_train_np, y_train_log)
    pred_val = predict_to_euros(xgbm, X_val_np, y_is_log=True)
    results.append({"model": "XGB", **evaluate_euros(y_val_eur, pred_val)})
    models["XGB"] = xgbm

    # ------------------------------------------------------------------------
    # Select best
    # ------------------------------------------------------------------------
    res = pd.DataFrame(results).sort_values("rmse")
    print("\n🏆 Validation results (LOG target, HYBRID TE in €):")
    print(res.to_string(index=False))

    best_name = res.iloc[0]["model"]
    best_model = models[best_name]

    # ------------------------------------------------------------------------
    # Final TEST
    # ------------------------------------------------------------------------
    test_pred = predict_to_euros(best_model, X_test_np, y_is_log=True)
    test_metrics = evaluate_euros(y_test_eur, test_pred, f"{best_name} (TEST)")

    # Diagnostics
    bin_df = binwise_report(y_test_eur, test_pred, title=f"{best_name} (TEST)")

        # Save all models
    for name, model in models.items():
        path = os.path.join(OUT_MODELS_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)


    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------
    with open(f"{OUT_MODELS_DIR}/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    res.to_csv(f"{OUT_RESULTS_DIR}/model_comparison.csv", index=False)
    bin_df.to_csv(f"{OUT_RESULTS_DIR}/test_binwise_errors.csv", index=False)

    meta = {
        "target": "log1p(price)",
        "target_encoding": {"type": "smoothed_te", "k": TE_K, "fit_target": "price_eur"},
        "best_model": best_name,
        "val_metrics": res.iloc[0].to_dict(),
        "test_metrics": test_metrics,
        "features": feature_names,
        "n_features": len(feature_names),
        "split_sizes": {"train": len(df_train), "val": len(df_val), "test": len(df_test)},
        "date": datetime.now().isoformat(),
        "seed": SEED,
    }

    with open(f"{OUT_RESULTS_DIR}/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n✅ DONE (log-price model + hybrid TE)")

