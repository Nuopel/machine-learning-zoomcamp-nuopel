#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aurora L1 Baseline/Tuned Training
=================================
- Loads frozen features/targets
- Walk-forward split (60/20/20)
- Tunes RF and XGBoost with a small grid (per horizon)
- Selects best model type by mean ROC-AUC across horizons
- Fits best model per horizon on train+val
- Saves models + metadata + (optionally) train medians for RF
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from itertools import product

from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

import joblib
import xgboost as xgb

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

# Use a recent subset of train for tuning (speed)
TUNE_ROWS = 20000

# Small fixed grids
rf_grid = {
    "n_estimators": [100],
    "max_depth": [None, 12],
    "min_samples_leaf": [1, 5],
    "max_features": ["sqrt"],
}

xgb_grid = {
    "n_estimators": [150],
    "max_depth": [3, 5],
    "learning_rate": [0.05],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}

# bigger grid
# rf_grid = {
#     "n_estimators": [100, 200],
#     "max_depth": [8, 12, None],
#     "min_samples_leaf": [1, 2, 5],
#     "max_features": ["sqrt"],
# }
#
# xgb_grid = {
#     "n_estimators": [150, 300],
#     "max_depth": [3, 4, 5],
#     "learning_rate": [0.05, 0.1],
#     "subsample": [0.8],
#     "colsample_bytree": [0.8],
#     "min_child_weight": [1, 5],
# }

# ============================================================
# HELPERS
# ============================================================

def grid_iter(grid):
    keys = list(grid.keys())
    for values in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def score_predictions(y_true, y_proba):
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "brier": brier_score_loss(y_true, y_proba),
    }


def is_better(new, best, eps=1e-6):
    if best is None:
        return True
    if new["roc_auc"] > best["roc_auc"] + eps:
        return True
    if abs(new["roc_auc"] - best["roc_auc"]) <= eps:
        if new["pr_auc"] > best["pr_auc"] + eps:
            return True
        if abs(new["pr_auc"] - best["pr_auc"]) <= eps:
            return new["brier"] < best["brier"]
    return False


def main():
    print("=" * 80)
    print("Aurora L1 Training (RF + XGB) — model selection by mean ROC-AUC")
    print("=" * 80)

    # Load
    features_df = pd.read_parquet(FEATURES_PATH).sort_index()
    targets_df = pd.read_parquet(TARGETS_PATH).sort_index()
    features_df = features_df.loc[targets_df.index]

    if not features_df.index.equals(targets_df.index):
        raise ValueError("Feature/target index mismatch")

    # Split
    n = len(targets_df)
    train_end = int(TRAIN_FRAC * n)
    val_end = int((TRAIN_FRAC + VAL_FRAC) * n)

    idx = targets_df.index
    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]

    print(f"Rows: train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}")
    print(f"Train range: {train_idx[0]} -> {train_idx[-1]}")
    print(f"Val range:   {val_idx[0]} -> {val_idx[-1]}")
    print(f"Test range:  {test_idx[0]} -> {test_idx[-1]}")

    # Tuning subset
    tune_idx = train_idx[-TUNE_ROWS:]
    X_train = features_df.loc[tune_idx]
    X_val = features_df.loc[val_idx]

    tuning = {h: {} for h in HORIZONS}

    for h in HORIZONS:
        y_train = targets_df.loc[tune_idx, f"target_{h}"]
        y_val = targets_df.loc[val_idx, f"target_{h}"]

        # RF (imputed)
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)

        best_rf = None
        best_rf_params = None
        for params in grid_iter(rf_grid):
            rf = RandomForestClassifier(random_state=SEED, n_jobs=1, **params)
            rf.fit(X_train_imp, y_train)
            y_proba = rf.predict_proba(X_val_imp)[:, 1]
            scores = score_predictions(y_val, y_proba)
            if is_better(scores, best_rf):
                best_rf = scores
                best_rf_params = params

        # XGB (no imputation)
        best_xgb = None
        best_xgb_params = None
        for params in grid_iter(xgb_grid):
            xgb_model = xgb.XGBClassifier(
                random_state=SEED,
                eval_metric="logloss",
                n_jobs=1,
                tree_method="hist",
                **params,
            )
            xgb_model.fit(X_train, y_train)
            y_proba = xgb_model.predict_proba(X_val)[:, 1]
            scores = score_predictions(y_val, y_proba)
            if is_better(scores, best_xgb):
                best_xgb = scores
                best_xgb_params = params

        tuning[h]["random_forest"] = {"best_params": best_rf_params, "val_scores": best_rf}
        tuning[h]["xgboost"] = {"best_params": best_xgb_params, "val_scores": best_xgb}

        print(f"{h}: RF {best_rf_params} | XGB {best_xgb_params}")

    # Model selection by mean ROC-AUC across horizons
    rf_mean = np.mean([tuning[h]["random_forest"]["val_scores"]["roc_auc"] for h in HORIZONS])
    xgb_mean = np.mean([tuning[h]["xgboost"]["val_scores"]["roc_auc"] for h in HORIZONS])

    best_model_type = "xgboost" if xgb_mean >= rf_mean else "random_forest"
    print(f"Selected model type: {best_model_type} (mean ROC-AUC RF={rf_mean:.4f}, XGB={xgb_mean:.4f})")

    # Refit best model per horizon on train+val
    trainval_idx = train_idx.union(val_idx)
    X_trainval = features_df.loc[trainval_idx]

    # Save feature list and splits
    metadata = {
        "best_model_type": best_model_type,
        "horizons": HORIZONS,
        "features": list(features_df.columns),
        "train_range": [str(train_idx[0]), str(train_idx[-1])],
        "val_range": [str(val_idx[0]), str(val_idx[-1])],
        "test_range": [str(test_idx[0]), str(test_idx[-1])],
        "rf_mean_roc_auc": float(rf_mean),
        "xgb_mean_roc_auc": float(xgb_mean),
    }

    # Save tuning summary
    (RESULTS_DIR / "tuning_summary.json").write_text(json.dumps(tuning, indent=2))
    (RESULTS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    if best_model_type == "random_forest":
        # Train medians for imputation
        imputer = SimpleImputer(strategy="median")
        X_trainval_imp = imputer.fit_transform(X_trainval)
        medians = pd.Series(imputer.statistics_, index=features_df.columns)
        medians.to_json(RESULTS_DIR / "train_medians.json")

        for h in HORIZONS:
            y_trainval = targets_df.loc[trainval_idx, f"target_{h}"]
            params = tuning[h]["random_forest"]["best_params"]
            model = RandomForestClassifier(random_state=SEED, n_jobs=1, **params)
            model.fit(X_trainval_imp, y_trainval)
            joblib.dump(model, MODELS_DIR / f"rf_{h}.joblib")

    else:
        # XGBoost (no imputation) — save medians for optional use
        medians = X_trainval.median()
        medians.to_json(RESULTS_DIR / "train_medians.json")

        for h in HORIZONS:
            y_trainval = targets_df.loc[trainval_idx, f"target_{h}"]
            params = tuning[h]["xgboost"]["best_params"]
            model = xgb.XGBClassifier(
                random_state=SEED,
                eval_metric="logloss",
                n_jobs=1,
                tree_method="hist",
                **params,
            )
            model.fit(X_trainval, y_trainval)
            joblib.dump(model, MODELS_DIR / f"xgb_{h}.joblib")

    print("\nSaved:")
    print(f"- Models: {MODELS_DIR}")
    print(f"- Metadata: {RESULTS_DIR / 'metadata.json'}")
    print(f"- Medians: {RESULTS_DIR / 'train_medians.json'}")
    print(f"- Tuning summary: {RESULTS_DIR / 'tuning_summary.json'}")


if __name__ == "__main__":
    main()
