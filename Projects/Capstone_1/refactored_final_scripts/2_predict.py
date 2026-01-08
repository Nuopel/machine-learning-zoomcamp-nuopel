"""
2_predict.py
============
Standalone prediction script for speaker prices (LOT-003).
Usage: python 2_predict.py
"""

import pickle
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from smoothed_target_encoder import SmoothedTargetEncoder  # needed for pickle.load

# ============================================================
# Paths (match training outputs)
# ============================================================
MODEL_PATH = "./models_log_hybrid/trained/best_model.pkl"
ENCODER_PATH = "./models_log_hybrid/encoders/target_encoder.pkl"
METADATA_PATH = "./results_log_hybrid/metadata.json"
MEDIANS_PATH = "./models_log_hybrid/encoders/train_medians.json"

# ============================================================
# CSV schema from the dataset (for quick local tests)
# ============================================================
CSV_HEADER = [
    "row_id",
    "price",
    "spec_general_marque",
    "spec_general_type_produit",
    "spec_general_reference",
    "spec_informations_impedance_nominale",
    "spec_forme_materiaux_systeme_magnetique",
    "spec_forme_materiaux_forme_facade",
    "spec_parametres_petits_signaux_qts",
    "spec_parametres_petits_signaux_qes",
    "spec_parametres_petits_signaux_qms",
    "spec_forme_materiaux_materiau_saladier",
    "spec_forme_materiaux_materiau_suspension",
    "spec_forme_materiaux_support_bobine",
    "spec_forme_materiaux_fil_bobine",
    "spec_forme_materiaux_materiau_dome",
    "spec_informations_puissance_nominale_w",
    "spec_informations_sensibilite_fabricant_db",
    "spec_parametres_petits_signaux_fs_hz",
    "spec_donnees_poids_kg",
    "spec_parametres_fondamentaux_re_ohm",
    "spec_donnees_xmax_mm",
    "spec_parametres_fondamentaux_mms_gr",
    "spec_donnees_ebp_hz",
    "spec_parametres_petits_signaux_vas_l",
    "spec_donnees_rendement_calcule_pct",
    "spec_parametres_fondamentaux_le_mh",
    "spec_parametres_fondamentaux_sd_cm2",
    "spec_parametres_fondamentaux_bl_t_m",
    "spec_dimensions_diametre_systeme_magnetique_mm",
    "spec_informations_puissance_max_w",
    "spec_dimensions_hauteur_entrefer_mm",
    "spec_dimensions_hauteur_bobinage_mm",
    "spec_donnees_rendement_pct",
]

TARGET_COL = "price"
GROUP_COL = "spec_general_reference"


def _safe_load_pickle(path: str):
    if not path:
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def _safe_load_joblib(path: str):
    if not path:
        return None
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None


def _safe_load_json(path: str):
    if not path:
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def predict_to_euros(model_obj, X_np: np.ndarray, y_is_log: bool = True) -> np.ndarray:
    pred = model_obj.predict(X_np)
    if y_is_log:
        pred = np.clip(pred, -20.0, 20.0)
        pred = np.expm1(pred)
    return np.clip(pred, 0.0, None)


def _coerce_value(v: str):
    v = v.strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return v


def row_to_payload_and_truth(row_csv: str) -> Tuple[Dict[str, Any], Optional[float]]:
    parts = [p.strip() for p in row_csv.split(",")]

    if len(parts) != len(CSV_HEADER):
        raise ValueError(f"Row has {len(parts)} fields but header has {len(CSV_HEADER)} columns")

    row = {k: _coerce_value(v) for k, v in zip(CSV_HEADER, parts)}

    true_price = float(row[TARGET_COL]) if row.get(TARGET_COL) is not None else None

    payload = dict(row)
    payload.pop(TARGET_COL, None)
    payload.pop(GROUP_COL, None)

    return payload, true_price


def preprocess_payloads(
    payloads: List[Dict[str, Any]],
    encoder,
    feature_names: List[str],
    train_medians: Dict[str, float],
) -> np.ndarray:
    if encoder is None:
        raise RuntimeError("Encoder not loaded")
    if not feature_names:
        raise RuntimeError("metadata.json missing 'features'")
    if train_medians is None:
        raise RuntimeError(f"Train medians not loaded from {MEDIANS_PATH}")

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


if __name__ == "__main__":
    print("Speaker price prediction (local)")
    print("=" * 60)

    model = _safe_load_joblib(MODEL_PATH) if MODEL_PATH else None
    encoder = _safe_load_pickle(ENCODER_PATH) if ENCODER_PATH else None
    metadata = _safe_load_json(METADATA_PATH) or {}
    feature_names = metadata.get("features", [])
    train_medians = _safe_load_json(MEDIANS_PATH)

    if model is None:
        raise RuntimeError(f"Model not found: {MODEL_PATH}")
    if encoder is None:
        raise RuntimeError(f"Encoder not found: {ENCODER_PATH}")

    print(f"Model loaded: {MODEL_PATH}")
    print(f"Encoder loaded: {ENCODER_PATH}")
    print(f"Features: {len(feature_names)}")
    print(f"Train medians loaded: {bool(train_medians)}")

    # Sample rows (same schema as dataset)
    rows = [
        "1139,200.0,PHL Audio,Haut parleur a cone,4071NdS-19,8 ohm,Neodymium,Non cylindrique,0.12,0.12,4.99,Aluminium,,,,,500.0,98.5,46.0,6.2,5.0,6.0,69.0,383.0,59.0,4.6,1.29,493.0,28.6,138.0,,12.0,17.5,4.5",
        "165,754.0,B&C Speakers,Haut parleur a cone,18IPAL,2 ohm,Neodymium,Cylindrique,0.14,0.14,4.2,Aluminium,,Fibre de verre,Aluminium,,1700.0,97.0,32.0,17.6,1.3,20.0,330.0,229.0,164.0,3.69,0.65,1210.0,24.5,,,12.0,44.0,3.3",
        "88,238.0,B&C Speakers,Haut parleur a cone,12NW76,4 ohm,Neodymium,Cylindrique,0.15,0.15,3.75,Aluminium,,Fibre de verre,Cuivre,,500.0,98.0,43.0,4.9,3.4,8.0,82.0,287.0,64.5,3.29,1.1,522.0,22.0,,,11.0,19.0,3.2",
    ]

    payloads: List[Dict[str, Any]] = []
    truths: List[Optional[float]] = []
    for r in rows:
        payload, y = row_to_payload_and_truth(r)
        payloads.append(payload)
        truths.append(y)

    X = preprocess_payloads(payloads, encoder, feature_names, train_medians)
    preds = predict_to_euros(model, X, y_is_log=True)

    print("\nPredictions:")
    for i, (pred, truth) in enumerate(zip(preds, truths), 1):
        if truth is None:
            print(f"Speaker {i}: pred=EUR {pred:.2f}")
        else:
            abs_err = abs(pred - truth)
            rel_err = abs_err / max(truth, 1e-8)
            print(
                f"Speaker {i}: true=EUR {truth:.2f} | pred=EUR {pred:.2f} "
                f"| abs=EUR {abs_err:.2f} | rel={100*rel_err:.2f}%"
            )

    print("\n" + "=" * 60)
