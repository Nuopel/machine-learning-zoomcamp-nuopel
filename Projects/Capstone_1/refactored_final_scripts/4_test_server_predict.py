"""
Test script for Speaker Price Prediction API (LOT-003)
======================================================
- Health check + schema
- Single prediction + batch prediction
- Compares predicted price (€) against true price from your CSV rows
"""

import requests
import json
import numpy as np

# API endpoint
BASE_URL = "http://127.0.0.1:7860"


def pretty(x):
    return json.dumps(x, indent=2, ensure_ascii=False)


# ✅ Real header from your CSV
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


def _coerce_value(v: str):
    """Convert numeric strings to float, keep categoricals as str, empty->None."""
    v = v.strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return v


def row_to_payload_and_truth(row_csv: str):
    """
    Convert a CSV row string to:
    - payload dict (features only)
    - true price (float)
    """
    parts = [p.strip() for p in row_csv.split(",")]

    if len(parts) != len(CSV_HEADER):
        raise ValueError(f"Row has {len(parts)} fields but header has {len(CSV_HEADER)} columns")

    row = {k: _coerce_value(v) for k, v in zip(CSV_HEADER, parts)}

    true_price = float(row[TARGET_COL]) if row.get(TARGET_COL) is not None else None

    # Build payload: drop target + group (training dropped them)
    payload = dict(row)
    payload.pop(TARGET_COL, None)
    payload.pop(GROUP_COL, None)

    return payload, true_price


def test_health_check():
    print("\n🔍 Testing health check...")
    r = requests.get(f"{BASE_URL}/", timeout=10)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        print(pretty(r.json()))
    else:
        print(r.text[:400])


def test_schema():
    print("\n📐 Testing /schema endpoint...")
    r = requests.get(f"{BASE_URL}/schema", timeout=10)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        feats = data.get("model_features_after_encoding", [])
        print(f"n_features_after_encoding = {len(feats)}")
    else:
        print(r.text[:400])


def test_single_prediction(payload: dict, true_price: float | None = None):
    print("\n🔊 Testing single prediction...")
    r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=30)
    print(f"Status: {r.status_code}")

    if r.status_code != 200:
        print("Error:", r.text[:800])
        return

    res = r.json()
    pred = float(res["predicted_price_eur"])

    print("\n✨ Prediction Result:")
    print(f"   Predicted Price (€): {pred:.2f}")

    if true_price is not None:
        abs_err = abs(pred - true_price)
        rel_err = abs_err / max(true_price, 1e-8)
        print(f"   True Price (€):      {true_price:.2f}")
        print(f"   Abs Error (€):       {abs_err:.2f}")
        print(f"   Rel Error (%):       {100*rel_err:.2f}%")


def test_batch_prediction(payloads: list[dict], true_prices: list[float | None]):
    print("\n🔊🔊 Testing batch prediction...")
    r = requests.post(f"{BASE_URL}/predict_batch", json=payloads, timeout=30)
    print(f"Status: {r.status_code}")

    if r.status_code != 200:
        print("Error:", r.text[:800])
        return

    res = r.json()
    preds = [float(x["predicted_price_eur"]) for x in res]

    print(f"\n✨ Batch Predictions ({len(preds)} speakers):")
    for i, (pred, y) in enumerate(zip(preds, true_prices), 1):
        if y is None:
            print(f"   Speaker {i}: pred=€{pred:.2f}")
            continue
        abs_err = abs(pred - y)
        rel_err = abs_err / max(y, 1e-8)
        print(f"   Speaker {i}: true=€{y:.2f} | pred=€{pred:.2f} | abs=€{abs_err:.2f} | rel={100*rel_err:.2f}%")

    # quick aggregate
    ys = np.array([y for y in true_prices if y is not None], dtype=float)
    ps = np.array([p for p, y in zip(preds, true_prices) if y is not None], dtype=float)
    if len(ys) > 0:
        mae = float(np.mean(np.abs(ps - ys)))
        rmse = float(np.sqrt(np.mean((ps - ys) ** 2)))
        print(f"\n   Summary on provided rows: MAE=€{mae:.2f} | RMSE=€{rmse:.2f}")


def test_invalid_input():
    print("\n❌ Testing invalid input handling...")
    invalid = {"foo": "bar"}  # wrong input on purpose
    r = requests.post(f"{BASE_URL}/predict", json=invalid, timeout=30)
    print(f"Status: {r.status_code}")
    print("Response:", r.text[:400])


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Speaker Price API Test Suite")
    print(f"Testing: {BASE_URL}")
    print("=" * 60)

    # Your pasted example rows
    ROWS = [
        "1139,200.0,PHL Audio,Haut parleur a cone,4071NdS-19,8 ohm,Neodymium,Non cylindrique,0.12,0.12,4.99,Aluminium,,,,,500.0,98.5,46.0,6.2,5.0,6.0,69.0,383.0,59.0,4.6,1.29,493.0,28.6,138.0,,12.0,17.5,4.5",
        "165,754.0,B&C Speakers,Haut parleur a cone,18IPAL,2 ohm,Neodymium,Cylindrique,0.14,0.14,4.2,Aluminium,,Fibre de verre,Aluminium,,1700.0,97.0,32.0,17.6,1.3,20.0,330.0,229.0,164.0,3.69,0.65,1210.0,24.5,,,12.0,44.0,3.3",
        "497,309.0,Eighteen Sound,Haut parleur a cone,12ND610,8 ohm,Neodymium,Cylindrique,0.14,0.15,4.3,,,,,,450.0,102.0,46.0,3.4,5.9,3.5,49.0,307.0,94.4,5.89,1.17,531.0,24.0,,700.0,,,",
        "88,238.0,B&C Speakers,Haut parleur a cone,12NW76,4 ohm,Neodymium,Cylindrique,0.15,0.15,3.75,Aluminium,,Fibre de verre,Cuivre,,500.0,98.0,43.0,4.9,3.4,8.0,82.0,287.0,64.5,3.29,1.1,522.0,22.0,,,11.0,19.0,3.2",
        "945,911.0,LaVoce,Haut parleur a cone,SAN184.50iP,8 ohm,Neodymium,Cylindrique,0.15,0.15,6.06,Aluminium,Tissu,Fibre de verre,CCAW (alu recouvert de cuivre),,1700.0,98.0,34.0,15.1,1.37,19.25,312.9,227.0,145.4,3.66,0.4,1225.0,24.85,165.0,3400.0,13.0,45.0,3.8",
        "89,226.0,B&C Speakers,Haut parleur a cone,12NW76,8 ohm,Neodymium,Cylindrique,0.16,0.17,3.7,Aluminium,,Fibre de verre,Cuivre,,500.0,98.5,40.0,4.9,5.3,8.0,77.0,235.0,76.0,2.75,1.25,522.0,25.5,,,11.0,19.0,2.8",
        "119,456.0,B&C Speakers,Haut parleur a cone,15DS115,4 ohm,Neodymium,Cylindrique,0.16,0.17,4.7,Aluminium,,Fibre de verre,Aluminium,,1600.0,96.0,34.0,11.6,3.2,16.5,273.0,200.0,83.0,1.85,3.2,855.0,33.6,,,14.0,40.0,1.9",
    ]

    try:
        test_health_check()
        test_schema()

        payloads = []
        truths = []
        for r in ROWS:
            payload, y = row_to_payload_and_truth(r)
            payloads.append(payload)
            truths.append(y)

        # single
        test_single_prediction(payloads[0], truths[0])

        # batch
        test_batch_prediction(payloads, truths)

        # invalid
        test_invalid_input()

        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)

    except requests.exceptions.ConnectionError as e:
        print("\n❌ Error: Could not connect to API")
        print(f"Make sure the API is running on {BASE_URL}")
        print(f"Error details: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
