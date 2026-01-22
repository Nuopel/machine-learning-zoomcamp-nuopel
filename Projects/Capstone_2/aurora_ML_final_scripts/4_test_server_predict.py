"""
Test script for Aurora Prediction API
=====================================
- Health check + schema
- Single prediction (random recent night examples)
- Compares predicted probability vs true label
"""

import json
import numpy as np
import pandas as pd
import requests

from pathlib import Path

BASE_URL = "http://127.0.0.1:7860"
HORIZON = "48h"

BASE_URL = "https://nuopel-aurora-api.hf.space/"
HORIZON = "24h"
# note due to restriction size model in HF only  "3h","6h","12h","24h" are available

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
FEATURES_PATH = DATA_DIR / "processed" / "features.parquet"
TARGETS_PATH = DATA_DIR / "processed" / "targets.parquet"

N_EXAMPLES = 5
SEED = 21


def pretty(x):
    return json.dumps(x, indent=2)


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
        feats = data.get("expected_features", [])
        print(f"n_features = {len(feats)}")
        print(f"horizons = {data.get('horizons', [])}")
    else:
        print(r.text[:400])


def pick_random_recent_night_rows():
    features_df = pd.read_parquet(FEATURES_PATH).sort_index()
    targets_df = pd.read_parquet(TARGETS_PATH).sort_index()

    last_year_start = targets_df.index.max() - pd.Timedelta(days=365)
    recent_idx = targets_df.loc[targets_df.index >= last_year_start].index

    night_hours = {21, 22, 23, 0, 1, 2, 3}
    night_idx = [ts for ts in recent_idx if ts.hour in night_hours]

    if len(night_idx) < N_EXAMPLES:
        sample_pool = recent_idx
    else:
        sample_pool = night_idx

    rng = np.random.default_rng(SEED)
    idx = rng.choice(sample_pool, size=N_EXAMPLES, replace=False)
    idx = pd.DatetimeIndex(idx).sort_values()

    X = features_df.loc[idx]
    y = targets_df.loc[idx, f"target_{HORIZON}"]

    return idx, X, y


def test_single_predictions():
    print("\n🔮 Testing single predictions...")
    idx, X, y = pick_random_recent_night_rows()

    for ts, row, y_true in zip(idx, X.to_dict(orient="records"), y.values):
        r = requests.post(f"{BASE_URL}/predict?horizon={HORIZON}", json=row, timeout=30)
        print(f"\n{ts} | true={int(y_true)}")
        if r.status_code != 200:
            print("Error:", r.text[:400])
            continue
        res = r.json()
        print(f"pred_prob={res['probability']:.6f} | model={res['model']}")


def test_invalid_input():
    print("\n❌ Testing invalid input handling...")
    r = requests.post(f"{BASE_URL}/predict?horizon={HORIZON}", json={"foo": "bar"}, timeout=30)
    print(f"Status: {r.status_code}")
    print("Response:", r.text[:400])


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Aurora API Test Suite")
    print(f"Testing: {BASE_URL}")
    print("=" * 60)

    test_health_check()
    test_schema()
    test_single_predictions()
    test_invalid_input()
