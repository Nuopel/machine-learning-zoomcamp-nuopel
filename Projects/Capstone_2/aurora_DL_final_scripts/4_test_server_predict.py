"""
Test script for Aurora DL Prediction API
=======================================
- Health check + schema
- Single prediction with history window
"""

import json
import numpy as np
import pandas as pd
import requests

from pathlib import Path

BASE_URL = "http://127.0.0.1:7860"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
FEATURES_PATH = DATA_DIR / "processed" / "features.parquet"
TARGETS_PATH = DATA_DIR / "processed" / "targets.parquet"

HORIZON = "24h"
LOOKBACK = 8
SEED = 42


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
        print(f"lookback_steps = {data.get('lookback_steps')}")
    else:
        print(r.text[:400])


def test_single_prediction():
    print("\n🔮 Testing single prediction...")

    features_df = pd.read_parquet(FEATURES_PATH).sort_index()
    targets_df = pd.read_parquet(TARGETS_PATH).sort_index()

    # Pick a random recent night timestamp
    last_year_start = targets_df.index.max() - pd.Timedelta(days=365)
    recent_idx = targets_df.loc[targets_df.index >= last_year_start].index
    night_hours = {21, 22, 23, 0, 1, 2, 3}
    night_idx = [ts for ts in recent_idx if ts.hour in night_hours]

    rng = np.random.default_rng(SEED)
    ts = rng.choice(night_idx)

    # Build history window
    pos = features_df.index.get_loc(ts)
    if pos < LOOKBACK - 1:
        raise ValueError("Not enough history for lookback")

    hist_idx = features_df.index[pos - (LOOKBACK - 1): pos]
    history = features_df.loc[hist_idx].to_dict(orient="records")
    current = features_df.loc[[ts]].to_dict(orient="records")[0]

    payload = dict(current)
    payload["history"] = history

    r = requests.post(f"{BASE_URL}/predict?horizon={HORIZON}", json=payload, timeout=30)
    print(f"Timestamp: {ts}")
    print(f"True label: {int(targets_df.loc[ts, f'target_{HORIZON}'])}")
    print(f"Status: {r.status_code}")

    if r.status_code != 200:
        print("Error:", r.text[:400])
        return

    print(pretty(r.json()))


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Aurora DL API Test Suite")
    print(f"Testing: {BASE_URL}")
    print("=" * 60)

    test_health_check()
    test_schema()
    test_single_prediction()
