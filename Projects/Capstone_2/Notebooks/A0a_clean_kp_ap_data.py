from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
KP_FILE = DATA_DIR / "kp_ap.txt"
OUTPUT_DIR = DATA_DIR / "processed"
PLOT_PATH = OUTPUT_DIR / "kp_3h_overview.png"


def load_kp() -> pd.DataFrame:
    columns = [
        "year",
        "month",
        "day",
        "hour_start",
        "hour_mid",
        "days_start",
        "days_mid",
        "kp",
        "ap",
        "definitive_flag",
    ]
    df = pd.read_csv(
        KP_FILE,
        sep=r"\s+",
        header=None,
        names=columns,
        comment="#",
    )

    hour_float = df["hour_start"]
    hours = hour_float.astype(int)
    minutes = ((hour_float - hours) * 60).round().astype(int)

    df["timestamp_utc"] = pd.to_datetime(
        {
            "year": df["year"],
            "month": df["month"],
            "day": df["day"],
            "hour": hours,
            "minute": minutes,
        },
        utc=True,
    )

    kp_df = df[["timestamp_utc", "kp"]].copy()
    kp_df = kp_df.sort_values("timestamp_utc").reset_index(drop=True)
    return kp_df


def assert_time_grid(kp_df: pd.DataFrame) -> None:
    timestamps = kp_df["timestamp_utc"]
    assert timestamps.is_monotonic_increasing, "timestamp_utc is not strictly monotonic"
    assert timestamps.is_unique, "timestamp_utc has duplicates"

    deltas = timestamps.diff().dropna()
    expected = pd.Timedelta(hours=3)
    assert (deltas == expected).all(), "timestamp_utc spacing is not uniform 3 hours"


def plot_kp(kp_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    series = (
        kp_df.set_index("timestamp_utc")["kp"]
        .resample("1D")
        .mean()
        .dropna()
    )

    plt.figure(figsize=(12, 4))
    plt.plot(series.index, series.values, linewidth=0.8)
    plt.title("Kp index (daily mean from 3-hour cadence)")
    plt.xlabel("Timestamp (UTC)")
    plt.ylabel("Kp")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.show()


def main() -> None:
    kp_df = load_kp()
    assert_time_grid(kp_df)
    plot_kp(kp_df)

    print("Kp rows:", len(kp_df))
    print("Date range:", kp_df["timestamp_utc"].min(), "→", kp_df["timestamp_utc"].max())
    print("Plot saved to:", PLOT_PATH)


if __name__ == "__main__":
    main()
