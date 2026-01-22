from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
OMNI_FILE = DATA_DIR / "omni2_data_2.lst"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_PARQUET = OUTPUT_DIR / "omni_3h.parquet"
PLOT_PATH = OUTPUT_DIR / "omni_3h_overview.png"


def load_omni() -> pd.DataFrame:
    columns = [
        "year",
        "doy",
        "hour",
        "b_scalar",
        "by_gsm",
        "bz_gsm",
        "sw_temperature",
        "sw_density",
        "sw_speed",
        "kp_index",
        "ap_index",
    ]

    df = pd.read_csv(OMNI_FILE, sep=r"\s+", header=None, names=columns)

    timestamp = pd.to_datetime(
        df["year"].astype(str)
        + df["doy"].astype(int).astype(str).str.zfill(3)
        + df["hour"].astype(int).astype(str).str.zfill(2),
        format="%Y%j%H",
        utc=True,
    )
    df["timestamp_utc"] = timestamp

    missing_thresholds = {
        "b_scalar": 999.0,
        "by_gsm": 999.0,
        "bz_gsm": 999.0,
        "sw_temperature": 9_990_000.0,
        "sw_density": 999.0,
        "sw_speed": 9_999.0,
    }
    for column, threshold in missing_thresholds.items():
        df.loc[df[column] >= threshold, column] = np.nan

    return df.set_index("timestamp_utc").sort_index()


def resample_3h(series: pd.Series, agg: str) -> pd.Series:
    resampled = series.resample("3H", label="right", closed="right")
    aggregated = resampled.agg(agg)
    coverage = resampled.count() / 3.0
    aggregated[coverage < (2 / 3)] = np.nan
    return aggregated


def build_omni_3h(df: pd.DataFrame) -> pd.DataFrame:
    agg_map = {
        "bz_gsm": "min",
        "by_gsm": "mean",
        "b_scalar": "mean",
        "sw_speed": "mean",
        "sw_density": "mean",
        "sw_temperature": "mean",
        "kp_index": "mean",
        "ap_index": "mean",
    }

    parts = {}
    for column, agg in agg_map.items():
        name = f"{column}_3h_{agg}"
        parts[name] = resample_3h(df[column], agg)

    omni_3h = pd.DataFrame(parts).reset_index()
    return omni_3h


def assert_time_grid(omni_3h: pd.DataFrame) -> None:
    timestamps = omni_3h["timestamp_utc"]
    assert timestamps.is_monotonic_increasing, "timestamp_utc is not strictly monotonic"
    assert timestamps.is_unique, "timestamp_utc has duplicates"

    deltas = timestamps.diff().dropna()
    expected = pd.Timedelta(hours=3)
    assert (deltas == expected).all(), "timestamp_utc spacing is not uniform 3 hours"


def plot_overview(omni_3h: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    window = omni_3h.tail(14 * 8)

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    axes = axes.ravel()

    series_map = [
        ("bz_gsm_3h_min", "Bz GSM min (nT)"),
        ("by_gsm_3h_mean", "By GSM mean (nT)"),
        ("b_scalar_3h_mean", "|B| mean (nT)"),
        ("sw_speed_3h_mean", "SW speed mean (km/s)"),
        ("sw_density_3h_mean", "SW density mean (cm^-3)"),
        ("sw_temperature_3h_mean", "SW temperature mean (K)"),
    ]

    for ax, (column, label) in zip(axes, series_map):
        ax.plot(window["timestamp_utc"], window[column], linewidth=0.8)
        ax.set_ylabel(label)

    axes[0].set_title("OMNI 3h resample (last 2 weeks)")
    axes[-1].set_xlabel("Timestamp (UTC)")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.show()


def main() -> None:
    df = load_omni()
    omni_3h = build_omni_3h(df)
    assert_time_grid(omni_3h)

    # Kp is multiply by 10 from the source (https://omniweb.gsfc.nasa.gov/form/dx1.html)
    omni_3h['kp_index_3h_mean'] = omni_3h['kp_index_3h_mean']/10

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    omni_3h.to_parquet(OUTPUT_PARQUET, index=False)

    plot_overview(omni_3h)
    print("Saved:", OUTPUT_PARQUET)
    print(
        "Date range:",
        omni_3h["timestamp_utc"].min(),
        "→",
        omni_3h["timestamp_utc"].max(),
    )


if __name__ == "__main__":
    main()
