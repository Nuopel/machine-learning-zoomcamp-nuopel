from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
KP_FILE = DATA_DIR / "kp_ap.txt"
OMNI_3H_FILE = DATA_DIR / "processed" / "omni_3h.parquet"
OUTPUT_DIR = DATA_DIR / "processed"
MASTER_FILE = OUTPUT_DIR / "master_3h.parquet"
MASTER_CSV = OUTPUT_DIR / "master_3h.csv"
PLOT_PATH = OUTPUT_DIR / "kp_comparison.png"


def load_kp_spine() -> pd.DataFrame:
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
    df = pd.read_csv(KP_FILE, sep=r"\s+", header=None, names=columns, comment="#")

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


def load_omni_3h() -> pd.DataFrame:
    omni = pd.read_parquet(OMNI_3H_FILE)
    omni = omni.drop(columns=["ap_index_3h_mean"], errors="ignore")
    return omni


def compare_kp(master: pd.DataFrame) -> None:
    compare = master.dropna(subset=["kp", "kp_index_3h_mean"])
    if compare.empty:
        print("No overlapping Kp values to compare.")
        return

    diff = compare["kp"] - compare["kp_index_3h_mean"]
    corr = compare["kp"].corr(compare["kp_index_3h_mean"])
    print("Kp comparison count:", len(compare))
    print("Kp correlation:", round(corr, 4))
    print("Kp mean absolute diff:", round(diff.abs().mean(), 4))
    print("Kp max absolute diff:", round(diff.abs().max(), 4))

    window = compare.tail(14 * 8)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(
        compare["kp"],
        compare["kp_index_3h_mean"],
        s=6,
        alpha=0.4,
        edgecolors="none",
    )
    axes[0].plot([0, 9], [0, 9], color="black", linewidth=0.8)
    axes[0].set_xlabel("GFZ Kp (kp_ap.txt)")
    axes[0].set_ylabel("OMNI Kp (3h mean)")
    axes[0].set_title("Kp comparison")

    axes[1].plot(window["timestamp_utc"], window["kp"], label="GFZ Kp", linewidth=0.8)
    axes[1].plot(
        window["timestamp_utc"],
        window["kp_index_3h_mean"],
        label="OMNI Kp",
        linewidth=0.8,
    )
    axes[1].set_xlabel("Timestamp (UTC)")
    axes[1].set_ylabel("Kp")
    axes[1].legend(loc="upper right")
    axes[1].set_title("Last 2 weeks")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.show()


def assert_time_grid(df: pd.DataFrame) -> None:
    timestamps = df["timestamp_utc"]
    assert timestamps.is_monotonic_increasing, "timestamp_utc is not strictly monotonic"
    assert timestamps.is_unique, "timestamp_utc has duplicates"

    deltas = timestamps.diff().dropna()
    expected = pd.Timedelta(hours=3)
    assert (deltas == expected).all(), "timestamp_utc spacing is not uniform 3 hours"


def report_time_gaps(df: pd.DataFrame, label: str, max_rows: int = 10) -> None:
    timestamps = df["timestamp_utc"]
    deltas = timestamps.diff().dropna()
    expected = pd.Timedelta(hours=3)
    gaps = deltas[deltas != expected]

    print(f"{label} gaps: {len(gaps)}")
    if gaps.empty:
        return

    gap_rows = gaps.head(max_rows)
    report = pd.DataFrame(
        {
            "prev_timestamp": timestamps.shift(1).loc[gap_rows.index],
            "next_timestamp": timestamps.loc[gap_rows.index],
            "delta": gap_rows.values,
        }
    )
    print(report)


def main() -> None:
    kp_df = load_kp_spine()
    omni_3h = load_omni_3h()
    master = kp_df.merge(omni_3h, on="timestamp_utc", how="inner")
    assert_time_grid(master)
    report_time_gaps(master, "Merged (pre-drop)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_master = master.drop(columns=["kp_index_3h_mean"], errors="ignore")
    export_master.to_parquet(MASTER_FILE, index=False)

    required_cols = [
        col
        for col in omni_3h.columns
        if col not in {"timestamp_utc", "sw_temperature_3h_mean"}
    ]
    missing_any = master[required_cols].isna().any(axis=1)
    print("Rows with any OMNI missing:", int(missing_any.sum()))
    if missing_any.any():
        print("First missing timestamps:")
        print(master.loc[missing_any, "timestamp_utc"].head(10).to_string(index=False))

    print("Rows (overlap):", len(master))

    print("Saved:", MASTER_FILE)
    print("Saved:", MASTER_CSV)
    print(
        "Date range:",
        master["timestamp_utc"].min(),
        "→",
        master["timestamp_utc"].max(),
    )

    compare_kp(master)


if __name__ == "__main__":
    main()
