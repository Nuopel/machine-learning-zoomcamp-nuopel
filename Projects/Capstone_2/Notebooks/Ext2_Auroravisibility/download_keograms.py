#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd

from keogram_utils import TrexKeogramDownloader

# --- Config (edit these) ---
BASE_DIR = Path(__file__).resolve().parent
LOCATION = "fortsmith"
JOBS_CSV = BASE_DIR / f"{LOCATION}_keogram_jobs.csv"
OUT_DIR = BASE_DIR / f"keograms_{LOCATION}"

# TREx site/device (set these before running)
SITE_UID = "fsmi"        # example placeholder
DEVICE = "fsmi_rgb-09"   # example placeholder

# Optional filters
START_DATE = "2023-11-23"
END_DATE = None          # "2025-12-31"
MAX_ROWS = None          # int, for quick tests
SLEEP = 0.0              # seconds between requests
LOG_EVERY = 20           # progress print interval

# Debug helpers
TEST_FETCH = False       # set True to test a single download
TEST_DATE = "2023-04-10"
TEST_HOUR = 6


def expected_path(out_dir: Path, date_str: str, hour: int, device: str) -> Path:
    y, m, d = date_str.split("-")
    ymd = f"{y}{m}{d}"
    hh = f"{hour:02d}"
    filename = f"{ymd}_{hh}_{device}_full-keogram.jpg"
    return out_dir / date_str / f"ut{hh}" / filename


def main():
    if not JOBS_CSV.exists():
        raise SystemExit(f"jobs CSV not found: {JOBS_CSV}")
    if "XX" in DEVICE:
        raise SystemExit("Set DEVICE to a real TREx device (e.g., yknf_rgb-08)")

    jobs = pd.read_csv(JOBS_CSV, parse_dates=["timestamp_utc"], low_memory=False)
    if "timestamp_utc" not in jobs.columns:
        raise SystemExit("jobs CSV must include timestamp_utc column")

    ts = pd.to_datetime(jobs["timestamp_utc"], utc=True)
    jobs = jobs.copy()
    jobs["timestamp_utc"] = ts

    if START_DATE:
        jobs = jobs[jobs["timestamp_utc"] >= pd.Timestamp(START_DATE, tz="UTC")]
    if END_DATE:
        jobs = jobs[jobs["timestamp_utc"] <= pd.Timestamp(END_DATE, tz="UTC")]

    jobs = jobs.sort_values("timestamp_utc")
    if MAX_ROWS:
        jobs = jobs.head(MAX_ROWS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dl = TrexKeogramDownloader(
        save_dir=OUT_DIR,
        stream=2,
        dataset="TREX_RGB_HOURLY_KEOGRAM",
        timeout=60,
    )

    print(f"jobs: {len(jobs)} rows")
    if len(jobs) == 0:
        raise SystemExit("No jobs after filtering.")
    print(f"range: {jobs['timestamp_utc'].min()}  ->  {jobs['timestamp_utc'].max()}")
    print(f"first job: {jobs.iloc[0]['timestamp_utc']}")
    print(f"site/device: {SITE_UID} / {DEVICE}")

    if TEST_FETCH:
        res = dl.fetch_one(TEST_DATE, TEST_HOUR, "Yellowknife", SITE_UID, DEVICE, save=True)
        print("test fetch:", res)
        return

    rows = []
    downloaded = 0
    skipped = 0
    failed = 0
    for _, row in jobs.iterrows():
        ts = row["timestamp_utc"]
        date_str = ts.strftime("%Y-%m-%d")
        hour = int(ts.hour)

        exp_path = expected_path(OUT_DIR, date_str, hour, DEVICE)
        if exp_path.exists() and exp_path.stat().st_size > 0:
            rows.append({
                "timestamp_utc": ts,
                "date": date_str,
                "hour": hour,
                "ok": True,
                "local_path": str(exp_path),
                "url": None,
                "source": "local",
                "error": None,
            })
            skipped += 1
            continue

        res = dl.fetch_one(date_str, hour, "Yellowknife", SITE_UID, DEVICE, save=True)
        rows.append({
            "timestamp_utc": ts,
            "date": date_str,
            "hour": hour,
            "ok": res.get("ok"),
            "local_path": res.get("local_path"),
            "url": res.get("url"),
            "source": res.get("source"),
            "error": res.get("error"),
        })
        if res.get("ok"):
            downloaded += 1
        else:
            failed += 1
            if res.get("error"):
                print(f"fail {date_str} UT{hour:02d}: {res['error']}")

        if SLEEP:
            import time
            time.sleep(SLEEP)

        if (len(rows) % LOG_EVERY) == 0:
            print(f"progress {len(rows)} / {len(jobs)} | downloaded={downloaded} skipped={skipped} failed={failed}")

    out_csv = OUT_DIR / f"{LOCATION}_keogram_downloads.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"saved: {out_csv}")
    print(f"done | downloaded={downloaded} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
