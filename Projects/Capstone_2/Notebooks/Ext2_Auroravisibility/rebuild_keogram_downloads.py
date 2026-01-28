#!/usr/bin/env python3
from pathlib import Path
import re
import pandas as pd

ROOT_DIR = Path("/media/nuopel/DATA/ProjectsAndTools/1_Projects/AI/Capstone_2/Notebooks/Ext2_Auroravisibility/keograms_fortsmith")
OUT_CSV = ROOT_DIR / "fortsmith_keogram_downloads2.csv"

BASE_HTTP = "https://data.phys.ucalgary.ca/sort_by_project/TREx/RGB"
STREAM = 2

pattern = re.compile(r"^(\d{8})_(\d{2})_(.+)_full-keogram\.jpg$")

rows = []
for p in ROOT_DIR.rglob("*.jpg"):
    print(p)
    m = pattern.match(p.name)
    if not m:
        continue
    ymd, hh, device = m.group(1), m.group(2), m.group(3)
    date_str = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
    hour = int(hh)

    url = f"{BASE_HTTP}/stream{STREAM}/{date_str[:4]}/{date_str[5:7]}/{date_str[8:10]}/{device}/ut{hour:02d}/{ymd}_{hour:02d}_{device}_full-keogram.jpg"

    rows.append({
        "timestamp_utc": f"{date_str} {hour:02d}:00:00+00:00",
        "date": date_str,
        "hour": hour,
        "ok": True,
        "local_path": str(p),
        "url": url,
        "source": "local-rebuild",
        "error": "",
    })

if not rows:
    raise SystemExit(f"no jpg files matched pattern under {ROOT_DIR}")

df = pd.DataFrame(rows).sort_values(["timestamp_utc", "hour"])
df.to_csv(OUT_CSV, index=False)
print("wrote", OUT_CSV, "rows:", len(df))
