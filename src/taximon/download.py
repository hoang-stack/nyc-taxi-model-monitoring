"""Download NYC TLC yellow-taxi trip records month by month and keep a uniform random sample.

The raw monthly files are 5-110 MB each; only a fixed *fraction* of rows is kept and each raw
file is deleted after sampling. A fixed fraction (not a fixed count) keeps the sampling rate the
same in every month, so a weekly batch that spans two months is not distorted, and traffic
volume stays visible in the sample. Rows are *not* cleaned here: data-quality problems are part
of what the monitoring must see.

    taximon download --start 2019-01 --end 2022-12 --fraction 0.03
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{}.parquet"
ZONES = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
COLUMNS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "total_amount",
]


def months(start: str, end: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def sample_month(month: str, raw_dir: Path, frac: float, seed: int) -> tuple[pd.DataFrame, int]:
    path = raw_dir / f"yellow_{month}.parquet"
    urllib.request.urlretrieve(URL.format(month), path)
    try:
        table = pq.read_table(path, columns=COLUMNS)
        total = table.num_rows
        rng = np.random.default_rng([seed, int(month.replace("-", ""))])
        idx = np.flatnonzero(rng.random(total) < frac)  # Bernoulli: same rate in every month
        df = table.take(idx).to_pandas()
    finally:
        path.unlink(missing_ok=True)
    for c in ("VendorID", "passenger_count", "RatecodeID", "payment_type"):
        df[c] = df[c].astype("Float32")  # nullable; int64 in some months, float/NaN in others
    for c in ("PULocationID", "DOLocationID"):
        df[c] = df[c].astype("int16")
    for c in ("trip_distance", "fare_amount", "total_amount"):
        df[c] = df[c].astype("float32")
    for c in ("tpep_pickup_datetime", "tpep_dropoff_datetime"):
        df[c] = df[c].astype("datetime64[us]")
    df.insert(0, "file_month", month)
    return df, total


def download(start="2019-01", end="2022-12", fraction=0.03, out="data", seed=0) -> None:
    out = Path(out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    counts_path = out / "monthly_counts.json"
    counts = json.loads(counts_path.read_text()) if counts_path.exists() else {}
    if not (out / "taxi_zone_lookup.csv").exists():
        urllib.request.urlretrieve(ZONES, out / "taxi_zone_lookup.csv")
    for month in months(start, end):
        target = out / "samples" / f"{month}.parquet"
        if target.exists() and month in counts:
            continue
        df, total = sample_month(month, raw, fraction, seed)
        df.to_parquet(target, index=False)
        counts[month] = total
        counts_path.write_text(json.dumps(counts, indent=1))
        print(f"{month}: {total:>9,} trips, kept {len(df):,}", flush=True)
