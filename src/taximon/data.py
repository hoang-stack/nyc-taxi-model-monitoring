"""Load the monthly samples, run data-quality checks, build model features, cut weekly batches.

The model is an ETA model: predict a yellow-taxi trip's duration from what is known when the
trip is requested (pickup and drop-off zones, time, planned distance, rate code, vendor,
passenger count). The fare and the drop-off time are *not* features.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

AIRPORTS = {1: "EWR", 132: "JFK", 138: "LGA"}
UNKNOWN_ZONES = (264, 265)  # "Unknown" / "Outside of NYC" in the zone lookup
CATEGORICAL = ["PULocationID", "DOLocationID", "pu_borough", "do_borough", "RatecodeID", "VendorID"]
NUMERIC = ["trip_distance", "hour", "weekday", "passenger_count"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "duration_min"

# Each check flags rows that cannot be used for training or scoring. The share of flagged rows
# per batch is itself a monitored signal (schema and pipeline breaks show up here first).
QUALITY_CHECKS = {
    "null_passenger_count": lambda d: d["passenger_count"].isna(),
    "null_ratecode": lambda d: d["RatecodeID"].isna(),
    "non_positive_duration": lambda d: d[TARGET] <= 0,
    "duration_under_1min": lambda d: (d[TARGET] > 0) & (d[TARGET] < 1),  # meter tests, cancels
    "duration_over_3h": lambda d: d[TARGET] > 180,
    "zero_distance": lambda d: d["trip_distance"] <= 0,
    "distance_over_100mi": lambda d: d["trip_distance"] > 100,
    "speed_over_80mph": lambda d: d["trip_distance"] / (d[TARGET].clip(lower=1e-3) / 60) > 80,
    "unknown_zone": lambda d: (
        d["PULocationID"].isin(UNKNOWN_ZONES) | d["DOLocationID"].isin(UNKNOWN_ZONES)
    ),
    "pickup_outside_file_month": lambda d: (
        d["tpep_pickup_datetime"].dt.strftime("%Y-%m") != d["file_month"]
    ),
}


def load(data_dir: str | Path = "data", months: list[str] | None = None) -> pd.DataFrame:
    d = Path(data_dir)
    files = sorted((d / "samples").glob("*.parquet"))
    if months is not None:
        files = [f for f in files if f.stem in set(months)]
    cols = [  # fares and payment type are not used by the model or the checks
        "file_month",
        "VendorID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "RatecodeID",
        "PULocationID",
        "DOLocationID",
    ]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    zones = pd.read_csv(d / "taxi_zone_lookup.csv").set_index("LocationID")["Borough"]
    df["pu_borough"] = df["PULocationID"].map(zones).fillna("Unknown")
    df["do_borough"] = df["DOLocationID"].map(zones).fillna("Unknown")
    df[TARGET] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds().astype("float32") / 60
    df["hour"] = df["tpep_pickup_datetime"].dt.hour.astype("int8")
    df["weekday"] = df["tpep_pickup_datetime"].dt.weekday.astype("int8")
    # Batches are calendar weeks (Monday start) of the pickup time.
    df["week"] = df["tpep_pickup_datetime"].dt.to_period("W-SUN").dt.start_time
    return df


def monthly_counts(data_dir: str | Path = "data") -> pd.Series:
    counts = json.loads((Path(data_dir) / "monthly_counts.json").read_text())
    return pd.Series(counts, dtype="int64").sort_index()


def quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per check (True = the row fails that check)."""
    return pd.DataFrame({name: check(df).fillna(True) for name, check in QUALITY_CHECKS.items()})


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that pass every check, with model-ready dtypes."""
    ok = ~quality_flags(df).any(axis=1)
    out = df.loc[ok].copy()
    for c in CATEGORICAL:
        out[c] = out[c].astype("str" if c.endswith("borough") else "int32")
    out["passenger_count"] = out["passenger_count"].astype("float32")
    return out


def as_model_input(df: pd.DataFrame, categories: dict[str, list] | None = None):
    """Feature frame with pandas categoricals fixed to the training categories (unseen -> NaN)."""
    X = df[FEATURES].copy()
    cats = categories or {c: sorted(X[c].unique().tolist()) for c in CATEGORICAL}
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c], categories=cats[c])
    for c in NUMERIC:
        X[c] = X[c].astype("float32")
    return X, cats


def weekly_batches(df: pd.DataFrame, min_rows: int = 500):
    """Yield (week_start, frame) for every complete week with enough rows."""
    for week, g in df.groupby("week", sort=True):
        if len(g) >= min_rows:
            yield week, g


def log_target(y: np.ndarray) -> np.ndarray:
    return np.log1p(y)


def inv_log_target(z: np.ndarray) -> np.ndarray:
    return np.expm1(z)
