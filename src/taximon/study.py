"""The monitoring study: one frozen model, every monitoring signal, week by week, 2019-2022.

Periods (all by pickup week, Monday start):

- train      2019-01-07 .. 2019-06-30   the model is fitted here, once
- reference  2019-07-01 .. 2019-12-29   ordinary weeks with labels: alert thresholds and the
                                        DLE error model are calibrated here
- monitor    2019-12-30 .. 2022-12-25   the model runs frozen; every signal is logged weekly

The question is not "is there drift" (there always is) but *which signals would have told an
on-call engineer, early and without crying wolf, that the model's error had really changed.*
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from taximon import data, drift, performance
from taximon.model import ETAModel

TRAIN = ("2019-01-07", "2019-06-30")
REFERENCE = ("2019-07-01", "2019-12-29")
MONITOR = ("2019-12-30", "2022-12-25")
SAMPLE_FRACTION = 0.03  # scripts/download_tlc.py


def _between(df: pd.DataFrame, period: tuple[str, str]) -> pd.DataFrame:
    lo, hi = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    return df[(df["week"] >= lo) & (df["week"] <= hi)]


T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def quality_by_week(raw: pd.DataFrame) -> pd.DataFrame:
    flags = data.quality_flags(raw)
    flags["week"] = raw["week"].to_numpy()
    q = flags.groupby("week").mean()
    q["any_check_failed"] = flags.drop(columns="week").any(axis=1).groupby(flags["week"]).mean()
    q["trips_estimated"] = raw.groupby("week").size() / SAMPLE_FRACTION
    return q


def run(data_dir="data", out_dir="results", domain_n: int = 5000) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = data.load(data_dir)
    raw = raw[(raw["week"] >= pd.Timestamp(TRAIN[0])) & (raw["week"] <= pd.Timestamp(MONITOR[1]))]
    log(f"{len(raw):,} sampled trips")
    quality_by_week(raw).to_csv(out / "quality_weekly.csv")
    df = data.clean(raw)
    del raw
    log(f"{len(df):,} trips pass every check")

    train = _between(df, TRAIN)
    if (out / "model_2019h1" / "model.txt").exists():  # resume: the model is deterministic
        model = ETAModel.load(out / "model_2019h1")
    else:
        model = ETAModel().fit(train, train_period=TRAIN)
        model.save(out / "model_2019h1")
    log(f"model trained on {len(train):,} trips")
    X_train, _ = data.as_model_input(train, model.categories)
    pred_train = model.predict(train)

    # DLE: fit on a random half of the reference rows, estimate reference weeks on the other
    # half, so the reference-week estimates used for its alert threshold are out-of-sample.
    ref = _between(df, REFERENCE)
    pred_ref = model.predict(ref)
    rng = np.random.default_rng(0)
    half = rng.random(len(ref)) < 0.5
    dle = performance.DLE().fit(
        ref[half], pred_ref[half], ref[data.TARGET].to_numpy()[half], model.categories
    )
    log("DLE fitted")
    ref_holdout_weeks = set(ref.loc[~half, "week"].unique())

    rows = []
    period_weeks = _between(df, (REFERENCE[0], MONITOR[1]))
    holdout_mask = pd.Series(~half, index=ref.index)
    for week, g in data.weekly_batches(period_weeks):
        in_ref = week <= pd.Timestamp(REFERENCE[1])
        p = model.predict(g)
        y = g[data.TARGET].to_numpy()
        row = {
            "week": week,
            "period": "reference" if in_ref else "monitor",
            "rows": len(g),
            "mae": performance.mae(p, y),
            "mape": performance.mape(p, y),
            "bias": performance.bias(p, y),
            "mean_duration": float(np.mean(y)),
            "mean_speed_mph": float(np.mean(g["trip_distance"] / (y / 60))),
        }
        if in_ref and week in ref_holdout_weeks:
            m = holdout_mask.reindex(g.index).fillna(False).to_numpy()
            row["dle_mae"] = dle.estimate_mae(g[m], p[m])
        elif not in_ref:
            row["dle_mae"] = dle.estimate_mae(g, p)
        Xg, _ = data.as_model_input(g, model.categories)
        for c in data.CATEGORICAL:
            row[f"psi:{c}"] = drift.psi(train[c], g[c], categorical=True)
            row[f"js:{c}"] = drift.jensen_shannon(train[c], g[c])
        for c in data.NUMERIC:
            row[f"psi:{c}"] = drift.psi(train[c], g[c])
            row[f"ks:{c}"] = drift.ks(train[c].to_numpy(float), g[c].to_numpy(float))
        row["psi:prediction"] = drift.psi(pred_train, p)
        row["ks:prediction"] = drift.ks(pred_train, p)
        row["domain_auc"] = drift.domain_classifier_auc(X_train, Xg, n=domain_n)
        rows.append(row)
        if len(rows) % 20 == 0:
            log(f"{len(rows)} weeks, at {week.date()}")
    weekly = pd.DataFrame(rows).set_index("week")
    weekly.to_csv(out / "weekly.csv")
    (out / "study.json").write_text(
        json.dumps(
            {
                "train": TRAIN,
                "reference": REFERENCE,
                "monitor": MONITOR,
                "train_rows": len(train),
                "reference_rows": len(ref),
                "monitor_rows": int(weekly.loc[weekly["period"] == "monitor", "rows"].sum()),
                "sample_fraction": SAMPLE_FRACTION,
            },
            indent=1,
        )
    )
    log("done")
    return out / "weekly.csv"


def run_policies(data_dir="data", out_dir="results") -> Path:
    """Replay every retraining policy over the monitored weeks (resumable, one CSV per run)."""
    from taximon.policies import POLICIES, replay

    out = Path(out_dir)
    df = data.clean(data.load(data_dir))
    df = df[(df["week"] >= pd.Timestamp(TRAIN[0])) & (df["week"] <= pd.Timestamp(MONITOR[1]))]
    train = _between(df, TRAIN)
    model = ETAModel.load(out / "model_2019h1")
    weeks = [w for w, _ in data.weekly_batches(_between(df, MONITOR))]
    path = out / "policies_weekly.csv"
    done = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else None
    for policy in POLICIES:
        if done is not None and policy.name in set(done["policy"]):
            continue
        res = replay(df, model, train, weeks, policy, log=log).assign(policy=policy.name)
        done = res if done is None else pd.concat([done, res])
        done.to_csv(path)
    return path
