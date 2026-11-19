"""Batch monitoring job over the service's JSONL logs (run from cron or a CI schedule).

Reads the predictions logged in a time window, joins any feedback (real durations), and checks:
data volume, feature drift against the model's profile (PSI), prediction drift, and the
realised MAE where labels have arrived. Returns a report dict; the CLI exits non-zero when an
alert fires so a scheduler can page someone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from taximon import profile as profiles
from taximon.data import CATEGORICAL, NUMERIC

PSI_ALERT = 0.2  # the conventional "significant shift" level for PSI
MIN_ROWS = 200


def _read(log_dir: Path, kind: str) -> pd.DataFrame:
    rows = []
    for f in sorted(log_dir.glob(f"{kind}-*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return pd.DataFrame(rows)


def run(
    log_dir: str | Path,
    model_dir: str | Path,
    psi_alert: float = PSI_ALERT,
    mae_alert: float | None = None,
) -> dict:
    log_dir, model_dir = Path(log_dir), Path(model_dir)
    prof = profiles.load(model_dir / "profile.json")
    preds = _read(log_dir, "predictions")
    report: dict = {"rows": len(preds), "alerts": [], "psi": {}}
    if len(preds) < MIN_ROWS:
        report["alerts"].append(f"only {len(preds)} predictions logged (< {MIN_ROWS})")
        return report
    feats = pd.json_normalize(preds["features"].tolist())
    feats["prediction"] = preds["eta_minutes"].to_numpy()
    for c in CATEGORICAL + NUMERIC + ["prediction"]:
        v = profiles.psi(prof, c, feats[c])
        report["psi"][c] = round(v, 4)
        if v > psi_alert:
            report["alerts"].append(f"PSI {c} = {v:.3f} > {psi_alert}")
    fb = _read(log_dir, "feedback")
    if len(fb):
        joined = preds.merge(fb, on="request_id", how="inner")
        report["labelled"] = len(joined)
        if len(joined):
            err = np.abs(joined["eta_minutes"] - joined["actual_minutes"])
            report["mae"] = round(float(err.mean()), 3)
            if mae_alert is not None and report["mae"] > mae_alert:
                report["alerts"].append(f"MAE {report['mae']:.2f} > {mae_alert}")
    return report
