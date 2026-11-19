"""A reference profile shipped with every model: the training distribution of each feature.

The service compares live traffic against it (PSI gauges on ``/metrics``), and the batch
monitor compares logged requests against it, without needing the training data itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from taximon import drift
from taximon.data import CATEGORICAL, NUMERIC


def build(df: pd.DataFrame, prediction: np.ndarray | None = None, bins: int = 10) -> dict:
    prof: dict = {"rows": len(df), "features": {}}
    cols = {c: df[c] for c in CATEGORICAL + NUMERIC}
    if prediction is not None:
        cols["prediction"] = pd.Series(prediction)
    for c, s in cols.items():
        if c in CATEGORICAL:
            vc = s.astype(str).value_counts(normalize=True)
            prof["features"][c] = {"kind": "categorical", "shares": vc.to_dict()}
        else:
            x = s.to_numpy(float)
            edges = np.unique(np.quantile(x[~np.isnan(x)], np.linspace(0, 1, bins + 1)[1:-1]))
            counts = np.bincount(
                np.searchsorted(edges, x[~np.isnan(x)], side="right"), minlength=len(edges) + 1
            )
            prof["features"][c] = {
                "kind": "numeric",
                "edges": edges.tolist(),
                "shares": (counts / counts.sum()).tolist(),
            }
    return prof


def psi(profile: dict, feature: str, values) -> float:
    f = profile["features"][feature]
    if f["kind"] == "categorical":
        cats = list(f["shares"])
        ref = np.append([f["shares"][c] for c in cats], 0.0)
        s = pd.Series(values).astype(str)
        vc = s.value_counts(normalize=True)
        cur = np.array([vc.get(c, 0.0) for c in cats])
        cur = np.append(cur, max(0.0, 1.0 - cur.sum()))
        return drift.psi_from_shares(ref, cur)
    x = np.asarray(values, float)
    x = x[~np.isnan(x)]
    edges = np.asarray(f["edges"])
    counts = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1)
    return drift.psi_from_shares(np.asarray(f["shares"]), counts / max(len(x), 1))


def save(profile: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(profile))


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
