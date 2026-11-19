"""Score every monitoring signal against what actually happened to the model's error.

Ground truth: a monitored week is *degraded* when the realised MAE leaves the band it occupied
over the 26 reference weeks (mean + 3 sd, the same rule every signal gets). Each signal gets
its own threshold from the same reference weeks, so no signal is tuned on the monitored period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from taximon.drift import Threshold

COVID_SHOCK = pd.Timestamp("2020-03-09")  # week of the first NY emergency declaration (7 March)


def signals(weekly: pd.DataFrame) -> list[str]:
    feature_psi = [c for c in weekly if c.startswith("psi:") and c != "psi:prediction"]
    out = ["dle_mae", "domain_auc", "psi:prediction", "ks:prediction", "max_feature_psi"]
    return out + sorted(feature_psi) + sorted(c for c in weekly if c.startswith(("ks:", "js:")))


def add_aggregates(weekly: pd.DataFrame) -> pd.DataFrame:
    w = weekly.copy()
    feature_psi = [c for c in w if c.startswith("psi:") and c != "psi:prediction"]
    w["max_feature_psi"] = w[feature_psi].max(axis=1)
    return w


def degraded(
    weekly: pd.DataFrame, metric: str = "mae", k: float = 3.0
) -> tuple[pd.Series, Threshold]:
    ref = weekly.loc[weekly["period"] == "reference", metric]
    t = Threshold(k=k).fit(ref)
    return pd.Series(t.alerts(weekly[metric]), index=weekly.index), t


def _first(mask: pd.Series, after: pd.Timestamp):
    hits = mask[(mask.index >= after) & mask]
    return hits.index[0] if len(hits) else None


def score(weekly: pd.DataFrame, k: float = 3.0, metric: str = "mae") -> pd.DataFrame:
    w = add_aggregates(weekly)
    truth, _ = degraded(w, metric, k)
    mon = w["period"] == "monitor"
    calm = mon & (w.index < COVID_SHOCK)  # 10 ordinary weeks before the shock
    first_truth = _first(truth[mon], COVID_SHOCK)
    rows = []
    for s in signals(w):
        ref = w.loc[w["period"] == "reference", s].dropna()
        if len(ref) < 5:
            continue
        t = Threshold(k=k).fit(ref)
        alert = pd.Series(t.alerts(w[s].fillna(-np.inf)), index=w.index)
        a, d = alert[mon], truth[mon]
        tp = int((a & d).sum())
        first = _first(a, COVID_SHOCK)
        rows.append(
            {
                "signal": s,
                "threshold": t.upper,
                "alert_weeks": int(a.sum()),
                "precision": tp / a.sum() if a.sum() else np.nan,
                "recall": tp / d.sum() if d.sum() else np.nan,
                "false_alarms_before_shock": int(alert[calm].sum()),
                "first_alert_after_shock": first,
                "delay_weeks_vs_error": (
                    (first - first_truth).days // 7
                    if first is not None and first_truth is not None
                    else np.nan
                ),
                "spearman_with_error": float(w.loc[mon, s].rank().corr(w.loc[mon, metric].rank())),
            }
        )
    return pd.DataFrame(rows).set_index("signal")
