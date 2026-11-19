"""Retraining policies replayed on the real weekly stream, 2020-2022.

Each policy starts from the same model (trained on 2019 H1) and, at the end of every week, may
retrain on recent labelled data; the new model serves from the next week on. Trip durations are
known as soon as a trip ends, so a week's labels are available at the end of that week.

All triggers and windows are fixed before looking at the results (they are round numbers a team
would pick, not tuned values); the README reports how sensitive the conclusions are to them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from taximon import data, drift, performance
from taximon.model import ETAModel


@dataclass
class Policy:
    name: str
    every: int | None = None  # periodic: retrain every `every` weeks
    window: int = 8  # weeks of labelled data used for a retrain
    error_ratio: float | None = None  # trigger: last week's MAE > ratio x MAE at deployment
    drift_auc: float | None = None  # trigger: domain AUC(training window, last week) > value

    def wants_retrain(self, weeks_since: int, mae_now: float, mae_at_deploy: float, auc: float):
        if self.every is not None:
            return weeks_since >= self.every
        if self.error_ratio is not None:
            return mae_now > self.error_ratio * mae_at_deploy
        if self.drift_auc is not None:
            return auc > self.drift_auc
        return False


POLICIES = [
    Policy("never retrain"),
    Policy("every week (8-week window)", every=1),
    Policy("every 4 weeks (8-week window)", every=4),
    Policy("every 13 weeks (8-week window)", every=13),
    Policy("every 13 weeks (52-week window)", every=13, window=52),
    Policy("error-triggered: MAE > 1.15x at deploy", error_ratio=1.15),
    Policy("drift-triggered: domain AUC > 0.60", drift_auc=0.60),
]


def replay(
    df: pd.DataFrame,
    initial: ETAModel,
    initial_train: pd.DataFrame,
    monitor_weeks: list[pd.Timestamp],
    policy: Policy,
    rounds: int = 300,  # same as the initial model
    log=print,
) -> pd.DataFrame:
    """Serve ``monitor_weeks`` in order; windows may reach back into 2019 (labelled history)."""
    by_week = {w: g for w, g in df.groupby("week")}
    all_weeks = sorted(by_week)
    model, train_X = initial, data.as_model_input(initial_train, initial.categories)[0]
    weeks_since, mae_at_deploy, rows = 0, None, []
    for i, week in enumerate(monitor_weeks):
        g = by_week[week]
        y = g[data.TARGET].to_numpy()
        p = model.predict(g)
        mae = performance.mae(p, y)
        if mae_at_deploy is None:
            mae_at_deploy = mae  # the first week a model serves sets its baseline
        auc = np.nan
        if policy.drift_auc is not None:
            Xg, _ = data.as_model_input(g, model.categories)
            auc = drift.domain_classifier_auc(train_X, Xg, n=3000)
        last = i + 1 == len(monitor_weeks)
        retrain = not last and policy.wants_retrain(weeks_since + 1, mae, mae_at_deploy, auc)
        rows.append(
            {
                "week": week,
                "mae": mae,
                "bias": performance.bias(p, y),
                "auc": auc,
                "retrained_after": retrain,
            }
        )
        weeks_since += 1
        if retrain:
            j = all_weeks.index(week)
            hist = pd.concat([by_week[w] for w in all_weeks[max(0, j + 1 - policy.window) : j + 1]])
            model = ETAModel(rounds=rounds).fit(hist, trained_through=str(week.date()))
            train_X = data.as_model_input(hist, model.categories)[0]
            weeks_since, mae_at_deploy = 0, None
    out = pd.DataFrame(rows).set_index("week")
    log(
        f"{policy.name}: mean weekly MAE {out['mae'].mean():.3f}, "
        f"{int(out['retrained_after'].sum())} retrains"
    )
    return out
