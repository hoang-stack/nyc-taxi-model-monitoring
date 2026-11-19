"""Realised performance (needs labels) and label-free performance estimation (DLE).

Direct loss estimation (DLE, the method NannyML uses for regression): on a reference period
where labels are known, fit a second model that predicts the *absolute error* of the monitored
model from its inputs and its own prediction. On new data, before any label arrives, the mean
predicted absolute error estimates the MAE. It can only see what shows up in the inputs: if
the relation between inputs and target changes (concept drift: same trip, different speed),
the estimate stays where it was. Measuring how often that happens is part of the study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from taximon.data import as_model_input


def mae(pred, y) -> float:
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(y, float))))


def mape(pred, y) -> float:
    y = np.asarray(y, float)
    return float(np.mean(np.abs(np.asarray(pred, float) - y) / y))


def bias(pred, y) -> float:
    """Mean of prediction minus truth: > 0 means trips end sooner than the model said."""
    return float(np.mean(np.asarray(pred, float) - np.asarray(y, float)))


class DLE:
    def __init__(self, rounds: int = 200, seed: int = 0):
        self.rounds, self.seed = rounds, seed

    def _input(self, df: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
        X, _ = as_model_input(df, self.categories)
        X["prediction"] = np.asarray(pred, np.float32)
        return X

    def fit(self, df: pd.DataFrame, pred: np.ndarray, y: np.ndarray, categories) -> DLE:
        import lightgbm as lgb

        self.categories = categories
        params = {
            "objective": "l2",  # conditional mean of |error| -> its average estimates the MAE
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 200,
            "verbose": -1,
            "num_threads": 8,
            "seed": self.seed,
        }
        err = np.abs(np.asarray(pred, float) - np.asarray(y, float))
        self.booster = lgb.train(
            params, lgb.Dataset(self._input(df, pred), err), num_boost_round=self.rounds
        )
        return self

    def estimate_mae(self, df: pd.DataFrame, pred: np.ndarray) -> float:
        return float(np.mean(np.clip(self.booster.predict(self._input(df, pred)), 0, None)))
