"""The monitored model: a LightGBM ETA regressor on log trip duration, with save/load."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from taximon.data import TARGET, as_model_input, inv_log_target, log_target

PARAMS = {
    "objective": "l2",  # on log1p(minutes): errors become relative, like the ETA error users feel
    "learning_rate": 0.1,
    "num_leaves": 255,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "cat_smooth": 10,
    "verbose": -1,
    "num_threads": 8,
    "seed": 0,
}


class ETAModel:
    def __init__(self, rounds: int = 300, params: dict | None = None):
        self.rounds = rounds
        self.params = {**PARAMS, **(params or {})}
        self.booster = None
        self.categories: dict[str, list] = {}
        self.meta: dict = {}

    def fit(self, df: pd.DataFrame, **meta) -> ETAModel:
        import lightgbm as lgb

        X, self.categories = as_model_input(df)
        y = log_target(df[TARGET].to_numpy(np.float64))
        self.booster = lgb.train(self.params, lgb.Dataset(X, y), num_boost_round=self.rounds)
        self.meta = {"rows": len(df), "rounds": self.rounds, **meta}
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = as_model_input(df, self.categories)
        return inv_log_target(self.booster.predict(X)).astype(np.float32)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(p / "model.txt"))
        (p / "meta.json").write_text(
            json.dumps({"categories": self.categories, "meta": self.meta}, default=str)
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> ETAModel:
        import lightgbm as lgb

        p = Path(path)
        m = cls()
        m.booster = lgb.Booster(model_file=str(p / "model.txt"))
        info = json.loads((p / "meta.json").read_text())
        m.categories, m.meta = info["categories"], info["meta"]
        return m
