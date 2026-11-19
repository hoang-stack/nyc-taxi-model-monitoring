"""Drift statistics, written from scratch (numpy only, plus LightGBM for the domain classifier).

Every function compares a *reference* sample with a *current* batch and returns one number that
grows with the amount of drift. None of them returns a p-value on purpose: with tens of
thousands of rows per batch every difference is "significant". Alert thresholds come instead
from how much each statistic moves between ordinary weeks (see ``Threshold``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-4


def _numeric_edges(ref: np.ndarray, bins: int) -> np.ndarray:
    edges = np.unique(np.quantile(ref[~np.isnan(ref)], np.linspace(0, 1, bins + 1)[1:-1]))
    return edges


def _shares_numeric(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    x = x[~np.isnan(x)]
    counts = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1)
    return counts / max(len(x), 1)


def _shares_categorical(x: pd.Series, categories: list) -> np.ndarray:
    """Shares of each reference category plus one bucket for everything unseen."""
    vc = x.value_counts(normalize=True)
    known = np.array([vc.get(c, 0.0) for c in categories])
    return np.append(known, max(0.0, 1.0 - known.sum()))


def psi_from_shares(p_ref: np.ndarray, p_cur: np.ndarray) -> float:
    """Population stability index: sum (cur - ref) * ln(cur / ref), with smoothing."""
    p_ref = (p_ref + EPS) / (1 + EPS * len(p_ref))
    p_cur = (p_cur + EPS) / (1 + EPS * len(p_cur))
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def psi(ref, cur, categorical: bool = False, bins: int = 10) -> float:
    if categorical:
        cats = pd.Series(ref).value_counts().index.tolist()
        return psi_from_shares(
            _shares_categorical(pd.Series(ref), cats), _shares_categorical(pd.Series(cur), cats)
        )
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = _numeric_edges(ref, bins)
    return psi_from_shares(_shares_numeric(ref, edges), _shares_numeric(cur, edges))


def ks(ref, cur) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: max distance between the empirical CDFs."""
    a = np.sort(np.asarray(ref, float)[~np.isnan(np.asarray(ref, float))])
    b = np.sort(np.asarray(cur, float)[~np.isnan(np.asarray(cur, float))])
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def jensen_shannon(ref, cur) -> float:
    """Jensen-Shannon distance (base 2, in [0, 1]) between two categorical distributions."""
    cats = pd.Series(ref).value_counts().index.tolist()
    p = _shares_categorical(pd.Series(ref), cats)
    q = _shares_categorical(pd.Series(cur), cats)
    m = (p + q) / 2

    def kl(a, b):
        nz = a > 0
        return np.sum(a[nz] * np.log2(a[nz] / b[nz]))

    return float(np.sqrt(max(0.0, (kl(p, m) + kl(q, m)) / 2)))


def domain_classifier_auc(
    X_ref: pd.DataFrame, X_cur: pd.DataFrame, n: int = 5000, seed: int = 0
) -> float:
    """Multivariate drift: how well a classifier tells current rows from reference rows.

    Equal-size samples, 2-fold cross-fitting, ROC AUC of the out-of-fold scores. 0.5 means the
    batches are indistinguishable; it catches changes in the *joint* distribution that no
    single-feature statistic sees (e.g. the same zones at different hours).
    """
    import lightgbm as lgb

    rng = np.random.default_rng(seed)
    k = min(n, len(X_ref), len(X_cur))
    a = X_ref.iloc[rng.choice(len(X_ref), k, replace=False)]
    b = X_cur.iloc[rng.choice(len(X_cur), k, replace=False)]
    X = pd.concat([a, b], ignore_index=True)
    y = np.r_[np.zeros(k), np.ones(k)]
    fold = rng.permutation(2 * k) % 2
    score = np.empty(2 * k)
    params = {
        "objective": "binary",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "verbose": -1,
        "num_threads": 4,
        "seed": seed,
    }
    for f in (0, 1):
        tr, te = fold != f, fold == f
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=100)
        score[te] = m.predict(X[te])
    return auc(y, score)


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC via the rank-sum (Mann-Whitney) identity, with ties averaged."""
    ranks = pd.Series(score).rank(method="average").to_numpy()
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@dataclass
class Threshold:
    """Alert when a statistic leaves the band it occupied over ordinary (reference) weeks.

    ``upper = mean + k * std`` of the statistic over the reference weeks, the rule used by
    NannyML and most monitoring tools; ``lower`` is optional (performance can also improve).
    """

    k: float = 3.0
    two_sided: bool = False

    def fit(self, values) -> Threshold:
        v = np.asarray(values, float)
        self.mean, self.std = float(np.mean(v)), float(np.std(v, ddof=1))
        self.upper = self.mean + self.k * self.std
        self.lower = self.mean - self.k * self.std if self.two_sided else -np.inf
        return self

    def alerts(self, values) -> np.ndarray:
        v = np.asarray(values, float)
        return (v > self.upper) | (v < self.lower)
