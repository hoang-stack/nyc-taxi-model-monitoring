import numpy as np
import pandas as pd
import pytest

from taximon import drift


def test_psi_is_zero_for_identical_and_grows_with_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=50_000)
    assert drift.psi(ref, ref) == pytest.approx(0.0, abs=1e-9)
    small = drift.psi(ref, rng.normal(0.1, 1, 50_000))
    large = drift.psi(ref, rng.normal(1.0, 1, 50_000))
    assert 0 < small < 0.05 < 0.5 < large


def test_categorical_psi_and_js_handle_unseen_categories():
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    assert drift.psi(ref, ref, categorical=True) == pytest.approx(0.0, abs=1e-9)
    assert drift.jensen_shannon(ref, ref) == pytest.approx(0.0, abs=1e-6)
    cur = pd.Series(["c"] * 100)  # nothing seen before
    assert drift.jensen_shannon(ref, cur) == pytest.approx(1.0, abs=1e-6)
    assert drift.psi(ref, cur, categorical=True) > 5


def test_ks_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=3000), rng.standard_t(3, size=2000)
    assert drift.ks(a, b) == pytest.approx(stats.ks_2samp(a, b).statistic, abs=1e-12)


def test_auc_matches_sklearn_with_ties():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 1000)
    s = np.round(y + rng.normal(size=1000), 1)  # rounding creates ties
    assert drift.auc(y, s) == pytest.approx(metrics.roc_auc_score(y, s), abs=1e-12)


def test_domain_classifier_sees_joint_shift_that_marginals_miss():
    rng = np.random.default_rng(3)
    n = 4000
    x = rng.normal(size=n)
    ref = pd.DataFrame({"x": x, "y": x + 0.1 * rng.normal(size=n)})
    x2 = rng.normal(size=n)
    cur = pd.DataFrame({"x": x2, "y": -x2 + 0.1 * rng.normal(size=n)})  # same marginals
    same = pd.DataFrame({"x": x2, "y": x2 + 0.1 * rng.normal(size=n)})
    assert drift.ks(ref["y"], cur["y"]) < 0.05  # each marginal looks unchanged ...
    assert drift.domain_classifier_auc(ref, cur, n=2000) > 0.9  # ... the joint does not
    assert abs(drift.domain_classifier_auc(ref, same, n=2000) - 0.5) < 0.05


def test_threshold_band():
    t = drift.Threshold(k=3).fit([1.0, 1.1, 0.9, 1.0, 1.05, 0.95])
    assert t.alerts([1.0, 2.0]).tolist() == [False, True]
    t2 = drift.Threshold(k=3, two_sided=True).fit([1.0, 1.1, 0.9, 1.0, 1.05, 0.95])
    assert t2.alerts([0.0]).tolist() == [True]


def test_average_precision():
    from taximon.evaluate import average_precision

    assert average_precision([3, 2, 1, 0], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert average_precision([0, 1, 2, 3], [1, 1, 0, 0]) == pytest.approx((1 / 3 + 2 / 4) / 2)
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 300)
    s = y + rng.normal(size=300)
    assert average_precision(s, y) == pytest.approx(metrics.average_precision_score(y, s))
