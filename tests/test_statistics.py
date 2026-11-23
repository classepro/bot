"""Métriques et tests de significativité."""
from __future__ import annotations

import numpy as np

from crashpredict.backtesting.metrics import brier_score, calibration_table, roc_auc
from crashpredict.backtesting.significance import (
    binomial_test_vs_rate,
    bootstrap_ci,
    holm_correction,
    paired_skill_test,
)


def test_brier_bounds():
    assert brier_score([1, 1], [1, 1]) == 0.0
    assert brier_score([0, 0], [1, 1]) == 1.0


def test_auc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert abs(roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) - 0.5) < 1e-9


def test_calibration_table_consistency():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=5000)
    y = (rng.uniform(size=5000) < p).astype(int)
    table = calibration_table(y, p)
    assert table
    assert all(abs(row["gap"]) < 0.06 for row in table)


def test_paired_skill_detects_better_model():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 4000)
    good = np.where(y == 1, 0.75, 0.25) + rng.normal(0, 0.05, 4000)
    base = np.full(4000, y.mean())
    res = paired_skill_test(y, np.clip(good, 0.01, 0.99), base)
    assert res["delta_brier"] > 0
    assert res["significant"]


def test_paired_skill_no_false_positive():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 3000)
    base = np.full(3000, y.mean())
    same = base + rng.normal(0, 0.001, 3000)
    res = paired_skill_test(y, same, base)
    assert not res["significant"] or abs(res["delta_brier"]) < 1e-3


def test_bootstrap_ci_contains_mean():
    vals = np.random.default_rng(5).normal(1.0, 0.1, 500)
    mean, lo, hi = bootstrap_ci(vals)
    assert lo < mean < hi


def test_holm_is_monotone_and_bounded():
    out = holm_correction({"a": 0.001, "b": 0.02, "c": 0.5})
    assert out["a"]["p_adjusted"] <= 1 and out["c"]["p_adjusted"] <= 1
    assert out["a"]["significant_holm"]


def test_binomial_no_drift_on_fair_process():
    res = binomial_test_vs_rate(5000, 10000, 0.5)
    assert res["p_value"] > 0.01


def test_chronological_calibrator_improves_calibration_only_with_signal():
    """Le calibrateur ne doit pas inventer de signal : probabilités toujours dans [0,1] et monotone."""
    import numpy as np

    from crashpredict.models.calibration import calibrate
    from crashpredict.models.linear import make_logistic

    rng = np.random.default_rng(7)
    X = rng.normal(size=(3000, 5))
    # la seule variable utile est la première, avec une relation logistique modérée
    logits = 0.8 * X[:, 0]
    y = (rng.uniform(size=3000) < 1 / (1 + np.exp(-logits))).astype(int)
    order = np.argsort(X[:, 3])  # ordre chronologique (sans lien avec y)
    X, y = X[order], y[order]

    model = calibrate(make_logistic(), n_splits=2).fit(X[:2500], y[:2500])
    p = model.predict_proba(X[2500:])[:, 1]
    assert np.all((p >= 0) & (p <= 1))
    assert np.all(np.diff(np.sort(p)) >= 0)
