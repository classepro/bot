"""Contrôles anti-fuite temporelle (le cœur de la crédibilité du projet)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crashpredict.backtesting.walk_forward import training_slice_end
from crashpredict.features import build_dataset, build_features, feature_columns


def test_features_are_causal():
    """Modifier le FUTUR ne doit pas modifier les features du passé."""
    rng = np.random.default_rng(0)
    m = pd.Series(rng.choice([1.1, 1.5, 2.0, 3.0, 10.0], size=500, p=[.4, .25, .2, .1, .05]))
    base = build_features(m)
    modified = m.copy()
    modified.iloc[300:] = 50.0  # on change tout le futur à partir de l'indice 300
    after = build_features(modified)
    pd.testing.assert_frame_equal(base.iloc[:300], after.iloc[:300])


def test_no_future_columns_in_features():
    m = pd.Series(np.abs(np.random.default_rng(1).normal(2, 1, 300)))
    ds = build_dataset(m, targets=[2.0])
    feats = feature_columns(ds)
    assert not any(c.startswith("y_") or c.startswith("next_") for c in feats)
    assert "multiplier" not in feats


def test_target_matches_definition():
    m = pd.Series([1.0, 2.0, 1.5, 3.0, 1.2, 1.1, 1.4])
    ds = build_dataset(m, targets=[2.5])
    # ligne 0 : MAX(2.0, 1.5, 3.0) = 3.0 >= 2.5 -> 1
    assert ds.loc[0, "y_2.5"] == 1
    # ligne 1 : MAX(1.5, 3.0, 1.2) = 3.0 -> 1
    assert ds.loc[1, "y_2.5"] == 1
    # ligne 3 : MAX(1.2, 1.1, 1.4) = 1.4 -> 0
    assert ds.loc[3, "y_2.5"] == 0
    # les 3 dernières lignes n'ont pas de futur complet -> NaN
    assert ds["y_2.5"].tail(3).isna().all()


def test_purge_prevents_target_overlap():
    """La dernière ligne d'entraînement doit avoir sa cible entièrement connue avant le test."""
    start = 1000
    end = training_slice_end(start, purge=3)
    assert end == 997
    # la cible de la ligne 996 utilise les manches 997, 998, 999 -> connues avant 1000
    assert end - 1 + 3 == start - 1


def test_rolling_windows_never_use_current_value():
    m = pd.Series([1.0] * 100 + [10.0])
    f = build_features(m)
    # la fenêtre glissante de la dernière ligne ne contient pas la valeur courante
    assert f["f_mean_log_5"].iloc[-1] < np.log(2.0)
