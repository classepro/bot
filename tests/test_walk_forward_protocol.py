"""Protocole walk-forward : purge, borne de fenêtre, baselines causales, agrégation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from crashpredict.backtesting.walk_forward import (
    build_prediction_records,
    choose_deployed_model,
    rolling_empirical,
    run_walk_forward,
    training_slice_end,
)
from crashpredict.features import build_dataset


def _synthetic(n: int = 4000, seed: int = 0) -> np.ndarray:
    """Suite i.i.d. réaliste (queue lourde) — données de test uniquement, jamais utilisées
    pour conclure quoi que ce soit sur les données réelles."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=n)
    m = np.maximum(1.0, 0.99 / np.maximum(1 - u, 1e-9))
    return np.round(m, 2)


def test_training_slice_never_touches_test_targets():
    for start in (500, 1000, 5000):
        assert training_slice_end(start, purge=3) == start - 3


def test_rolling_baseline_is_causal():
    """La baseline glissante à l'instant j ne doit pas dépendre des manches j, j+1, j+2."""
    history = _synthetic(1000, seed=1)
    future = _synthetic(50, seed=2)
    base = rolling_empirical(history, future, 2.0, window=200, horizon=3)

    tampered = future.copy()
    tampered[10:] = 99.0  # on change tout le futur à partir de l'indice 10
    after = rolling_empirical(history, tampered, 2.0, window=200, horizon=3)
    assert np.allclose(base[:11], after[:11], atol=0)  # les 11 premières prédictions sont identiques
    assert not np.allclose(base, after)


def test_purge_and_no_future_in_features():
    m = pd.Series(_synthetic(6000, seed=3))
    res = run_walk_forward(
        m, pd.Series(np.arange(len(m))), targets=[2.0], models=["logistic"],
        initial_train=3000, step=1000, max_folds=1, record_predictions=True,
    )
    # la première manche testée respecte l'entraînement initial demandé
    assert res.summary["first_test_round"] >= 3000
    assert res.summary["purge"] == 3
    # chaque prédiction référence des manches strictement postérieures à son ancrage
    preds = res.predictions
    assert not preds.empty
    for _, row in preds.head(50).iterrows():
        assert min(row["window_round_ids"]) > row["anchor_round_id"]
        assert len(row["window_round_ids"]) == 3


def test_baseline_probabilities_in_range_and_not_constant():
    m = pd.Series(_synthetic(8000, seed=4))
    res = run_walk_forward(
        m, pd.Series(np.arange(len(m))), targets=[2.0, 5.0], models=["logistic"],
        initial_train=4000, step=2000, max_folds=2, record_predictions=False,
    )
    base = res.probabilities[res.probabilities["model"] == "baseline_rolling_long"]
    assert base["p"].between(0, 1).all()
    assert base["p"].nunique() > 10  # la baseline réagit à l'historique (pas une constante)


def test_deployed_model_falls_back_to_baseline_without_proof():
    metrics = pd.DataFrame()
    skill = pd.DataFrame(
        [{"model": "lightgbm", "target": 2.0, "delta_brier": -0.001, "significant_holm": True},
         {"model": "logistic", "target": 2.0, "delta_brier": 0.0001, "significant_holm": False}]
    )
    assert choose_deployed_model(metrics, skill) == "baseline_rolling_long"


def test_deployed_model_selected_when_significant():
    metrics = pd.DataFrame()
    skill = pd.DataFrame(
        [{"model": "lightgbm", "target": 2.0, "delta_brier": 0.002, "significant_holm": True},
         {"model": "logistic", "target": 2.0, "delta_brier": 0.0001, "significant_holm": False}]
    )
    assert choose_deployed_model(metrics, skill) == "lightgbm"


def test_prediction_records_follow_spec():
    """Une prédiction = ancrage + 3 manches suivantes + statut calculé par la règle officielle."""
    mult = pd.Series([1.0, 2.0, 1.5, 3.0, 1.1, 1.2, 1.3])
    ds = build_dataset(mult, pd.Series(range(7)), targets=[2.0])
    probs = pd.DataFrame(
        [
            {"fold": 0, "round_id": 0, "model": "baseline_rolling_long", "target": 2.0, "y": 1.0, "p": 0.8},
            {"fold": 0, "round_id": 1, "model": "baseline_rolling_long", "target": 2.0, "y": 1.0, "p": 0.8},
            {"fold": 0, "round_id": 6, "model": "baseline_rolling_long", "target": 2.0, "y": 0.0, "p": 0.5},
        ]
    )
    recs = build_prediction_records(probs, ds, "baseline_rolling_long").set_index("anchor_round_id")

    # ancrage 0 : MAX(2.0, 1.5, 3.0) = 3.0 >= 2.0 -> SUCCÈS ; fenêtre = manches 1, 2, 3
    assert recs.loc[0, "status"] == "SUCCESS"
    assert recs.loc[0, "window_round_ids"] == [1, 2, 3]
    assert recs.loc[0, "actual_max"] == 3.0

    # ancrage 1 : MAX(1.5, 3.0, 1.1) = 3.0 -> SUCCÈS
    assert recs.loc[1, "status"] == "SUCCESS"

    # ancrage 6 (dernière manche) : fenêtre incomplète -> EN ATTENTE
    assert recs.loc[6, "status"] == "PENDING"
    assert pd.isna(recs.loc[6, "actual_max"])

    # aucune fenêtre ne contient de valeur antérieure à l'ancrage
    for anchor, row in recs.iterrows():
        assert min(row["window_round_ids"]) > anchor
