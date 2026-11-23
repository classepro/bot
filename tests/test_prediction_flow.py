"""Flux complet : prédiction -> attente -> validation sur 3 manches -> statistiques."""
from __future__ import annotations

from crashpredict import db
from crashpredict.prediction import make_prediction
from crashpredict.validation import close_pending_predictions


def test_prediction_requires_data(monkeypatch, tmp_path):
    from crashpredict import config as cfg

    empty = tmp_path / "empty.db"
    monkeypatch.setattr(cfg, "DB_PATH", empty)
    db.init_db(empty)
    import pytest
    with pytest.raises(RuntimeError):
        make_prediction(target=2.0)


def test_prediction_lifecycle(temp_db):
    pred = make_prediction(target=1.5)
    assert pred["status"] == "PENDING"
    assert len(pred["window_round_ids"]) == 3
    stored = db.get_prediction(pred["id"])
    assert stored is not None and stored["status"] == "PENDING"

    # on ajoute les 3 manches suivantes au hasard mais connues
    last = db.last_round_id()
    db.upsert_rounds([
        {"round_id": last + 1, "multiplier": 1.0},
        {"round_id": last + 2, "multiplier": 1.1},
        {"round_id": last + 3, "multiplier": 1.2},
    ])
    n = close_pending_predictions()
    assert n >= 1
    stored = db.get_prediction(pred["id"])
    assert stored["status"] == "FAILURE"  # aucun des trois n'atteint 1.5x
    assert stored["actual_max"] == 1.2


def test_probability_is_bounded(temp_db):
    from crashpredict.prediction import current_analysis

    analysis = current_analysis()
    for row in analysis["grid"]:
        assert 0.0 <= row["probability"] <= 1.0
    assert analysis["n_rounds"] > 0
