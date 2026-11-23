"""La règle MAX(3 manches) >= cible doit être appliquée de façon identique partout."""
from __future__ import annotations

import pytest

from crashpredict.config import CONFIG
from crashpredict.validation import apply_rule, validate_window


def test_examples_from_spec():
    # Exemple 1 du cahier des charges : MAX = 2.76 >= 2.50 -> SUCCÈS
    assert apply_rule(2.50, [1.42, 1.87, 2.76])[0] == "SUCCESS"
    # Exemple 2 : MAX = 2.10 < 2.50 -> ÉCHEC
    assert apply_rule(2.50, [1.35, 2.10, 1.99])[0] == "FAILURE"


def test_exact_target_is_success():
    assert apply_rule(2.0, [1.5, 2.0, 1.2])[0] == "SUCCESS"


def test_uses_only_first_three_rounds():
    # une 4e manche énorme ne doit PAS sauver la prédiction
    assert apply_rule(5.0, [1.1, 1.2, 1.3, 100.0])[0] == "FAILURE"


def test_pending_when_window_incomplete():
    status, mx = apply_rule(2.0, [1.5, 3.0])
    assert status == "PENDING"
    assert mx == 3.0


def test_horizon_is_configuration_driven():
    assert CONFIG.rules.horizon == 3


def test_validate_window_payload():
    out = validate_window([10, 11, 12], [1.2, 1.3, 3.0], 2.5)
    assert out["status"] == "SUCCESS"
    assert out["actual_max"] == 3.0
    assert out["window_round_ids"] == [10, 11, 12]
    assert out["validated_at"] is not None
