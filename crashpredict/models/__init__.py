"""Registre des modèles comparés par le backtest.

Chaque modèle expose `fit(X, y)` / `predict_proba(X)` et est *calibré* :
la probabilité produite doit pouvoir être lue comme une fréquence attendue.
Les baselines empiriques sont traitées à part (elles ne consomment pas X).
"""
from __future__ import annotations

from .baselines import (
    BaselineFixed,
    BaselineRolling,
    CategoryFrequencyBaseline,
    empirical_validation_probability,
)
from .registry import MODEL_FACTORIES, build_model, model_names

__all__ = [
    "BaselineFixed", "BaselineRolling", "CategoryFrequencyBaseline",
    "empirical_validation_probability",
    "MODEL_FACTORIES", "build_model", "model_names",
]
