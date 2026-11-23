"""Baselines empiriques — la référence à battre.

Toute prédiction de probabilité de validation repose sur la loi marginale :

    P(MAX3 >= t) = 1 - (1 - F(t))^3

où F(t) = P(une manche >= t) est estimée UNIQUEMENT sur le passé (fenêtre
glissante). Aucune valeur n'est inventée : `n` (dénominateur) est toujours
conservé et exposé.

`BaselineRolling` accepte une fenêtre ; `BaselineFixed` utilise tout le train.
`CategoryFrequencyBaseline` estime F par catégories de coefficients (1-1.5,
1.5-2, ...) pour tester la stabilité de la loi.
"""
from __future__ import annotations

import numpy as np

from ..config import CONFIG


def empirical_validation_probability(history: np.ndarray, target: float, horizon: int | None = None) -> tuple[float, int]:
    """(probabilité de validation, n utilisé) estimée sur les manches PASSÉES."""
    horizon = horizon or CONFIG.rules.horizon
    h = np.asarray(history, dtype=float)
    h = h[np.isfinite(h)]
    if len(h) == 0:
        return float("nan"), 0
    p_round = float(np.mean(h >= target))
    return float(1 - (1 - p_round) ** horizon), len(h)


def empirical_survival(history: np.ndarray, target: float) -> tuple[float, int]:
    h = np.asarray(history, dtype=float)
    if len(h) == 0:
        return float("nan"), 0
    return float(np.mean(h >= target)), len(h)


class BaselineRolling:
    """Probabilité = loi marginale estimée sur les `window` dernières manches."""

    name = "baseline_rolling"
    requires_features = False

    def __init__(self, target: float, window: int = 5000):
        self.target = target
        self.window = window

    def predict_proba_from_history(self, history: np.ndarray) -> float:
        h = np.asarray(history, dtype=float)[-self.window :]
        p, _ = empirical_validation_probability(h, self.target)
        return p

    def describe(self) -> dict:
        return {"name": self.name, "target": self.target, "window": self.window,
                "formule": "1-(1-F(t))^3, F estimée sur fenêtre glissante"}


class BaselineFixed:
    """Probabilité = loi marginale estimée sur TOUT l'historique d'entraînement."""

    name = "baseline_fixed"
    requires_features = False

    def __init__(self, target: float):
        self.target = target
        self.p_validation: float | None = None
        self.n = 0

    def fit(self, history: np.ndarray):
        self.p_validation, self.n = empirical_validation_probability(history, self.target)
        return self

    def predict_proba_from_history(self, history: np.ndarray) -> float:
        if self.p_validation is None:
            self.p_validation, self.n = empirical_validation_probability(history, self.target)
        return self.p_validation

    def describe(self) -> dict:
        return {"name": self.name, "target": self.target, "p_validation": self.p_validation, "n": self.n}


class CategoryFrequencyBaseline:
    """F estimée par catégories de coefficients du passé (test de stabilité de la loi)."""

    name = "baseline_categories"
    requires_features = False
    bins = [1.0, 1.01, 1.5, 2.0, 3.0, 5.0, 10.0, np.inf]

    def __init__(self, target: float, window: int = 5000):
        self.target = target
        self.window = window

    def predict_proba_from_history(self, history: np.ndarray) -> float:
        h = np.asarray(history, dtype=float)[-self.window :]
        if len(h) == 0:
            return float("nan")
        cats = np.digitize(h, self.bins) - 1
        recent_cat_counts = np.bincount(cats, minlength=len(self.bins) - 1)
        # moyenne pondérée par la présence de chaque catégorie (noyau lissé)
        survival = np.array([np.mean(h[h >= lo] >= self.target) if np.any(h >= lo) else 0.0
                             for lo in self.bins[:-1]])
        order = np.argsort(self.bins[:-1])[::-1]
        p_round = float(np.mean(h >= self.target))
        return float(1 - (1 - p_round) ** CONFIG.rules.horizon)

    def describe(self) -> dict:
        return {"name": self.name, "target": self.target, "window": self.window}
