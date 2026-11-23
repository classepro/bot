"""Réseau de neurones (MLP) — utile seulement pour de gros volumes (>= 20k lignes)."""
from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MIN_ROWS = 20_000


def make_mlp(**kwargs):
    hidden = kwargs.get("hidden_layer_sizes", (32, 16))
    clf = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        alpha=kwargs.get("alpha", 1e-4),
        learning_rate_init=kwargs.get("learning_rate_init", 1e-3),
        max_iter=kwargs.get("max_iter", 120),
        early_stopping=True,
        n_iter_no_change=8,
        validation_fraction=0.1,
        random_state=kwargs.get("random_state", 42),
    )
    return Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", clf)]
    )
