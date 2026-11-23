"""Forêts aléatoires (baseline non linéaire robuste, probabilités brutes puis calibrées)."""
from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def make_random_forest(**kwargs):
    n = int(kwargs.get("n_estimators", 250))
    depth = kwargs.get("max_depth", 8)
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n,
                    max_depth=depth,
                    min_samples_leaf=50,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=kwargs.get("random_state", 42),
                ),
            ),
        ]
    )


def make_extra_trees(**kwargs):
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=int(kwargs.get("n_estimators", 250)),
                    max_depth=kwargs.get("max_depth", 10),
                    min_samples_leaf=50,
                    n_jobs=-1,
                    random_state=kwargs.get("random_state", 42),
                ),
            ),
        ]
    )
