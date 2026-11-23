"""Régression logistique avec base spline (relation non linéaire, calibration fiable)."""
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic(**kwargs):
    C = kwargs.get("C", 0.1)
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=2000, solver="lbfgs", class_weight=None)),
        ]
    )
