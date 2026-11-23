"""Gradient boosting : LightGBM, XGBoost, CatBoost (optionnels si absents)."""
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def make_lightgbm(**kwargs):
    from lightgbm import LGBMClassifier

    clf = LGBMClassifier(
        n_estimators=int(kwargs.get("n_estimators", 200)),
        learning_rate=kwargs.get("learning_rate", 0.02),
        num_leaves=int(kwargs.get("num_leaves", 8)),        # peu de feuilles : moins de bruit appris
        min_child_samples=int(kwargs.get("min_child_samples", 1000)),
        max_depth=4,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        n_jobs=kwargs.get("n_jobs", 2),
        random_state=kwargs.get("random_state", 42),
        verbose=-1,
    )
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", clf)])


def make_xgboost(**kwargs):
    from xgboost import XGBClassifier

    clf = XGBClassifier(
        n_estimators=int(kwargs.get("n_estimators", 200)),
        learning_rate=kwargs.get("learning_rate", 0.02),
        max_depth=int(kwargs.get("max_depth", 3)),
        min_child_weight=kwargs.get("min_child_weight", 500.0),
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        tree_method="hist",
        n_jobs=kwargs.get("n_jobs", 2),
        random_state=kwargs.get("random_state", 42),
        eval_metric="logloss",
    )
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", clf)])


def make_catboost(**kwargs):
    from catboost import CatBoostClassifier

    clf = CatBoostClassifier(
        iterations=int(kwargs.get("iterations", 250)),
        learning_rate=kwargs.get("learning_rate", 0.03),
        depth=int(kwargs.get("depth", 3)),
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        random_seed=kwargs.get("random_state", 42),
        verbose=False,
        thread_count=2,
        allow_writing_files=False,
    )
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", clf)])
