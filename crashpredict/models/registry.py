"""Usines de modèles (une famille par fichier).

Chaque fabrique renvoie un objet scikit-learn compatible `predict_proba`.
Les modèles coûteux (CatBoost, MLP) restent optionnels : si la librairie
n'est pas installée, la fabrique n'est pas proposée (aucun faux résultat).
"""
from __future__ import annotations

from typing import Callable

from .forest import make_extra_trees, make_random_forest
from .gbm import make_catboost, make_lightgbm, make_xgboost
from .linear import make_logistic
from .nn import make_mlp

MODEL_FACTORIES: dict[str, Callable[..., object]] = {
    "logistic": make_logistic,
    "random_forest": make_random_forest,
    "extra_trees": make_extra_trees,
    "lightgbm": make_lightgbm,
    "xgboost": make_xgboost,
    "catboost": make_catboost,
    "mlp": make_mlp,
}


def available_models() -> list[str]:
    return list(MODEL_FACTORIES)


def build_model(name: str, **kwargs):
    if name not in MODEL_FACTORIES:
        raise KeyError(f"modèle inconnu: {name}. Disponibles: {available_models()}")
    return MODEL_FACTORIES[name](**kwargs)


def model_names() -> list[str]:
    return available_models()
