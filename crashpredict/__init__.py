"""Crash Game AI Predictor — package principal.

Le projet mesure ses performances réelles ; il ne prétend pas prédire
le coefficient exact d'une manche à venir.
"""
from __future__ import annotations

__version__ = "1.0.0"

from .config import CONFIG  # noqa: F401

__all__ = ["CONFIG", "__version__"]
