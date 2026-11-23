"""Interfaces : Streamlit (web) et PySide6 (desktop).

L'interface ne contient AUCUNE logique métier : elle appelle uniquement les
modules crashpredict (db, stats, prediction, training, collection).
"""
from __future__ import annotations

__all__ = ["theme"]
