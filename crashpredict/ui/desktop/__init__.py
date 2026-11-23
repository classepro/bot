"""Interface desktop PySide6 (portage de l'interface Streamlit).

Lancement :
    python -m crashpredict.ui.desktop.app

Elle utilise exactement les mêmes modules métier que la version Streamlit.
"""
from __future__ import annotations

__all__ = ["main"]


def main() -> int:
    from .app import run

    return run()
