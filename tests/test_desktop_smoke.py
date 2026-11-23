"""Vérification de la couche desktop (PySide6) — test « smoke » à exécuter hors sandbox.

Ce test est **optionnel** : il est sauté automatiquement quand Qt ne peut pas
s'initialiser (pas d'affichage, libGL absente, conteneur). Dans ce cas, on vérifie
au moins que la logique de données utilisée par les pages fonctionne.

Exécution :
    QT_QPA_PLATFORM=offscreen pytest tests/test_desktop_smoke.py -q
"""
from __future__ import annotations

import importlib.util

import pytest


def _qt_available() -> bool:
    if importlib.util.find_spec("PySide6") is None:
        return False
    try:
        import ctypes

        ctypes.CDLL("libGL.so.1")
        return True
    except OSError:
        return False


def test_desktop_pages_need_data_only():
    """Les pages desktop ne dépendent que des modules métier (vérifié sans Qt)."""
    from crashpredict import db
    from crashpredict.prediction import current_analysis, strategy_table
    from crashpredict.stats import dashboard_stats, prediction_history

    fp = db.data_fingerprint()
    if not fp["n"]:
        pytest.skip("base vide")
    analysis = current_analysis()
    assert analysis["n_rounds"] == fp["n"]
    assert strategy_table()
    assert dashboard_stats(backtest=True)["n_predictions"] >= 0
    assert prediction_history(limit=5) is not None


@pytest.mark.skipif(not _qt_available(), reason="Qt/libGL indisponible dans cet environnement")
def test_desktop_window_builds():
    from PySide6.QtWidgets import QApplication

    from crashpredict.ui.desktop.app import MainWindow
    from crashpredict.ui.theme import pyside_stylesheet

    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(pyside_stylesheet())
    win = MainWindow()
    assert win.stack.count() == 6  # Dashboard, Prédiction, Historique, Entraînement, Données, Rapport
    for i in range(win.stack.count()):
        win.nav.setCurrentRow(i)
        app.processEvents()
        assert win.stack.currentWidget() is not None
    win.close()
