"""Collecte des données : point d'entrée unique.

Deux modes :
  * import historique d'un dump vérifiable (XLSX de la table de vérification) ;
  * collecteur live de la page publique de vérification (BC.Game).

Les deux passent par le même chemin de validation/déduplication.
"""
from __future__ import annotations

from .cleaner import clean_records, estimate_timestamps, verify_chain
from .historical_import import import_xlsx_dump
from .live import collect_once, run_live_collector

__all__ = [
    "clean_records",
    "estimate_timestamps",
    "verify_chain",
    "import_xlsx_dump",
    "collect_once",
    "run_live_collector",
]
