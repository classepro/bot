"""Fixtures : base temporaire peuplée avec des manches réelles (jamais simulées pour les tests de données)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """Base SQLite isolée + 2 000 manches réelles du dump vérifié."""
    import csv

    from crashpredict import db
    from crashpredict.collection.cleaner import clean_records, mark_chain

    path = tmp_path / "test.db"
    import crashpredict.config as cfg

    monkeypatch.setattr(cfg, "DB_PATH", path)  # db.connect résout le chemin à l'appel
    db.init_db(path)

    raw_path = ROOT / "data" / "raw" / "bcgame" / "bcgame_crash_5822149_5927041.csv"
    rows = []
    with open(raw_path, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if i >= 3000:
                break
            rows.append({"round_id": int(row["round_id"]), "round_hash": row["hash"],
                         "multiplier": float(row["multiplier"]), "ts": "2026-01-01T00:00:00+00:00",
                         "ts_source": "estimated", "source": "test"})
    db.upsert_rounds(mark_chain(clean_records(rows).records), path)
    yield path
