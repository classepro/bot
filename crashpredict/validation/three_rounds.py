"""Application de la règle des trois manches — module unique et partagé.

    SUCCÈS si MAX(coefficient manche+1, manche+2, manche+3) >= coefficient cible

Cette fonction est utilisée par le backtest, par la validation des prédictions
en direct et par l'interface : il ne peut donc pas exister deux interprétations
différentes de la règle.
"""
from __future__ import annotations

import json
from typing import Iterable

from .. import db
from ..config import CONFIG
from ..db import utcnow

STATUSES = ("PENDING", "SUCCESS", "FAILURE")


def apply_rule(target: float, values: Iterable[float]) -> tuple[str, float | None]:
    """Retourne (statut, max observé). Statut PENDING si moins de 3 manches."""
    vals = [float(v) for v in values][: CONFIG.rules.horizon]
    if len(vals) < CONFIG.rules.horizon:
        return "PENDING", (max(vals) if vals else None)
    mx = max(vals)
    return ("SUCCESS" if mx >= float(target) else "FAILURE"), mx


def validate_window(round_ids: list[int], multipliers: list[float], target: float) -> dict:
    status, mx = apply_rule(target, multipliers)
    return {
        "status": status,
        "actual_max": mx,
        "window_round_ids": round_ids,
        "window_values": multipliers,
        "validated_at": utcnow() if status != "PENDING" else None,
    }


def close_pending_predictions(db_path=None) -> int:
    """Valide toutes les prédictions EN ATTENTE dont les 3 manches existent en base."""
    from ..config import DB_PATH  # local import : chemin surchargeable en test

    path = db_path or DB_PATH
    with db.connect(path) as conn:
        pending = conn.execute(
            "SELECT * FROM predictions WHERE status='PENDING'"
        ).fetchall()
        updated = 0
        for p in pending:
            anchor = p["anchor_round_id"]
            if anchor is None:
                continue
            win = conn.execute(
                "SELECT round_id, multiplier FROM rounds WHERE round_id > ? ORDER BY round_id ASC LIMIT ?",
                (anchor, CONFIG.rules.horizon),
            ).fetchall()
            if len(win) < CONFIG.rules.horizon:
                continue
            ids = [r["round_id"] for r in win]
            vals = [r["multiplier"] for r in win]
            status, mx = apply_rule(p["target"], vals)
            conn.execute(
                """UPDATE predictions SET status=?, actual_max=?, window_round_ids=?,
                   window_values=?, validated_at=? WHERE id=?""",
                (status, mx, json.dumps(ids), json.dumps(vals), utcnow(), p["id"]),
            )
            updated += 1
    return updated
