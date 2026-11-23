"""Nettoyage / validation avant insertion en base.

Règles (aucune donnée n'est modifiée en valeur, seulement acceptée ou rejetée) :
  * multiplier >= 1.0 et fini ;
  * round_id entier > 0 ;
  * dédoublonnage sur round_id (le dernier import ne peut pas écraser un existant) ;
  * ordre chronologique garanti par round_id ;
  * vérification optionnelle de la chaîne provably-fair SHA256(h[i+1]) == h[i].
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable

from ..config import CONFIG


@dataclass
class CleanResult:
    records: list[dict]
    rejected: list[tuple[object, str]]

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)


def clean_records(records: Iterable[dict]) -> CleanResult:
    seen: set[int] = set()
    out: list[dict] = []
    rejected: list[tuple[object, str]] = []

    for rec in records:
        try:
            rid = int(rec["round_id"])
            mult = float(rec["multiplier"])
        except (KeyError, TypeError, ValueError):
            rejected.append((rec.get("round_id"), "champs manquants ou non numériques"))
            continue
        if rid <= 0:
            rejected.append((rid, "round_id invalide"))
            continue
        if not math.isfinite(mult) or mult < CONFIG.rules.min_multiplier:
            rejected.append((rid, f"coefficient invalide ({mult})"))
            continue
        if rid in seen:
            rejected.append((rid, "doublon dans le lot"))
            continue
        seen.add(rid)
        out.append(
            {
                "round_id": rid,
                "multiplier": mult,
                "round_hash": (rec.get("round_hash") or "").strip().lower() or None,
                "ts": rec.get("ts"),
                "ts_source": rec.get("ts_source", "unknown"),
                "source": rec.get("source", "unknown"),
                "collected_at": rec.get("collected_at"),
                "chain_ok": rec.get("chain_ok"),
            }
        )
    out.sort(key=lambda r: r["round_id"])
    return CleanResult(out, rejected)


def verify_chain(records: Iterable[dict]) -> tuple[int, list[int]]:
    """Vérifie SHA256(hash[i+1]) == hash[i] sur les paires consécutives.

    Retourne (nb_paires_testées, liste des round_id en échec).
    Les paires non consécutives (trou) sont ignorées.
    """
    recs = sorted(records, key=lambda r: int(r["round_id"]))
    tested = 0
    bad: list[int] = []
    for a, b in zip(recs, recs[1:]):
        if int(b["round_id"]) != int(a["round_id"]) + 1:
            continue
        if not a.get("round_hash") or not b.get("round_hash"):
            continue
        tested += 1
        if hashlib.sha256(b["round_hash"].encode()).hexdigest() != a["round_hash"]:
            bad.append(int(b["round_id"]))
    return tested, bad


def mark_chain(records: list[dict]) -> list[dict]:
    """Ajoute chain_ok (1/0/None) à chaque enregistrement, paires consécutives seulement."""
    recs = sorted(records, key=lambda r: int(r["round_id"]))
    prev = None
    for rec in recs:
        rec["chain_ok"] = None
        if prev is not None and int(rec["round_id"]) == int(prev["round_id"]) + 1:
            if prev.get("round_hash") and rec.get("round_hash"):
                ok = hashlib.sha256(rec["round_hash"].encode()).hexdigest() == prev["round_hash"]
                rec["chain_ok"] = 1 if ok else 0
        prev = rec
    return recs


def estimate_timestamps(records: list[dict], rounds_per_minute: float = 5.0, anchor_ts: str | None = None) -> list[dict]:
    """Horodatage ESTIMÉ (jamais présenté comme mesuré).

    Le dump historique ne contient pas d'horodatage. On répartit les manches
    selon une cadence moyenne documentée et on marque ts_source='estimated'.
    Les enregistrements déjà horodatés (collecteur live) ne sont pas touchés.
    """
    from datetime import datetime, timedelta, timezone

    step = timedelta(seconds=60.0 / max(rounds_per_minute, 0.01))
    last = max((int(r["round_id"]) for r in records), default=0)
    anchor = datetime.fromisoformat(anchor_ts) if anchor_ts else datetime.now(timezone.utc)
    for rec in records:
        if rec.get("ts"):
            continue
        rec["ts"] = (anchor - step * (last - int(rec["round_id"]))).isoformat(timespec="seconds")
        rec["ts_source"] = "estimated"
    return records
