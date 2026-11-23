"""Import d'un dump historique de la table de vérification BC.Game (XLSX).

Format attendu par feuille `Verify Table` :
    Row No | Hash Content | Bang Content      (Bang Content = coefficient, ex. "2.16")

Le module :
  * lit les XLSX (dossier ou liste de fichiers) ;
  * normalise et dédoublonne (cleaner) ;
  * vérifie la chaîne provably-fair SHA256 ;
  * horodate en mode `estimated` si absent ;
  * insère de façon idempotente et journalise dans ingest_log.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from .. import db
from ..config import CONFIG
from .cleaner import clean_records, estimate_timestamps, mark_chain

_MULT_RE = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*[xX]?\s*$")


def _parse_multiplier(raw: object) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    m = _MULT_RE.match(str(raw))
    return float(m.group(1).replace(",", ".")) if m else None


def read_xlsx(path: Path) -> list[dict]:
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb["Verify Table"] if "Verify Table" in wb.sheetnames else wb[wb.sheetnames[0]]
    out: list[dict] = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or row[0] is None:
            continue
        mult = _parse_multiplier(row[2])
        if mult is None:
            continue
        out.append({"round_id": int(row[0]), "round_hash": str(row[1] or "").lower(), "multiplier": mult})
    wb.close()
    return out


def import_xlsx_dump(
    paths: Iterable[Path | str],
    rounds_per_minute: float = 5.0,
    store: bool = True,
    verbose: bool = True,
) -> dict:
    """Importe un ou plusieurs XLSX. `store=False` renvoie seulement le lot nettoyé."""
    files = [Path(p) for p in (paths if isinstance(paths, (list, tuple, set)) else [paths])]
    raw: list[dict] = []
    for f in files:
        raw.extend(read_xlsx(f))

    cleaned = clean_records(raw)
    recs = mark_chain(cleaned.records)
    tested_pairs = sum(1 for r in recs if r["chain_ok"] is not None)
    bad_ids = [int(r["round_id"]) for r in recs if r["chain_ok"] == 0]


    estimate_timestamps(recs, rounds_per_minute=rounds_per_minute)
    for r in recs:
        r["source"] = f"{CONFIG.source_platform.lower()}_verify_xlsx"

    report = {
        "files": [str(f) for f in files],
        "raw_rows": len(raw),
        "accepted": len(recs),
        "rejected": cleaned.n_rejected,
        "rejected_sample": cleaned.rejected[:10],
        "chain_pairs_tested": tested_pairs,
        "chain_failures": len(bad_ids),
        "chain_failure_sample": bad_ids[:20],
    }

    if store:
        inserted, skipped = db.upsert_rounds(recs)
        report.update(inserted=inserted, skipped=skipped)
        db.log_ingest(
            source="bcgame_verify_xlsx",
            action="import_historical",
            inserted=inserted,
            skipped=skipped,
            status="OK" if not bad_ids else "WARN",
            message=json.dumps(report)[:2000],
        )
        db.set_app_config("last_import", report)
    if verbose:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return report
