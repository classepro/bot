#!/usr/bin/env python
"""Import du dump historique vérifié dans la base locale.

Usage:
    python scripts/import_data.py                 # importe data/raw/bcgame/*.csv
    python scripts/import_data.py --xlsx dir/     # importe des XLSX de vérification
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crashpredict import db
from crashpredict.collection.cleaner import clean_records, estimate_timestamps, mark_chain
from crashpredict.config import RAW_DIR


def read_csv_dump(path: Path) -> list[dict]:
    out = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out.append({
                "round_id": int(row["round_id"]),
                "round_hash": (row.get("hash") or "").strip().lower(),
                "multiplier": float(row["multiplier"]),
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="*", help="fichiers CSV normalisés (défaut: data/raw/bcgame/*.csv)")
    ap.add_argument("--xlsx", nargs="*", help="fichiers/dossiers XLSX de la table de vérification")
    ap.add_argument("--rounds-per-minute", type=float, default=5.0, help="cadence pour l'horodatage estimé")
    args = ap.parse_args()

    db.init_db()
    if args.xlsx:
        from crashpredict.collection import import_xlsx_dump
        paths: list[Path] = []
        for p in args.xlsx:
            p = Path(p)
            paths.extend(sorted(p.glob("*.xlsx")) if p.is_dir() else [p])
        report = import_xlsx_dump(paths, rounds_per_minute=args.rounds_per_minute)
        print(report)
        return 0

    files = [Path(p) for p in args.csv] if args.csv else sorted((RAW_DIR / "bcgame").glob("*.csv"))
    if not files:
        print("Aucun fichier à importer.", file=sys.stderr)
        return 2

    total_in = total_skip = 0
    for f in files:
        raw = read_csv_dump(f)
        cleaned = clean_records(raw)
        recs = mark_chain(cleaned.records)
        estimate_timestamps(recs, rounds_per_minute=args.rounds_per_minute)
        for r in recs:
            r["source"] = "bcgame_verify_xlsx"
        ins, skip = db.upsert_rounds(recs)
        bad = sum(1 for r in recs if r["chain_ok"] == 0)
        tested = sum(1 for r in recs if r["chain_ok"] is not None)
        print(f"{f.name}: lues={len(raw)} rejetées={cleaned.n_rejected} insérées={ins} déjà_en_base={skip} "
              f"chaîne_testée={tested} échecs={bad}")
        db.log_ingest("bcgame_verify_xlsx", f"import_csv:{f.name}", ins, skip, "OK" if not bad else "WARN",
                      f"chain_failures={bad}")
        total_in += ins
        total_skip += skip
    fp = db.data_fingerprint()
    print("Total inséré:", total_in, "ignoré:", total_skip)
    print("Base:", fp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
