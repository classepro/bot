#!/usr/bin/env python
"""Collecteur temps réel BC.Game (page publique de vérification).

Prérequis (uniquement pour ce script) :
    pip install playwright && playwright install chromium

Usage:
    python scripts/collect_live.py --interval 120 --amount 500
    python scripts/collect_live.py --once --amount 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crashpredict import db
from crashpredict.collection import collect_once, run_live_collector


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=120, help="secondes entre deux collectes")
    ap.add_argument("--amount", type=int, default=500, help="manches demandées par requête (max 10 000)")
    ap.add_argument("--iterations", type=int, default=None, help="nombre de cycles (défaut: infini)")
    ap.add_argument("--once", action="store_true", help="une seule collecte")
    ap.add_argument("--visible", action="store_true", help="navigateur visible (débogage)")
    args = ap.parse_args()

    db.init_db()
    if args.once:
        try:
            batch = collect_once(amount=args.amount, headless=not args.visible)
        except Exception as exc:
            print(f"Collecteur indisponible: {exc}", file=sys.stderr)
            return 2
        from crashpredict.collection.cleaner import clean_records

        ins, skip = db.upsert_rounds(clean_records(batch.records).records)
        print(f"collectées={len(batch.records)} insérées={ins} déjà présentes={skip} échecs_chaîne={batch.chain_failures}")
        return 0

    run_live_collector(
        interval_seconds=args.interval,
        amount=args.amount,
        iterations=args.iterations or (1 if args.once else None),
        headless=not args.visible,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
