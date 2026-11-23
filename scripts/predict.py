#!/usr/bin/env python
"""Prédiction en ligne de commande + validation automatique des prédictions en attente.

Usage:
    python scripts/predict.py --target 2.5
    python scripts/predict.py --rate 0.8
    python scripts/predict.py --validate-only
    python scripts/predict.py --strategies
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crashpredict import db
from crashpredict.prediction import current_analysis, make_prediction, strategy_table
from crashpredict.validation import close_pending_predictions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=None, help="coefficient cible (ex. 2.5)")
    ap.add_argument("--rate", type=float, default=None, help="taux de réussite visé (ex. 0.8)")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--strategies", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    db.init_db()
    n = close_pending_predictions()
    if n:
        print(f"{n} prédiction(s) validée(s) sur les 3 manches.")
    if args.validate_only:
        return 0
    if args.strategies:
        for s in strategy_table():
            label = s.get("strategie", s.get("stratégie"))
            print(f"{label:<22} cible {s['cible']:>6.2f}x  probabilité {s['probabilité']*100:>5.1f} %  ({s['niveau']})")
        return 0
    try:
        pred = make_prediction(target=args.target, desired_rate=args.rate)
    except RuntimeError as exc:
        print(f"Erreur: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({k: v for k, v in pred.items() if k != "explanation"}, indent=2, default=str))
    else:
        print(f"ID               : {pred['id']}")
        print(f"Créée le         : {pred['created_at']}")
        print(f"Cible            : {pred['target']:.2f}x")
        print(f"Probabilité      : {pred['probability']*100:.1f} % ({pred['confidence']})")
        print(f"Méthode          : {pred['model_name']}")
        print(f"Ancrage          : manche {pred['anchor_round_id']}")
        print(f"Manches validées : {pred['window_round_ids']}")
        print(f"Statut           : EN ATTENTE (résolution automatique au fil des manches)")
        loi = pred["explanation"]["loi_marginale"]
        print(f"Calcul           : P(1 manche ≥ cible) = {loi['P(1 manche >= cible)']*100:.2f} % sur "
              f"{loi['n_manches_utilisees']} manches → P(3 manches) = {loi['P(validation 3 manches)']*100:.2f} %")
        print("\n" + pred["explanation"]["avertissement"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
