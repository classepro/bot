#!/usr/bin/env python
"""Entraînement complet + backtest walk-forward + rapport.

Usage:
    python scripts/train.py                       # 4 cibles, 4 modèles, tous les folds
    python scripts/train.py --targets 2 2.5 5
    python scripts/train.py --models lightgbm xgboost catboost
    python scripts/train.py --quick               # 3 folds, pour test rapide
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crashpredict import db
from crashpredict.config import CONFIG
from crashpredict.training import DEFAULT_MODELS, run_training


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", type=float, default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--initial-train", type=int, default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--catboost", action="store_true", help="ajouter CatBoost (plus lent)")
    args = ap.parse_args()

    db.init_db()
    fp = db.data_fingerprint()
    if not fp["n"]:
        print("Base vide : lancez scripts/import_data.py", file=sys.stderr)
        return 2
    print("Données:", fp, flush=True)
    targets = args.targets or CONFIG.targets.fixed
    models = args.models or DEFAULT_MODELS
    if args.catboost and "catboost" not in models:
        models = models + ["catboost"]
    if args.quick:
        args.no_ablation = True
        args.max_folds = args.max_folds or 3
        args.step = args.step or 10000
    try:
        summary = run_training(
            targets=targets, models=models,
            initial_train=args.initial_train, step=args.step, max_folds=args.max_folds,
            with_ablation=not args.no_ablation,
            progress=lambda i, n, msg: print(f"  {msg}", flush=True),
        )
    except KeyboardInterrupt:
        print("Interrompu.")
        return 130
    print("\nRésumé:")
    for k in ("run_id", "n_folds", "deployed_model", "duration_seconds", "first_test_round", "last_test_round"):
        if k in summary:
            print(f"  {k}: {summary[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
