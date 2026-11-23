#!/usr/bin/env python
"""Backtest walk-forward seul (sans réentraîner via l'UI) + export du rapport."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crashpredict import db
from crashpredict.backtesting import run_walk_forward
from crashpredict.config import CONFIG
from crashpredict.training import load_history


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", type=float, default=CONFIG.targets.fixed)
    ap.add_argument("--models", nargs="*", default=["logistic", "random_forest", "lightgbm"])
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--out", type=str, default="reports/backtest.json")
    args = ap.parse_args()

    db.init_db()
    m, rid, ts = load_history()
    res = run_walk_forward(m, rid, targets=args.targets, models=args.models, max_folds=args.folds)
    print(json.dumps(res.summary, indent=2, default=str))
    piv = res.metrics.pivot_table(index="model", columns="metric", values="value", aggfunc="mean")
    print(piv.to_string())
    if not res.skill.empty:
        print(res.skill[["model", "target", "delta_brier", "ci_low", "ci_high", "p_adjusted_holm", "verdict"]].to_string())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "summary": res.summary,
        "metrics": res.metrics.to_dict(orient="records"),
        "skill": res.skill.to_dict(orient="records"),
    }, indent=2, default=str))
    print(f"Rapport écrit : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
