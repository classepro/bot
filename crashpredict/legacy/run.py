"""Exécution complète de l'évaluation ROBOT V4.5 et persistance des résultats.

Un seul point d'entrée : `run_full_evaluation()` produit un dictionnaire
sérialisable (JSON) contenant

  * le protocole exact (constantes utilisées, aucune valeur implicite) ;
  * les trois modes, sous les deux horizons, avec effectifs / succès / échecs /
    taux / IC de Wilson / écart à la référence et sa p-value ;
  * la sensibilité (fenêtre de saisie, ordre de saisie, sous-périodes, pas de
    sous-échantillonnage) et la correction de Holm sur toute cette famille ;
  * la politique de mise conseillée par le guide, avec son témoin ;
  * la décomposition chiffrée de l'illusion « 85–95 % » ;
  * la confrontation aux modèles ML et à la loi marginale.

Le résultat est écrit en base (table `metrics`, run dédié) **et** dans
`reports/robotv45.json`, afin que le rapport, l'interface et les tests lisent
tous la même vérité.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from .. import db
from ..backtesting.significance import holm_correction
from ..config import CONFIG, REPORT_DIR
from .backtest import (
    LEGACY_STRATEGIES,
    declared_probabilities,
    evaluate_betting,
    evaluate_mode,
    evaluate_policies,
    legacy_targets,
    reference_probabilities,
    start_of_window,
    validation_thresholds,
)
from .compare import claim_quality, labelling_bias_curve, ml_vs_reference, round_of_success

RUN_KIND = "ROBOTV45"
JSON_PATH = REPORT_DIR / "robotv45.json"

# Modes affichés en priorité dans le rapport (les variantes de mise suivent).
MAIN_STRATEGIES = ("standard", "advanced", "pro", "pro_trend", "pro_guide")


def load_series(db_path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Série complète (multiplicateurs, identifiants) en ordre chronologique strict."""
    import sqlite3

    path = Path(db_path or CONFIG.db_path)
    conn = sqlite3.connect(path)
    try:
        n = int(conn.execute("SELECT COUNT(*) FROM rounds").fetchone()[0])
        m = np.empty(n, dtype=float)
        ids = np.empty(n, dtype=np.int64)
        for i, (mult, rid) in enumerate(conn.execute("SELECT multiplier, round_id FROM rounds ORDER BY round_id")):
            m[i] = mult
            ids[i] = rid
    finally:
        conn.close()
    return m, ids


# --------------------------------------------------------------------------- #
def run_full_evaluation(
    multipliers: Sequence[float] | None = None,
    round_ids: Sequence[int] | None = None,
    window: int | None = None,
    order: str | None = None,
    with_ml: bool = True,
    with_sensitivity: bool = True,
    db_path: str | Path | None = None,
) -> dict:
    """Évalue toutes les stratégies légacy et renvoie le rapport complet."""
    started = time.time()
    if multipliers is None:
        multipliers, round_ids = load_series(db_path)
    m = np.asarray(multipliers, dtype=float)
    window = int(window or CONFIG.legacy.default_window)
    order = order or CONFIG.legacy.default_order
    h3, h5 = CONFIG.rules.horizon, CONFIG.legacy.horizon_historical

    protocol = {
        "n_rounds": int(len(m)),
        "first_round": int(round_ids[0]) if round_ids is not None else None,
        "last_round": int(round_ids[-1]) if round_ids is not None else None,
        "window": window,
        "order": order,
        "horizons": [h3, h5],
        "min_inputs": CONFIG.legacy.min_inputs,
        "weight_step": CONFIG.legacy.weight_step,
        "first_anchor_round": int(round_ids[window]) if round_ids is not None and len(round_ids) > window else window,
        "leakage": "cible = agrégat de m[a-window:a] uniquement ; validation = m[a:a+horizon]",
    }

    # ---- 1. modes principaux, deux horizons -------------------------------- #
    main: list[dict] = []
    p_values: dict[str, float] = {}
    for h in (h3, h5):
        for strategy in LEGACY_STRATEGIES:
            eval_ = evaluate_mode(m, strategy, window=window, order=order, horizon=h)
            row = eval_.as_dict()
            row["horizon"] = h
            main.append(row)
            key = f"{strategy}@{h}"
            p_values[key] = float(eval_.p_value_vs_reference)

    holm = holm_correction(p_values)
    for row in main:
        res = holm.get(f"{row['strategy']}@{row['horizon']}", {})
        row["p_value_holm"] = res.get("p_adjusted")
        row["verdict"] = _verdict(row, res.get("p_adjusted"))

    # ---- 2. sensibilité ----------------------------------------------------- #
    sensitivity: list[dict] = []
    if with_sensitivity:
        for h in (h3, h5):
            for w in CONFIG.legacy.windows:
                for o in ("recent", "chrono", "asc"):
                    for strategy in MAIN_STRATEGIES:
                        e = evaluate_mode(m, strategy, window=w, order=o, horizon=h)
                        d = e.as_dict()
                        d.update({"horizon": h, "window": w, "order": o})
                        sensitivity.append(d)
        sens_p = {f"{d['strategy']}@{d['horizon']}@{d['window']}@{d['order']}": float(d["p_value_vs_reference"])
                  for d in sensitivity}
        sens_holm = holm_correction(sens_p)
        for d in sensitivity:
            key = f"{d['strategy']}@{d['horizon']}@{d['window']}@{d['order']}"
            adj = sens_holm.get(key, {}).get("p_adjusted")
            d["p_value_holm_sensitivity"] = adj
            d["verdict"] = _verdict(d, adj)

    # ---- 3. stabilité temporelle : trois sous-périodes ---------------------- #
    periods: list[dict] = []
    thirds = len(m) // 3
    for label, lo, hi in (("début", 0, thirds), ("milieu", thirds, 2 * thirds), ("fin", 2 * thirds, len(m))):
        for strategy in ("standard", "advanced", "pro"):
            for h in (h3, h5):
                sub = m[lo:hi]
                try:
                    e = evaluate_mode(sub, strategy, window=window, order=order, horizon=h)
                except ValueError:
                    continue
                d = e.as_dict()
                d.update({"period": label, "horizon": h, "from": int(lo), "to": int(hi)})
                periods.append(d)

    # ---- 4. checkpoints non chevauchants (IC valides) ---------------------- #
    non_overlapping: list[dict] = []
    for h in (h3, h5):
        for strategy in MAIN_STRATEGIES:
            e = evaluate_mode(m, strategy, window=window, order=order, horizon=h, step=h)
            d = e.as_dict()
            d["horizon"] = h
            non_overlapping.append(d)

    # ---- 5. politique de mise + témoin ------------------------------------ #
    betting = evaluate_policies(m, horizon=h5)
    betting_simple = []
    for strategy in ("standard", "advanced", "pro_guide"):
        real = evaluate_betting(m, strategy=strategy, horizon=h5)
        ref = evaluate_betting(m, strategy=strategy, horizon=h5, reference=True)
        d = real.as_dict()
        d["reference_profit_units"] = ref.profit_units
        d["reference_strike_rate"] = ref.strike_rate
        d["reference_max_drawdown"] = ref.max_drawdown
        betting_simple.append(d)
    for row in betting:
        row["reference_max_drawdown"] = row.get("reference_max_drawdown", None)

    # ---- 6. illusion 85–95 % ---------------------------------------------- #
    illusion = {
        "labelling": labelling_bias_curve(m, "pro_guide", horizon=h5, window=window, order=order),
        "labelling_h3": labelling_bias_curve(m, "pro_guide", horizon=h3, window=window, order=order),
        "rounds": round_of_success(m, "pro_guide", horizon=h5, window=window, order=order),
        "claim": claim_quality(m, "pro_guide", window=window, order=order, horizons=(h3, h5)),
    }

    # ---- 7. comparaison ML ------------------------------------------------- #
    ml = ml_vs_reference(m, round_ids, targets=CONFIG.targets.fixed) if with_ml else {"available": False}

    # ---- 8. recouvrement avec le walk-forward ----------------------------- #
    overlap = _overlap_with_walk_forward(m, round_ids, window, order)

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": round(time.time() - started, 1),
        "protocol": protocol,
        "main": main,
        "sensitivity": sensitivity,
        "periods": periods,
        "non_overlapping": non_overlapping,
        "betting": betting,
        "betting_simple": betting_simple,
        "illusion": illusion,
        "ml": ml,
        "overlap": overlap,
        "summary": _summary(main, non_overlapping, ml, illusion),
    }
    return out


def _verdict(row: dict, p_holm: float | None) -> str:
    """Phrase de conclusion, dépendante uniquement des chiffres."""
    gap = row.get("gap_vs_reference")
    if gap is None:
        return "non mesurable"
    if p_holm is not None and p_holm < 0.05:
        return "meilleur que la loi marginale (significatif)" if gap > 0 else \
               "pire que la loi marginale (significatif)"
    return "aucun avantage démontré (IC 95 % contient 0)"


def _overlap_with_walk_forward(m, round_ids, window: int, order: str) -> dict:
    """Part des manches du walk-forward réellement couvertes par l'évaluation.

    Le run ML a été conduit sur les manches 5 887 042 → 5 927 038 (60 000
    manches d'entraînement initial), alors que l'évaluation légacy couvre toute
    la série. Ce bloc chiffre explicitement le recouvrement pour que la
    comparaison ne soit pas présentée comme portant sur 104 893 manches.
    """
    from .compare import load_ml_probabilities, ml_window

    ml = load_ml_probabilities()
    if ml.empty:
        return {"available": False}
    lo, hi = ml_window(ml)
    inside = (round_ids >= lo) & (round_ids <= hi)
    n_inside = int(inside.sum())
    return {
        "available": True,
        "ml_window": [int(lo), int(hi)],
        "ml_rounds": n_inside,
        "share_of_series": n_inside / len(m),
        "legacy_rounds": int(len(m)),
    }


def _summary(main, non_overlapping, ml, illusion) -> dict:
    """Synthèse factuelle : ce qui est démontré, ce qui ne l'est pas."""
    h3 = [r for r in main if r["horizon"] == 3]
    h5 = [r for r in main if r["horizon"] == 5]
    better = [r["strategy"] for r in main if r.get("verdict", "").startswith("meilleur")]
    worse = [r["strategy"] for r in main if r.get("verdict", "").startswith("pire")]
    non_over_better = [r["strategy"] for r in non_overlapping if r.get("verdict", "").startswith("meilleur")]
    rates_h3 = {r["strategy"]: r["rate"] for r in h3}
    rates_h5 = {r["strategy"]: r["rate"] for r in h5}
    claim = illusion.get("claim", [])
    return {
        "strategies_evaluated": len(main),
        "better_significant": sorted(set(better)),
        "worse_significant": sorted(set(worse)),
        "better_significant_non_overlapping": sorted(set(non_over_better)),
        "rates_h3": rates_h3,
        "rates_h5": rates_h5,
        "ml_better": ml.get("n_better"),
        "ml_worse": ml.get("n_worse"),
        "ml_null": ml.get("n_null"),
        "declared_gap_h3": next((c["declared_gap"] for c in claim if c["horizon"] == 3), None),
        "declared_gap_h5": next((c["declared_gap"] for c in claim if c["horizon"] == 5), None),
        "conclusion": _conclusion(better, non_over_better, ml),
    }


def _conclusion(better: list[str], non_over: list[str], ml: dict) -> str:
    if better and non_over:
        return ("Au moins une stratégie ROBOT V4.5 conserve un avantage significatif "
                "hors chevauchement : à étudier comme candidat.")
    if better:
        return ("Les seuls avantages significatifs disparaissent quand les fenêtres de "
                "validation ne se chevauchent plus : signature du hasard, pas d'un edge.")
    return ("Aucune stratégie ROBOT V4.5 ne bat la loi marginale de façon démontrée ; "
            "leur taux de réussite est entièrement expliqué par le niveau de cible choisi.")


# --------------------------------------------------------------------------- #
def persist(report: dict, json_path: Path | None = None, db_path: str | Path | None = None) -> str:
    """Écrit le JSON et enregistre le run dans les tables `backtest_runs`/`metrics`."""
    path = Path(json_path or JSON_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    run_id = "R" + format(abs(hash(report["generated_at"] + str(report["protocol"]["n_rounds"]))) % (16 ** 16), "016X")
    fingerprint = db.data_fingerprint(db_path)
    with db.connect(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO backtest_runs (id, created_at, finished_at, status, config, summary, data_fingerprint)
               VALUES (?,?,?,?,?,?,?)""",
            (
                run_id,
                report["generated_at"],
                report["generated_at"],
                "DONE",
                json.dumps({"kind": RUN_KIND, **report["protocol"]}, ensure_ascii=False),
                json.dumps(report["summary"], ensure_ascii=False),
                json.dumps(fingerprint),
            ),
        )
        conn.execute("DELETE FROM metrics WHERE run_id=?", (run_id,))
        rows = []
        for r in report["main"]:
            for metric, value in (
                ("n", r["n_predictions"]),
                ("successes", r["successes"]),
                ("failures", r["failures"]),
                ("rate", r["rate"]),
                ("expected_rate", r["expected_rate"]),
                ("gap_vs_reference", r["gap_vs_reference"]),
                ("gap_ci_low", r["gap_ci_low"]),
                ("gap_ci_high", r["gap_ci_high"]),
                ("p_value", r["p_value_vs_reference"]),
                ("p_value_holm", r["p_value_holm"]),
                ("mean_target", r["mean_target"]),
                ("declared_rate", r["declared_rate"]),
            ):
                if value is None:
                    continue
                rows.append((run_id, r["strategy"], float(r["horizon"]), metric, float(value),
                             r["ci_low"], r["ci_high"], r["n_predictions"], "legacy"))
        for r in report.get("ml", {}).get("table", []):
            for metric, value in (
                ("brier", r["brier"]),
                ("brier_reference", r["brier_reference"]),
                ("delta_brier_vs_reference", r["delta_vs_reference"]),
                ("p_value_holm", r["p_holm"]),
            ):
                if value is None:
                    continue
                rows.append((run_id, r["model"], float(r["target"]), metric, float(value),
                             r["ci_low"], r["ci_high"], r["n"], "ml"))
        conn.executemany(
            """INSERT INTO metrics (run_id, model_name, target, metric, value, ci_low, ci_high, n, segment)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        conn.execute(
            "INSERT OR REPLACE INTO app_config(key, value, updated_at) VALUES (?,?,?)",
            ("last_legacy_run", run_id, db.utcnow()),
        )
    return run_id


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Évalue ROBOT V4.5 sur les manches réelles")
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--order", default=None, choices=("recent", "chrono", "asc"))
    ap.add_argument("--no-ml", action="store_true")
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--no-persist", action="store_true")
    ap.add_argument("--quick", action="store_true", help="échantillon réduit (vérification seulement)")
    args = ap.parse_args(argv)

    db.init_db()
    if args.quick:
        # échantillon de 20 000 manches : sert à vérifier la chaîne, jamais à publier
        mult, ids = load_series()
        cut = 20_000
        mult, ids = mult[-cut:], ids[-cut:]
        report = run_full_evaluation(mult, ids, window=args.window, order=args.order,
                                     with_ml=False, with_sensitivity=False)
        report["protocol"]["quick_sample_rounds"] = len(mult)
        report["protocol"]["note"] = "échantillon réduit (mode --quick), ne pas publier"
        print("MODE RAPIDE : échantillon réduit, résultats non publiables")
    else:
        report = run_full_evaluation(
            window=args.window,
            order=args.order,
            with_ml=not args.no_ml,
            with_sensitivity=not args.no_sensitivity,
        )
    if not args.no_persist:
        run_id = persist(report)
        print(f"Run légacy enregistré : {run_id}")
    print(f"JSON : {JSON_PATH}")
    s = report["summary"]
    for h, rates in (("3", s["rates_h3"]), ("5", s["rates_h5"])):
        print(f"--- horizon {h} manches ---")
        for k, v in rates.items():
            print(f"   {k:22s} {v*100:6.2f} %")
    print("meilleurs (significatif) :", s["better_significant"] or "aucun")
    print("pires (significatif)     :", s["worse_significant"] or "aucun")
    print("conclusion :", s["conclusion"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
