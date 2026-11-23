"""Moteur de prédiction en direct.

Produit une prédiction vérifiable :
  * identifiant unique ;
  * date/heure de prédiction ;
  * coefficient cible ;
  * probabilité estimée (baseline empirique causale, ou modèle si — et seulement si —
    le backtest a démontré un avantage significatif) ;
  * les 3 manches de validation attendues (round_id + valeurs une fois connues) ;
  * statut EN ATTENTE -> SUCCÈS / ÉCHEC appliqué par `validation/three_rounds.py`.

Le diagnostic (`current_analysis`) est volontairement explicite : il compare la
fréquence observée sur les 500 dernières manches à la loi marginale et indique
si l'écart est compatible avec le hasard (test binomial).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import db
from ..backtesting.significance import binomial_test_vs_rate
from ..config import CONFIG
from ..features import build_features
from ..models.baselines import empirical_validation_probability
from .targets import choose_target_for_success_rate, confidence_level, empirical_survival, probability_grid


@dataclass
class PredictionEngine:
    """Sélectionne la méthode de probabilité la plus honnête disponible."""

    model_name: str = "baseline_rolling_long"

    @classmethod
    def from_storage(cls) -> "PredictionEngine":
        return cls(model_name=db.get_app_config("deployed_model", "baseline_rolling_long") or "baseline_rolling_long")

    # ------------------------------------------------------------------ #
    def history(self, limit: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = db.get_rounds(limit=limit, ascending=False if limit else True)
        if limit:
            rows = list(reversed(rows))
        m = np.array([r["multiplier"] for r in rows], dtype=float)
        rid = np.array([r["round_id"] for r in rows], dtype=int)
        ts = np.array([r["ts"] for r in rows], dtype=object)
        return m, rid, ts

    def probability_for(self, history: np.ndarray, target: float) -> tuple[float, str]:
        """Probabilité de validation + méthode utilisée (toujours traçable)."""
        if self.model_name and not self.model_name.startswith("baseline"):
            p, method = self._model_probability(history, target)
            if np.isfinite(p):
                return p, method
        window = history[-CONFIG.probabilities.long_window:]
        p, n = empirical_validation_probability(window, target)
        return p, f"loi marginale empirique (n={n} manches)"

    def _model_probability(self, history: np.ndarray, target: float) -> tuple[float, str]:
        try:
            from pathlib import Path

            from ..config import MODEL_DIR
            from ..models import build_model

            path = Path(MODEL_DIR) / f"deployed_{self.model_name}_{target}.joblib"
            if not path.exists():
                return float("nan"), ""
            import joblib

            model = joblib.load(path)
            feats = build_features(pd.Series(history))
            X = feats.iloc[[-1]]
            p = float(model.predict_proba(X)[0, 1])
            return p, f"modèle {self.model_name} (calibré, entraîné hors échantillon)"
        except Exception:
            return float("nan"), ""


def current_analysis(engine: PredictionEngine | None = None) -> dict:
    """État du robot : dernières manches, probabilités par cible, séries, tendance."""
    engine = engine or PredictionEngine.from_storage()
    m, rid, ts = engine.history()
    if len(m) == 0:
        return {"error": "aucune donnée en base", "n_rounds": 0}
    last = m[-1]
    window = m[-CONFIG.probabilities.long_window:]
    grid = probability_grid(m)
    optimized = choose_target_for_success_rate(m)

    p_short, n_short = empirical_survival(m[-CONFIG.probabilities.short_window:], 2.0)
    p_long, n_long = empirical_survival(window, 2.0)
    test = binomial_test_vs_rate(int((m[-CONFIG.probabilities.short_window:] >= 2).sum()),
                                 int(n_short), float(p_long))

    streak_low = 0
    for v in m[::-1]:
        if v < 1.5:
            streak_low += 1
        else:
            break
    streak_high = 0
    for v in m[::-1]:
        if v >= 2.0:
            streak_high += 1
        else:
            break

    feats = build_features(pd.Series(m))
    last_feats = feats.iloc[-1]

    return {
        "n_rounds": int(len(m)),
        "last_round_id": int(rid[-1]),
        "last_multiplier": float(last),
        "last_ts": ts[-1] if len(ts) else None,
        "recent_values": [float(v) for v in m[-30:]][::-1],
        "mean_100": float(np.mean(m[-100:])),
        "mean_1000": float(np.mean(m[-1000:])),
        "median_1000": float(np.median(m[-1000:])),
        "share_ge_2_1000": float(np.mean(m[-1000:] >= 2.0)),
        "streak_low_1_5": streak_low,
        "streak_high_2": streak_high,
        "grid": grid,
        "optimized": optimized,
        "drift_test": {
            "target": 2.0,
            "rate_short": p_short,
            "rate_long": p_long,
            "n_short": n_short,
            "n_long": n_long,
            **test,
        },
        "features": {k: (None if pd.isna(v) else float(v)) for k, v in last_feats.items()},
        "engine": engine.model_name,
        "data_fingerprint": db.data_fingerprint(),
    }


def make_prediction(
    target: float | None = None,
    strategy: str | None = None,
    desired_rate: float | None = None,
    persist: bool = True,
    engine: PredictionEngine | None = None,
) -> dict:
    """Crée UNE prédiction vérifiable et la met EN ATTENTE.

    Si `target` est None, la cible est déduite du taux de réussite visé
    (`desired_rate`, sinon CONFIG.strategy.desired_success_rate).
    """
    engine = engine or PredictionEngine.from_storage()
    m, rid, ts = engine.history()
    if len(m) < CONFIG.backtest.min_history_features:
        raise RuntimeError("Historique insuffisant pour produire une prédiction fiable.")

    if target is None:
        chosen = choose_target_for_success_rate(m, desired_rate)
        if chosen["target"] is None:
            raise RuntimeError(chosen.get("reason") or "aucune cible atteignable")
        target = chosen["target"]
        strategy = strategy or "optimisée"

    target = float(target)
    probability, method = engine.probability_for(m, target)
    history_window = m[-CONFIG.probabilities.long_window:]
    p_round, n_used = empirical_survival(history_window, target)
    grid = probability_grid(m)

    anchor = int(rid[-1])
    window_ids = [anchor + k for k in range(1, CONFIG.rules.horizon + 1)]

    explanation = {
        "methode": method,
        "loi_marginale": {
            "cible": target,
            "P(1 manche >= cible)": p_round,
            "P(validation 3 manches)": 1 - (1 - p_round) ** CONFIG.rules.horizon,
            "n_manches_utilisees": n_used,
            "formule": "1 - (1 - P(1 manche))^3",
        },
        "contexte": {
            "derniere_manche": float(m[-1]),
            "moyenne_100": float(np.mean(m[-100:])),
            "part_ge_2_sur_1000": float(np.mean(m[-1000:] >= 2.0)),
            "serie_sous_1_5": int(_streak(m, 1.5, below=True)),
            "serie_au_dessus_2": int(_streak(m, 2.0, below=False)),
        },
        "avertissement": (
            "La probabilité est une fréquence historique de validation, pas une garantie. "
            "Aucun avantage sur le hasard n'a été démontré si le backtest ne l'indique pas."
        ),
        "grille": grid,
    }

    pred = {
        "model_name": method if not method.startswith("loi") else "baseline_rolling_long",
        "model_version": "1.0",
        "strategy": strategy,
        "target": target,
        "probability": probability,
        "status": "PENDING",
        "anchor_round_id": anchor,
        "n_rounds_seen": int(len(m)),
        "window_round_ids": window_ids,
        "window_values": None,
        "explanation": explanation,
        "is_backtest": 0,
    }
    if persist:
        pred["id"] = db.insert_prediction(pred)
    else:
        pred["id"] = "(non persistée)"
    pred["confidence"] = confidence_level(probability)
    pred["created_at"] = db.utcnow()
    return pred


def strategy_table(history: np.ndarray | None = None) -> list[dict]:
    """Les 4 stratégies du cahier des charges + la cible optimisée."""
    if history is None:
        m, _, _ = PredictionEngine.from_storage().history()
    else:
        m = np.asarray(history, dtype=float)
    out = []
    for name, t in (
        ("Prudente", CONFIG.targets.conservative),
        ("Équilibrée", CONFIG.targets.balanced),
        ("Ambitieuse", CONFIG.targets.ambitious),
        ("Très ambitieuse", CONFIG.targets.very_ambitious),
    ):
        p_round, n = empirical_survival(m, t)
        p_val = 1 - (1 - p_round) ** CONFIG.rules.horizon
        out.append({"strategie": name, "cible": t, "probabilité": p_val,
                    "n": n, "succès_attendus": int(round(p_val * n)),
                    "niveau": confidence_level(p_val)})
    opt = choose_target_for_success_rate(m)
    if opt["target"] is not None:
        out.append({"strategie": "Optimisée (80 %)", "cible": opt["target"], "probabilité": opt["probability"],
                    "n": opt["n_rounds"], "succès_attendus": int(round(opt["probability"] * opt["n_rounds"])),
                    "niveau": confidence_level(opt["probability"])})
    return out


def _streak(m: np.ndarray, threshold: float, below: bool) -> int:
    n = 0
    for v in m[::-1]:
        cond = v < threshold if below else v >= threshold
        if cond:
            n += 1
        else:
            break
    return n
