"""Choix de la cible : conversion taux de réussite visé -> coefficient.

La cible n'est pas devinée : elle est déduite de la loi marginale estimée sur
le PASSÉ, en inversant la formule de validation sur trois manches :

    P(validation) = 1 - (1 - F(t))^3 = p   =>   F(t) = 1 - (1-p)^(1/3)

Convention d'arrondi : la cible proposée est le plus grand palier standard
(0,05x) dont la probabilité estimée reste >= p visé. Aucune valeur n'est
inventée ; toutes proviennent de l'échantillon et le dénominateur est exposé.
"""
from __future__ import annotations

import numpy as np

from ..config import CONFIG


def empirical_survival(history: np.ndarray, t: float) -> tuple[float, int]:
    h = np.asarray(history, dtype=float)
    h = h[np.isfinite(h)]
    if len(h) == 0:
        return float("nan"), 0
    return float(np.mean(h >= t)), len(h)


def probability_grid(history: np.ndarray, targets: list[float] | None = None) -> list[dict]:
    """Table cible -> probabilité estimée -> historique vérifié (succès/total)."""
    targets = targets or CONFIG.targets.grid
    h = np.asarray(history, dtype=float)
    out = []
    for t in targets:
        p_round, n = empirical_survival(h, t)
        p_val = 1 - (1 - p_round) ** CONFIG.rules.horizon if np.isfinite(p_round) else float("nan")
        out.append(
            {
                "target": t,
                "probability": p_val,
                "p_round": p_round,
                "n_rounds": n,
                "successes_expected": int(round(p_val * n)) if np.isfinite(p_val) else None,
                "label": f"{p_val*100:.1f} %" if np.isfinite(p_val) else "n/d",
            }
        )
    return out


def choose_target_for_success_rate(history: np.ndarray, desired_rate: float | None = None,
                                   step: float = 0.05, minimum: float = 1.05,
                                   maximum: float = 100.0) -> dict:
    """Plus grande cible dont P(validation sur 3 manches) >= desired_rate."""
    desired_rate = desired_rate if desired_rate is not None else CONFIG.strategy.desired_success_rate
    h = np.asarray(history, dtype=float)
    h = h[np.isfinite(h)]
    if len(h) < 50:
        return {"target": None, "probability": None, "n_rounds": len(h), "reason": "historique insuffisant"}
    grid = np.round(np.arange(minimum, maximum + step, step), 2)
    surv = np.array([np.mean(h >= t) for t in grid])
    p_val = 1 - (1 - surv) ** CONFIG.rules.horizon
    ok = np.where(p_val >= desired_rate)[0]
    if len(ok) == 0:
        return {"target": None, "probability": None, "n_rounds": len(h),
                "reason": "aucune cible ne garantit ce taux avec l'historique disponible"}
    idx = ok[-1]
    return {
        "target": float(grid[idx]),
        "probability": float(p_val[idx]),
        "n_rounds": int(len(h)),
        "desired_rate": desired_rate,
        "reason": None,
    }


def confidence_level(probability: float) -> str:
    if probability is None or not np.isfinite(probability):
        return "indéterminé"
    if probability >= CONFIG.strategy.confidence_high:
        return "élevé"
    if probability >= CONFIG.strategy.confidence_medium:
        return "moyen"
    return "faible"
