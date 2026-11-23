"""Confrontation ROBOT V4.5 vs modèles ML vs loi marginale de référence.

Trois comparaisons distinctes, qui ne mesurent pas la même chose — les mélanger
serait la faute méthodologique la plus facile à commettre ici :

1. **Compétence** (`skill_table`) : le taux observé d'une stratégie legacy
   diffère-t-il de la probabilité annoncée par la loi marginale causale pour
   *ces mêmes manches et cette même cible* ? Test apparié, IC bootstrap.
2. **Qualité de l'affirmation** (`claim_quality`) : le pourcentage que le robot
   affiche (`1/o_0`) est-il une probabilité honnête ? On le note au Brier contre
   l'issue réelle, et on le compare au Brier de la référence sur les mêmes
   manches. Un Brier ne se compare pas entre cibles de difficultés différentes :
   il ne sert ici qu'à juger une *affirmation*, pas un classement.
3. **Confrontation au ML** (`ml_vs_reference`) : les modèles du walk-forward
   battent-ils la loi marginale, et de combien, sur les manches où les deux
   générations sont toutes deux évaluables.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..backtesting.significance import holm_correction, paired_skill_test
from ..config import MODEL_DIR
from .backtest import (
    _SPECS,
    declared_probabilities,
    legacy_targets,
    reference_probabilities,
    validation_thresholds,
)


def load_ml_probabilities(path: Path | None = None) -> pd.DataFrame:
    """Probabilités hors échantillon du dernier walk-forward (`models_store/`)."""
    path = Path(path or MODEL_DIR / "last_probabilities.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def ml_window(ml: pd.DataFrame) -> tuple[int, int]:
    """Plage de manches couverte par le walk-forward (comparaison appariée)."""
    return int(ml["round_id"].min()), int(ml["round_id"].max())


# --------------------------------------------------------------------------- #
# 2. Qualité de l'affirmation du robot
# --------------------------------------------------------------------------- #
def claim_quality(
    multipliers: Sequence[float],
    strategy: str = "pro_guide",
    window: int | None = None,
    order: str | None = None,
    horizons: Sequence[int] = (3, 5),
) -> list[dict]:
    """Note le pourcentage affiché par le robot, au Brier, contre l'issue réelle.

    Le robot affiche `1/o_0` (L777/L810) et *présente* ce nombre comme
    « % de chances » que la cible soit atteinte. On le prend au mot : on le
    confronte à `MAX_h >= cible`. La référence (loi marginale) est notée sur
    les mêmes manches, ce qui donne l'échelle.
    """
    from ..config import CONFIG

    spec = _SPECS[strategy]
    m = np.asarray(multipliers, dtype=float)
    w = int(window or CONFIG.legacy.default_window)
    o = order or CONFIG.legacy.default_order
    target = legacy_targets(m, w, spec["mode"], o, spec["exclude_extremes"], spec["trend_analysis"])
    declared = declared_probabilities(m, w, spec["exclude_extremes"], o) / 100.0

    out = []
    for h in horizons:
        thr = validation_thresholds(m, h)
        p_ref = reference_probabilities(m, target, h)
        ok = np.isfinite(target) & np.isfinite(thr) & np.isfinite(declared) & np.isfinite(p_ref)
        ok[: max(w, 0)] = False
        if ok.sum() < 100:
            continue
        y = (thr[ok] >= target[ok]).astype(float)
        p = np.clip(declared[ok], 1e-6, 1 - 1e-6)
        r = p_ref[ok]
        out.append(
            {
                "strategy": strategy,
                "horizon": int(h),
                "n": int(len(y)),
                "observed_rate": float(y.mean()),
                "declared_mean": float(p.mean()),
                "declared_gap": float(y.mean() - p.mean()),
                "brier_declared": float(np.mean((p - y) ** 2)),
                "brier_reference": float(np.mean((r - y) ** 2)),
                "reference_mean": float(r.mean()),
                "mean_target": float(target[ok].mean()),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# 3. Confrontation au ML
# --------------------------------------------------------------------------- #
def ml_vs_reference(
    multipliers: Sequence[float],
    round_ids: Sequence[int] | None = None,
    targets: Sequence[float] = (1.5, 2.0, 2.5, 5.0),
    ml_path: Path | None = None,
    min_n: int = 500,
) -> dict:
    """Les modèles ML battent-ils la loi marginale causale sur leur propre terrain ?

    Comparaison appariée manche par manche, sur la plage du walk-forward et sur
    les cibles qu'il a réellement prédites. La référence est recalculée ici
    (fenêtre glissante causale) plutôt que lue en base, pour garantir que les
    deux probabilités sont produites par le même code.
    """
    ml = load_ml_probabilities(ml_path)
    if ml.empty:
        return {"available": False, "reason": "aucune probabilité ML (lancer un entraînement)"}

    m = np.asarray(multipliers, dtype=float)
    # Les `round_id` BC.Game sont des identifiants (5 822 149, ...), pas des
    # positions : on les convertit en index de position par `searchsorted`.
    if round_ids is None:
        ids = np.arange(len(m), dtype=np.int64)
    else:
        ids = np.asarray(round_ids, dtype=np.int64)
    rows: list[dict] = []
    p_values: dict[str, float] = {}
    for t in sorted({float(x) for x in targets}):
        sub_t = ml[np.isclose(ml["target"], t)]
        if sub_t.empty:
            continue
        ref = reference_probabilities(m, np.full(len(m), float(t)), 3)
        for model, grp in sub_t.groupby("model"):
            rid = grp["round_id"].to_numpy(dtype=np.int64)
            pos = np.searchsorted(ids, rid)
            pos = np.clip(pos, 0, len(ids) - 1)
            mapped = ids[pos] == rid           # ignore toute manche absente de la série
            idx = pos
            y = grp["y"].to_numpy(dtype=float)
            p = np.clip(grp["p"].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
            r = ref[idx]
            ok = np.isfinite(r) & mapped
            if ok.sum() < min_n:
                continue
            y, p, r = y[ok], p[ok], r[ok]
            test = paired_skill_test(y, p, r)
            rows.append(
                {
                    "target": float(t),
                    "model": str(model),
                    "n": int(len(y)),
                    "observed_rate": float(y.mean()),
                    "mean_probability": float(p.mean()),
                    "brier": float(np.mean((p - y) ** 2)),
                    "brier_reference": float(np.mean((r - y) ** 2)),
                    "delta_vs_reference": test["delta_brier"],
                    "ci_low": test["ci_low"],
                    "ci_high": test["ci_high"],
                    "p_value": test["p_value"],
                }
            )
            p_values[f"{model}@{t:g}"] = float(test["p_value"])

    holm = holm_correction(p_values)
    for row in rows:
        key = f"{row['model']}@{row['target']:g}"
        row["p_holm"] = holm.get(key, {}).get("p_adjusted")
        if row["p_holm"] is not None and row["p_holm"] < 0.05:
            row["verdict"] = "meilleur" if row["delta_vs_reference"] > 0 else "pire"
        else:
            row["verdict"] = "aucune différence démontrée"
    return {
        "available": True,
        "horizon": 3,
        "n_rounds": int(len(ml["round_id"].unique())),
        "window": ml_window(ml),
        "table": rows,
        "n_better": sum(1 for r in rows if r["verdict"] == "meilleur"),
        "n_worse": sum(1 for r in rows if r["verdict"] == "pire"),
        "n_null": sum(1 for r in rows if r["verdict"] == "aucune différence démontrée"),
    }


# --------------------------------------------------------------------------- #
# Décomposition de l'illusion « 85–95 % »
# --------------------------------------------------------------------------- #
def labelling_bias_curve(
    multipliers: Sequence[float],
    strategy: str = "pro_guide",
    horizon: int = 5,
    window: int | None = None,
    order: str | None = None,
    honest_failure_rates: Sequence[float] = (1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05),
) -> list[dict]:
    """Pourquoi l'utilisateur voit 85–95 % alors que le taux réel est plus bas.

    Le robot ne se valide pas lui-même : l'utilisateur clique « ✓ Validé » ou
    « ✕ Échoué » (L910-965). Une prédiction qu'il n'a pas clôturée reste
    `pending` et **sort du dénominateur** (`updateStatistics` : succès /
    (succès + échecs), L970-975).

    On simule donc : parmi les prédictions réellement manquées, quelle part
    l'utilisateur prend-il la peine de marquer « échouée » ? Plus cette part est
    faible, plus le taux affiché monte — sans qu'aucune prédiction n'ait été
    falsifiée. C'est la mesure chiffrée du biais de survie.

    Aucune donnée n'est inventée : `succès` et `échecs` sont mesurés.
    """
    from ..config import CONFIG

    spec = _SPECS[strategy]
    m = np.asarray(multipliers, dtype=float)
    w = int(window or CONFIG.legacy.default_window)
    o = order or CONFIG.legacy.default_order
    target = legacy_targets(m, w, spec["mode"], o, spec["exclude_extremes"], spec["trend_analysis"])
    thr = validation_thresholds(m, horizon)
    ok = np.isfinite(target) & np.isfinite(thr)
    ok[:w] = False
    y = (thr[ok] >= target[ok]).astype(float)
    s, f = int(y.sum()), int((1 - y).sum())

    out = []
    for rho in honest_failure_rates:
        closed_failures = rho * f
        rate = s / (s + closed_failures) if (s + closed_failures) else float("nan")
        out.append(
            {
                "honest_failure_rate": float(rho),
                "successes": s,
                "true_failures": f,
                "closed_failures": int(round(closed_failures)),
                "displayed_accuracy": float(rate),
            }
        )
    return out


def round_of_success(
    multipliers: Sequence[float],
    strategy: str = "pro_guide",
    horizon: int = 5,
    window: int | None = None,
    order: str | None = None,
) -> list[dict]:
    """Manche (1..5) à laquelle la cible est atteinte — l'équivalent du menu du robot.

    Permet de vérifier l'affirmation « commencer à partir du 3e round » et de
    reproduire la statistique « Round le plus performant » (L1194-1225), qui est
    dans le robot une simple fréquence descriptive.
    """
    from ..config import CONFIG

    spec = _SPECS[strategy]
    m = np.asarray(multipliers, dtype=float)
    w = int(window or CONFIG.legacy.default_window)
    o = order or CONFIG.legacy.default_order
    target = legacy_targets(m, w, spec["mode"], o, spec["exclude_extremes"], spec["trend_analysis"])
    n = len(m)
    counts = {k: 0 for k in range(1, horizon + 1)}
    missed = 0
    for a in range(w, n - horizon + 1):
        t = target[a]
        if not np.isfinite(t):
            continue
        hit = None
        for k in range(horizon):
            if m[a + k] >= t:
                hit = k + 1
                break
        if hit is None:
            missed += 1
        else:
            counts[hit] += 1
    total = sum(counts.values()) + missed
    return [
        {
            "round": k,
            "successes": v,
            "share_of_predictions": v / total if total else float("nan"),
            "share_of_successes": v / max(sum(counts.values()), 1),
        }
        for k, v in counts.items()
    ] + [{"round": None, "successes": missed, "share_of_predictions": missed / total if total else float("nan"),
          "share_of_successes": 0.0}]
