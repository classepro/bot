"""Reproduction fidèle de ROBOT V4.5 (`legacy/ROBOTV4.5.html`).

Le robot d'origine est une page HTML unique : l'utilisateur COLLE une série de
cotes passées, clique sur « Générer », et lit une « Côte Probable ». Chaque
mode est une agrégation arithmétique de cette série. Il n'y a **ni mémoire, ni
modèle, ni base de données, ni apprentissage** : la seule entrée est la série de
cotes saisie, la seule sortie est un nombre.

Reproduction ligne à ligne (références = numéros de ligne du fichier original) :

    L742-744   oddsArray = input.split(',').map(parseFloat).filter(o => !isNaN(o) && o > 1)
    L746-752   if (oddsArray.length < 5) -> refus (« au moins 5 côtes valides »)
    L755-759   if (excludeExtremes) { sort(asc); slice(1,-1) }
    L762-767   'advanced'  : w_i = 1 + i*0.1  ->  Σ(o_i w_i) / Σ(w_i)
    L768-772   'pro'       : trend = trendAnalysis ? calculateTrend(o) : 1
                             target = mean(o) * trend
    L773-774   'standard'  : target = mean(o)
    L777       probabilities = o.map(odd => (1/odd*100).toFixed(2))
    L783-784   prediction  = { averageOdds: target.toFixed(2), probabilities }
    L821-826   calculateTrend = 1 + (mean(secondHalf) - mean(firstHalf)) / mean(firstHalf)

Trois propriétés du code d'origine sont reproduites telles quelles car elles
conditionnent les résultats observés :

1. **Ordre de saisie non normalisé.** Le mode « Avancé » pondère par *position
   dans le champ de saisie*. Les cotes ne sont triées que si l'option « exclure
   les valeurs extrêmes » est cochée (L757). L'ordre de saisie est donc un
   paramètre à part entière — reproduit ici par `order` et testé en sensibilité.
2. **`probabilities[0]` n'est pas la probabilité de la cible.** L'interface
   affiche `1/o_0` (la PREMIÈRE côte saisie, ou la plus petite si triée), et non
   `1/cible`. Le pourcentage affiché par le robot n'a aucun rapport avec le
   nombre de fois où la cible est effectivement atteinte.
3. **Aucune validation automatique.** « ✓ Validé » est un clic humain suivi du
   choix d'un numéro de manche 1..5 (L910-965). Le taux affiché est
   `succès / (succès + échecs)` sur les seules prédictions que l'utilisateur a
   bien voulu clôturer : c'est une mesure déclarative, non un taux observé.

Le mode « Professionnel » recommandé par le guide du robot est, par défaut
(`trendAnalysis: false`), **arithmétiquement identique** au mode Standard :
`trend` vaut alors exactement 1. Le guide vante donc trois niveaux d'analyse
pour deux formules distinctes. `pro_trend` est fourni séparément pour évaluer
l'option « Analyse tendancielle » quand elle est activée.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Iterable, Sequence

import numpy as np

from ..config import CONFIG

# Modes tels qu'orthographiés en L765/L768/default du `switch` d'origine.
MODES = ("standard", "advanced", "pro")


def js_to_fixed(value: float, digits: int = 2) -> float:
    """Reproduit `Number.prototype.toFixed` du navigateur, au bit près.

    `toFixed` arrondit la valeur binaire EXACTE du double, les égalités étant
    tranchées vers le haut (`(2.675).toFixed(2) === "2.67"` car le double vaut
    2.67499999999999982…). `round()` de Python arrondit au pair et travaille sur
    la représentation décimale courte : les deux diffèrent sur les cas limites.
    On passe donc par `Decimal(float)` (valeur exacte) et un arrondi half-up.
    """
    quantum = Decimal(1).scaleb(-digits)
    return float(Decimal(value).quantize(quantum, rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class LegacyPrediction:
    """Sortie complète d'un clic « Générer » — champ pour champ.

    `displayed_probability` est le `probabilities[0]` affiché par l'interface
    (pourcentage purement décoratif, voir docstring du module).
    """

    target: float                     # « Côte Probable » (arrondie 2 déc.)
    target_raw: float                 # valeur non arrondie
    displayed_probability: float      # probabilities[0] en % (1/o_0 * 100)
    displayed_odds: float             # o_0 : la côte utilisée pour ce %
    mode: str
    exclude_extremes: bool
    trend_applied: float              # 1.0 si « Analyse tendancielle » inactive
    n_inputs_used: int

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Entrée
# --------------------------------------------------------------------------- #
def parse_odds_input(raw: str | Iterable[float]) -> list[float]:
    """L742-744 : découpe, convertit, filtre `!isNaN && > 1`, conserve l'ordre.

    Les cotes à 1.00 sont donc écartées — comportement documenté par le guide du
    robot lui-même (« pour un crash à 1.00, entrez 1.01 »).
    """
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",")]
    else:
        parts = list(raw)
    out: list[float] = []
    for p in parts:
        try:
            v = float(p)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v) and v > 1:
            out.append(v)
    return out


def _exclude_extremes(odds: Sequence[float]) -> list[float]:
    """L755-759 : tri croissant PUIS suppression du min et du max."""
    arr = sorted(float(o) for o in odds)
    return arr[1:-1]


def calculate_trend(odds: Sequence[float]) -> float:
    """L821-826 : `1 + (moyenne(2e moitié) - moyenne(1re moitié)) / moyenne(1re moitié)`."""
    arr = [float(o) for o in odds]
    if len(arr) < 2:
        return 1.0
    cut = len(arr) // 2                      # Math.floor(length / 2)
    first, second = arr[:cut], arr[cut:]
    if not first or not second:
        return 1.0
    first_avg = float(np.mean(first))
    second_avg = float(np.mean(second))
    if first_avg == 0:
        return 1.0
    return 1.0 + (second_avg - first_avg) / first_avg


# --------------------------------------------------------------------------- #
# Agrégation
# --------------------------------------------------------------------------- #
def aggregate_odds(
    odds: Sequence[float],
    mode: str = "standard",
    exclude_extremes: bool = False,
    trend_analysis: bool = False,
) -> tuple[float, float, list[float]]:
    """Cœur du `switch` (L761-775).

    Returns
    -------
    (cible_non_arrondie, trend_appliqué, cotes_effectivement_utilisées)

    Lève `ValueError` si moins de 5 cotes valides (L746-752).
    """
    arr = [float(o) for o in odds]
    if len(arr) < CONFIG.legacy.min_inputs:
        raise ValueError(
            f"au moins {CONFIG.legacy.min_inputs} côtes valides sont requises "
            f"(reçu {len(arr)}) — comportement identique à ROBOT V4.5"
        )
    if exclude_extremes:
        arr = _exclude_extremes(arr)

    if mode == "advanced":
        weights = [1.0 + i * CONFIG.legacy.weight_step for i in range(len(arr))]
        weighted_sum = sum(o * w for o, w in zip(arr, weights))
        target = weighted_sum / sum(weights)
        return target, 1.0, arr

    if mode == "pro":
        trend = calculate_trend(arr) if trend_analysis else 1.0
        return float(np.mean(arr)) * trend, trend, arr

    if mode == "standard":
        return float(np.mean(arr)), 1.0, arr

    raise ValueError(f"mode inconnu : {mode!r} (attendu : {MODES})")


def legacy_target(
    odds: Sequence[float],
    mode: str = "standard",
    exclude_extremes: bool = False,
    trend_analysis: bool = False,
) -> float:
    """« Côte Probable » telle qu'affichée : `toFixed(2)` (L783)."""
    target, _, _ = aggregate_odds(odds, mode, exclude_extremes, trend_analysis)
    return js_to_fixed(target, 2)


def displayed_probability(odds_used: Sequence[float]) -> tuple[float, float]:
    """L777 + L810 : `(1/o_0*100).toFixed(2)` — pourcentage décoratif.

    Retourne (pourcentage affiché, côte o_0 utilisée pour ce pourcentage).
    """
    if not odds_used:
        raise ValueError("aucune côte disponible")
    o0 = float(odds_used[0])
    return js_to_fixed(1.0 / o0 * 100.0, 2), o0


# --------------------------------------------------------------------------- #
# Clic « Générer » complet
# --------------------------------------------------------------------------- #
def predict(
    raw_odds: str | Iterable[float],
    mode: str = "standard",
    exclude_extremes: bool = False,
    trend_analysis: bool = False,
) -> LegacyPrediction:
    """Reproduit un cycle complet : saisie -> filtre -> agrégation -> affichage."""
    arr = parse_odds_input(raw_odds)
    target_raw, trend, used = aggregate_odds(arr, mode, exclude_extremes, trend_analysis)
    pct, o0 = displayed_probability(used)
    return LegacyPrediction(
        target=js_to_fixed(target_raw, 2),
        target_raw=float(target_raw),
        displayed_probability=pct,
        displayed_odds=o0,
        mode=mode,
        exclude_extremes=bool(exclude_extremes),
        trend_applied=float(trend),
        n_inputs_used=len(used),
    )


def prediction_from_sequence(
    history: Sequence[float],
    window: int = 10,
    order: str = "recent",
    mode: str = "standard",
    exclude_extremes: bool = False,
    trend_analysis: bool = False,
) -> LegacyPrediction:
    """Alimente le robot avec les `window` dernières cotes, sans jamais voir la suite.

    `history` doit contenir **uniquement des manches antérieures** (c'est
    l'appelant, dans `backtest.py`, qui garantit l'absence de fuite).

    order
    -----
    ``"recent"``  : la côte la plus récente en premier — un utilisateur qui
                    recopie l'historique de l'écran du haut vers le bas
                    (interprétation par défaut, `CONFIG.legacy.default_order`).
    ``"chrono"``  : la plus ancienne en premier (ordre de lecture temporel).
    ``"asc"``     : cotes triées croissantes. Combiné à `exclude_extremes`, le
                    tri rend le mode « Avancé » déterministe et le rapproche
                    d'une moyenne pondérée par le rang.

    Le mode « Avancé » dépend de cet ordre : c'est précisément l'ambiguïté du
    robot d'origine, testée en sensibilité plutôt que masquée.
    """
    vals = [float(v) for v in history][-int(window):]
    if not vals:
        raise ValueError("historique vide")
    if order == "recent":
        seq = list(reversed(vals))
    elif order == "chrono":
        seq = list(vals)
    elif order == "asc":
        seq = sorted(vals)
    else:
        raise ValueError(f"ordre inconnu : {order!r}")
    return predict(seq, mode=mode, exclude_extremes=exclude_extremes, trend_analysis=trend_analysis)
