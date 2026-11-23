"""Évaluation expérimentale de ROBOT V4.5 sur les manches réelles.

Protocole (aucune fuite, aucune exception) :

  * pour chaque **manche d'ancrage** `a`, le robot ne reçoit que les cotes des
    manches **strictement antérieures** : `m[a-w .. a-1]` (fenêtre `w`) ;
  * il produit sa « côte probable » `T` avec ses propres formules et son propre
    arrondi à 2 décimales (L783 du fichier d'origine) ;
  * la validation est **automatique**, contrairement au robot d'origine :

        horizon 3 : succès si MAX(m[a], m[a+1], m[a+2]) >= T
        horizon 5 : succès si MAX(m[a] .. m[a+4]) >= T

    (règle officielle du cahier des charges ; la seconde reproduit la promesse
    commerciale « la prédiction apparaît dans les 5 rounds successifs ») ;
  * une manche n'est évaluée que si les `horizon` manches suivantes existent :
    les dernières manches forment la fenêtre de validation, pas l'échantillon.

Aucune écriture en base : le module renvoie des structures sérialisables que le
rapport, l'interface et les tests consomment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..backtesting.significance import binomial_test_vs_rate, wilson_interval
from ..config import CONFIG
from .robotv45 import aggregate_odds, js_to_fixed

# Une « stratégie » = un mode du robot + la politique de mise conseillée.
# Le robot d'origine n'a que trois modes ; les variantes `*_guide*` existent
# parce que son guide recommande explicitement « commencer au 3e round si échec
# aux 2 premiers » et « pari auto avec multiplication de la mise en cas de
# perte ». On les évalue comme des stratégies déclarées, pas comme des astuces.
LEGACY_STRATEGIES: tuple[str, ...] = (
    "standard",
    "advanced",
    "pro",
    "pro_trend",
    "pro_guide",
    "pro_guide_skip2",
    "pro_guide_martingale",
)

_SPECS: dict[str, dict] = {
    # exclude_extremes = True correspond au réglage affiché dans les captures
    # d'usage courant (« Exclure les valeurs extrêmes » coché).
    "standard": {"mode": "standard", "exclude_extremes": True, "trend_analysis": False},
    "advanced": {"mode": "advanced", "exclude_extremes": True, "trend_analysis": False},
    "pro": {"mode": "pro", "exclude_extremes": False, "trend_analysis": False},
    "pro_trend": {"mode": "pro", "exclude_extremes": True, "trend_analysis": True},
    # réglage exact conseillé par le guide du robot : Pro + extrêmes exclus + tendance OFF
    "pro_guide": {"mode": "pro", "exclude_extremes": True, "trend_analysis": False},
    "pro_guide_skip2": {"mode": "pro", "exclude_extremes": True, "trend_analysis": False},
    "pro_guide_martingale": {"mode": "pro", "exclude_extremes": True, "trend_analysis": False},
}


@dataclass
class ModeEvaluation:
    """Résultat d'un mode sur une fenêtre de saisie et un horizon."""

    strategy: str
    window: int
    order: str
    horizon: int
    n_predictions: int
    successes: int
    failures: int
    rate: float
    ci_low: float
    ci_high: float
    expected_rate: float
    mean_target: float
    median_target: float
    share_target_below_1_5: float
    share_target_above_5: float
    declared_rate: float
    identical_to: str | None = None
    gap_vs_reference: float = float("nan")
    gap_ci_low: float = float("nan")
    gap_ci_high: float = float("nan")
    p_value_vs_chance: float = float("nan")
    p_value_vs_reference: float = float("nan")
    p_value_holm: float = float("nan")
    verdict: str = ""

    def as_dict(self) -> dict:
        return {k: _clean(v) for k, v in self.__dict__.items()}


@dataclass
class BettingEvaluation:
    """Politique de mise conseillée, appliquée aux prédictions d'un mode."""

    strategy: str
    horizon: int
    rounds_played: int
    wins: int
    losses: int
    strike_rate: float
    ci_low: float
    ci_high: float
    expected_strike_rate: float
    profit_units: float
    expected_profit_units: float
    max_drawdown: float
    ruin_events: int
    expected_ruin_events: float
    mean_exposure: float

    def as_dict(self) -> dict:
        return {k: _clean(v) for k, v in self.__dict__.items()}


def _clean(v):
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


# --------------------------------------------------------------------------- #
# Vecteurs de calcul
# --------------------------------------------------------------------------- #
def validation_thresholds(multipliers: np.ndarray, horizon: int) -> np.ndarray:
    """MAX des `horizon` manches à partir de l'indice i (NaN si indisponible).

    Balayage unique sur le vecteur, sans matrice intermédiaire : une matrice
    105 000 × 5 coûterait 4 Mo et un `nanmax` sur l'axe 1 pour un résultat
    trivial. `out[i]` est rempli pour tout `i <= n - horizon`.
    """
    n = len(multipliers)
    if n == 0:
        return np.array([], dtype=float)
    out = np.full(n, np.nan)
    size = n - horizon + 1
    if size <= 0:
        return out
    curves = [multipliers[h : h + size] for h in range(horizon)]
    out[:size] = np.fmax.reduce(np.vstack(curves), axis=0)
    return out


def _ordered_window(hist: np.ndarray, order: str) -> list[float]:
    if order == "recent":
        seq = hist[::-1]
    elif order == "chrono":
        seq = hist
    elif order == "asc":
        seq = np.sort(hist)
    else:  # pragma: no cover
        raise ValueError(f"ordre inconnu : {order!r}")
    return [float(v) for v in seq if np.isfinite(v)]


def legacy_targets(
    multipliers: np.ndarray,
    window: int,
    mode: str,
    order: str,
    exclude_extremes: bool,
    trend_analysis: bool,
) -> np.ndarray:
    """Cible du robot pour chaque manche d'ancrage, à partir du passé seul.

    Sortie alignée sur l'indice d'ancrage : la fenêtre utilisée est
    `multipliers[a-window : a]`. NaN si la fenêtre est incomplète, si le robot
    refuse la saisie (< 5 cotes valides, L746) ou s'il ne reste plus rien après
    exclusion des extrêmes.
    """
    n = len(multipliers)
    out = np.full(n, np.nan)
    for a in range(window, n):
        vals = _ordered_window(multipliers[a - window : a], order)
        if len(vals) < CONFIG.legacy.min_inputs:
            continue
        try:
            target, _, _ = aggregate_odds(vals, mode, exclude_extremes, trend_analysis)
        except ValueError:
            continue
        if np.isfinite(target) and target > 1:
            out[a] = js_to_fixed(float(target), 2)
    return out


def declared_probabilities(
    multipliers: np.ndarray, window: int, exclude_extremes: bool, order: str
) -> np.ndarray:
    """`probabilities[0]` (L777/L810) : 1/o_0, c'est-à-dire le pourcentage affiché."""
    n = len(multipliers)
    out = np.full(n, np.nan)
    for a in range(window, n):
        used = _ordered_window(multipliers[a - window : a], order)
        if len(used) < CONFIG.legacy.min_inputs:
            continue
        if exclude_extremes:
            used = sorted(used)[1:-1]
        if not used:
            continue
        out[a] = js_to_fixed(1.0 / used[0] * 100.0, 2)
    return out


def reference_probabilities(multipliers: np.ndarray, targets: np.ndarray, horizon: int) -> np.ndarray:
    """Loi marginale causale : P(MAX_h >= T) avec F estimée sur le passé seul.

        P(validation) = 1 - (1 - F(T))^h        F(T) = P(une manche >= T)

    `F` est la fonction de survie empirique des `reference_window` manches qui
    précèdent la manche d'ancrage. Elle est réévaluée à chaque manche sur la
    cible **courante** (et non sur celle d'une manche voisine) : une cible de
    1,2x et une cible de 8x n'ont évidemment pas la même probabilité.
    Pour rester rapide sur 105 000 manches, la fenêtre glissante est triée une
    fois tous les `reference_refresh` pas puis interrogée exactement
    (`searchsorted`) : l'erreur introduite est celle, infime, d'une fenêtre
    décalée de moins de 5 % de sa longueur.
    """
    w = CONFIG.legacy.reference_window
    refresh = CONFIG.legacy.reference_refresh
    n = len(multipliers)
    out = np.full(n, np.nan)
    sorted_window: np.ndarray | None = None
    for a in range(n):
        if a % refresh == 0:
            hist = multipliers[max(0, a - w) : a]
            hist = hist[np.isfinite(hist)]
            sorted_window = np.sort(hist) if len(hist) >= 100 else None
        if sorted_window is None:
            continue
        t = targets[a]
        if not np.isfinite(t):
            continue
        # F̂(T) = part des manches passées >= T, exactement lue sur la ECDF
        n_ge = len(sorted_window) - int(np.searchsorted(sorted_window, t, side="left"))
        p_round = n_ge / len(sorted_window)
        out[a] = 1 - (1 - p_round) ** horizon
    return out


def start_of_window(multipliers: Sequence[float], window: int, horizon: int) -> int:
    """Première manche exploitable si la série est traitée comme une mise à zéro.

    Convention du rapport : le robot est « branché » à froid, sans historique
    préalable. Ses `window` premières cotes n'existant pas encore, la première
    prédiction possible est la manche d'indice `window`. `horizon` manches sont
    réservées à la fin pour la validation.
    """
    return max(int(window), 0)


def _paired_rate_gap(y: np.ndarray, p_ref: np.ndarray, n_boot: int | None = None,
                     seed: int = 42) -> dict:
    """Écart moyen (succès observé − probabilité de référence) et son IC bootstrap.

    Apparié manche par manche : on mesure si le taux obtenu sur les manches
    réellement prédites diffère de ce que la loi marginale causale annonçait
    pour ces mêmes manches. IC contenant 0 => aucun avantage démontré.
    """
    from ..backtesting.significance import _st, bootstrap_ci

    n_boot = n_boot or CONFIG.legacy.n_boot
    d = np.asarray(y, dtype=float) - np.asarray(p_ref, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 30:
        return {"gap": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan")}
    mean, lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    if _st is not None:
        pval = float(_st.ttest_1samp(d, 0.0, alternative="two-sided").pvalue)
    else:  # pragma: no cover
        se = d.std(ddof=1) / np.sqrt(len(d))
        pval = float(2 * (1 - 0.5 * (1 + math.erf(abs(mean / se) / np.sqrt(2)))))
    return {"gap": float(mean), "ci_low": float(lo), "ci_high": float(hi), "p_value": pval}


# --------------------------------------------------------------------------- #
# Évaluation d'un mode
# --------------------------------------------------------------------------- #
def evaluate_mode(
    multipliers: Sequence[float],
    strategy: str = "standard",
    window: int | None = None,
    order: str | None = None,
    horizon: int | None = None,
    start: int = 0,
    step: int = 1,
) -> ModeEvaluation:
    """Évalue un mode sur une fenêtre de saisie et un horizon donnés.

    `step` permet de ne retenir qu'une manche sur `step` : avec `step=horizon`
    les fenêtres de validation ne se chevauchent plus, ce qui rend les IC
    valides sans hypothèse d'indépendance. `step=1` (défaut) utilise toutes les
    manches — estimateur plus précis, IC légèrement optimistes par
    chevauchement, écart quantifié dans le rapport par comparaison des deux.
    """
    spec = _SPECS.get(strategy)
    if spec is None:
        raise ValueError(f"stratégie inconnue : {strategy!r} (connues : {LEGACY_STRATEGIES})")
    m = np.asarray(multipliers, dtype=float)
    horizon = int(horizon or CONFIG.rules.horizon)
    window = int(window or CONFIG.legacy.default_window)
    order = order or CONFIG.legacy.default_order

    target = legacy_targets(m, window, spec["mode"], order,
                            spec["exclude_extremes"], spec["trend_analysis"])
    threshold = validation_thresholds(m, horizon)
    valid = np.isfinite(target) & np.isfinite(threshold)
    valid[: max(start, window)] = False
    idx = np.flatnonzero(valid)
    if step > 1:
        idx = idx[:: int(step)]
    if len(idx) == 0:
        raise ValueError("aucune prédiction évaluable (historique ou fenêtre insuffisants)")

    t = target[idx]
    y = (threshold[idx] >= t).astype(float)
    n = int(len(y))
    successes = int(y.sum())
    failures = n - successes
    rate, lo, hi = wilson_interval(successes, n)

    p_ref = reference_probabilities(m, target, horizon)[idx]
    ref_ok = np.isfinite(p_ref)

    declared = declared_probabilities(m, window, spec["exclude_extremes"], order)[idx]
    declared_ok = np.isfinite(declared)

    # Le robot n'émet aucune probabilité : sa prévision est un NIVEAU (la cible).
    # On le compare donc à la référence par le taux observé, sous deux formes :
    #   * binomial exact : taux observé vs probabilité moyenne de la référence ;
    #   * bootstrap apparié sur l'écart élémentaire (succès - p_réf), IC 95 %.
    if ref_ok.any():
        p_mean = float(np.mean(p_ref[ref_ok]))
        test_chance = binomial_test_vs_rate(successes, n, min(max(p_mean, 1e-9), 1 - 1e-9))
        skill = _paired_rate_gap(y[ref_ok], p_ref[ref_ok])
    else:  # pragma: no cover
        test_chance = {"p_value": float("nan")}
        skill = {"p_value": float("nan"), "gap": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    identical_to = None
    if strategy == "pro":
        identical_to = "standard"
    elif strategy == "standard":
        identical_to = "pro"

    return ModeEvaluation(
        strategy=strategy,
        window=window,
        order=order,
        horizon=horizon,
        n_predictions=n,
        successes=successes,
        failures=failures,
        rate=rate,
        ci_low=lo,
        ci_high=hi,
        expected_rate=float(np.mean(p_ref[ref_ok])) if ref_ok.any() else float("nan"),
        mean_target=float(np.mean(t)),
        median_target=float(np.median(t)),
        share_target_below_1_5=float(np.mean(t < 1.5)),
        share_target_above_5=float(np.mean(t > 5.0)),
        declared_rate=float(np.mean(declared[declared_ok])) if declared_ok.any() else float("nan"),
        identical_to=identical_to,
        gap_vs_reference=float(skill.get("gap", float("nan"))),
        gap_ci_low=float(skill.get("ci_low", float("nan"))),
        gap_ci_high=float(skill.get("ci_high", float("nan"))),
        p_value_vs_chance=float(test_chance["p_value"]),
        p_value_vs_reference=float(skill.get("p_value", float("nan"))),
    )


# --------------------------------------------------------------------------- #
# Politique de mise conseillée par le guide du robot
# --------------------------------------------------------------------------- #
def _round_survival(multipliers: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """F(T) = P(une manche >= T), estimée sur le passé seul (fenêtre causale).

    Utilisée pour l'espérance théorique de la politique de mise : la marge
    maison doit apparaître dans le calcul, sinon la simulation est fausse.
    """
    w = CONFIG.legacy.reference_window
    refresh = CONFIG.legacy.reference_refresh
    n = len(multipliers)
    out = np.full(n, np.nan)
    sorted_window: np.ndarray | None = None
    for a in range(n):
        if a % refresh == 0:
            hist = multipliers[max(0, a - w) : a]
            hist = hist[np.isfinite(hist)]
            sorted_window = np.sort(hist) if len(hist) >= 100 else None
        if sorted_window is None:
            continue
        t = targets[a]
        if not np.isfinite(t):
            continue
        n_ge = len(sorted_window) - int(np.searchsorted(sorted_window, t, side="left"))
        out[a] = n_ge / len(sorted_window)
    return out


def _betting_drawdown(profits: Sequence[float], stakes: Sequence[float]) -> tuple[float, int]:
    """Décroissance maximale de la bankroll (unités) et nombre de ruines (< 1 %)."""
    bank = 1.0
    peak = 1.0
    max_dd = 0.0
    ruins = 0
    for p in profits:
        bank += p
        peak = max(peak, bank)
        max_dd = max(max_dd, peak - bank)
        if bank <= 0.01:
            ruins += 1
            bank = 1.0
            peak = 1.0
    return max_dd, ruins


def evaluate_betting(
    multipliers: Sequence[float],
    strategy: str = "pro_guide",
    horizon: int | None = None,
    window: int | None = None,
    order: str | None = None,
    skip_first: int = 0,
    martingale: bool = False,
    max_stake_fraction: float = 0.20,
    reference: bool = False,
) -> BettingEvaluation:
    """Simule, manche par manche, la politique de mise recommandée par le guide.

    Modèle de gain — c'est le point où une simulation naïve devient fausse :
    le joueur mise **à chaque manche** de la fenêtre, pas une seule fois.

      * il mise `s` sur la manche 1 ; si elle atteint la cible `T`, il encaisse
        `s·(T−1)` et s'arrête ;
      * sinon il perd `s`, mise à nouveau sur la manche 2, etc. ;
      * après `horizon` manches sans atteinte, il a perdu `horizon · s`.

    Une atteinte au k-ième essai rapporte donc `s·(T−1) − (k−1)·s`. C'est la
    seule façon pour que la marge maison apparaisse : `E[gain] ≈ −marge ×
    nombre de mises`, ce que la version « une seule mise » surestimait.

    Politiques (toutes prescrites explicitement par le guide du robot) :

      * ``skip_first=2`` : « commencer à partir du 3e round si échec aux 2
        premiers » — les 2 premières manches sont observées sans miser ;
      * ``martingale=True`` : « pari auto avec multiplication de la mise en cas
        de perte » — la mise double après chaque échec, plafonnée à
        `max_stake_fraction` de la bankroll courante ;
      * ``reference=True`` : mêmes cibles, même calendrier, mêmes mises, mais
        l'atteinte de chaque manche est tirée avec `F(T)` de la loi marginale
        causale au lieu du résultat réel. **Témoin indispensable** : il mesure
        ce qu'un joueur sans aucune information obtiendrait en visant les
        mêmes cibles.

    Bankroll initiale = 1 ; mise de base = `legacy.bankroll_fraction` (2 %).
    Une **ruine** (< 1 % de la bankroll initiale) est comptée puis la bankroll
    est rechargée : elle est reportée, jamais effacée.
    """
    spec = _SPECS.get(strategy)
    if spec is None:
        raise ValueError(f"stratégie inconnue : {strategy!r}")
    m = np.asarray(multipliers, dtype=float)
    horizon = int(horizon or CONFIG.legacy.horizon_historical)
    window = int(window or CONFIG.legacy.default_window)
    order = order or CONFIG.legacy.default_order

    target = legacy_targets(m, window, spec["mode"], order,
                            spec["exclude_extremes"], spec["trend_analysis"])
    f_round = _round_survival(m, target)
    rng = np.random.default_rng(20261123)
    base = CONFIG.legacy.bankroll_fraction

    profits: list[float] = []
    stakes_used: list[float] = []
    expected: list[float] = []
    bankroll = 1.0
    streak_losses = 0
    played = won = 0
    expected_won = 0.0

    i = window
    while i + horizon <= len(m):
        t = target[i]
        if not np.isfinite(t) or t <= 1:
            i += 1
            continue
        stake = base * (2 ** streak_losses) if martingale else base
        gain = 0.0
        hit_at = None
        rounds = range(i + skip_first, i + horizon)
        for k, r in enumerate(rounds, start=1):
            stake = min(stake, max_stake_fraction * bankroll)
            if stake <= 0:
                break
            if reference:
                p = f_round[i] if np.isfinite(f_round[i]) else 0.0
                hit = bool(rng.random() < p)
                expected.append(stake * (p * (t - 1) - (1 - p)))
            else:
                p = f_round[i] if np.isfinite(f_round[i]) else float("nan")
                hit = bool(m[r] >= t)
                if np.isfinite(p):
                    expected.append(stake * (p * (t - 1) - (1 - p)))
            stakes_used.append(stake)
            if hit:
                gain += stake * (t - 1)
                hit_at = k
                break
            gain -= stake
            stake = stake * 2 if martingale else base
        played += 1
        if hit_at is not None:
            won += 1
        profits.append(gain)
        bankroll = max(bankroll + gain, 0.0)
        if bankroll <= 0.01:
            bankroll = 1.0
        streak_losses = 0 if hit_at is not None else min(streak_losses + 1, CONFIG.legacy.max_attempts - 1)
        i += horizon
        if i + horizon > len(m):
            break

    strike, lo, hi = wilson_interval(won, played)
    max_dd, ruins = _betting_drawdown(profits, stakes_used)

    # taux d'atteinte attendu : 1 - (1-F)^(fenêtre réellement jouée)
    span = max(horizon - skip_first, 1)
    f_ok = f_round[np.isfinite(target) & np.isfinite(f_round)]
    expected_strike = float(np.mean(1 - (1 - f_ok) ** span)) if len(f_ok) else float("nan")

    return BettingEvaluation(
        strategy=strategy,
        horizon=horizon,
        rounds_played=played,
        wins=won,
        losses=played - won,
        strike_rate=strike,
        ci_low=lo,
        ci_high=hi,
        expected_strike_rate=expected_strike,
        profit_units=float(np.sum(profits)),
        expected_profit_units=float(np.sum(expected)) if expected else float("nan"),
        max_drawdown=float(max_dd),
        ruin_events=int(ruins),
        expected_ruin_events=float("nan") if reference else 0.0,
        mean_exposure=float(np.mean(stakes_used)) if stakes_used else 0.0,
    )


def evaluate_policies(multipliers: Sequence[float], horizon: int | None = None) -> list[dict]:
    """Compare les trois politiques de mise du guide à leur témoin sans information."""
    out = []
    for strategy in ("standard", "advanced", "pro", "pro_guide"):
        for label, kw in (
            ("mise à chaque manche", {}),
            ("report au 3e round", {"skip_first": 2}),
            ("martingale (5 essais)", {"martingale": True}),
        ):
            real = evaluate_betting(multipliers, strategy=strategy, horizon=horizon, **kw)
            ref = evaluate_betting(multipliers, strategy=strategy, horizon=horizon, reference=True, **kw)
            d = real.as_dict()
            d["policy"] = label
            d["reference_profit_units"] = ref.profit_units
            d["reference_strike_rate"] = ref.strike_rate
            d["reference_ruin_events"] = ref.ruin_events
            d["reference_max_drawdown"] = ref.max_drawdown
            out.append(d)
    return out
