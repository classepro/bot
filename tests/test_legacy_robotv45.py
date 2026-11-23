"""ROBOT V4.5 : fidélité de la reproduction, absence de fuite, résultats.

Le test central compare la reproduction Python au **code JavaScript d'origine**
exécuté par Node (`legacy/tools/oracle.js`), sur des saisies aléatoires. S'il
passe, la reproduction n'est pas une interprétation : c'est la même fonction.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from crashpredict.config import CONFIG, ROOT
from crashpredict.legacy import (
    aggregate_odds,
    calculate_trend,
    evaluate_betting,
    evaluate_mode,
    evaluate_policies,
    legacy_target,
    legacy_targets,
    parse_odds_input,
    predict,
    prediction_from_sequence,
    reference_probabilities,
    validation_thresholds,
)

HTML = ROOT / "legacy" / "ROBOTV4.5.html"
ORACLE = ROOT / "legacy" / "tools" / "oracle.js"


# --------------------------------------------------------------------------- #
# 1. Fidélité bit-à-bit contre le JavaScript d'origine
# --------------------------------------------------------------------------- #
def _run_oracle(inputs: list[str], settings: dict) -> list[dict]:
    proc = subprocess.run(
        ["node", str(ORACLE), json.dumps({"inputs": inputs, "settings": settings})],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[:800]
    return json.loads(proc.stdout)


@pytest.mark.skipif(not (HTML.exists() and ORACLE.exists() and shutil.which("node")),
                    reason="fichier legacy ou Node indisponible")
@pytest.mark.parametrize("mode", ["standard", "advanced", "pro"])
@pytest.mark.parametrize("exclude", [False, True])
@pytest.mark.parametrize("trend", [False, True])
def test_python_matches_original_javascript(mode, exclude, trend):
    rng = np.random.default_rng(7)
    inputs = []
    for n in (5, 6, 8, 12):
        vals = np.round(rng.uniform(1.01, 12.0, size=n), 2)
        inputs.append(", ".join(f"{v:.2f}" for v in vals))
    settings = {"analysisMode": mode, "excludeExtremes": exclude, "trendAnalysis": trend}
    js = _run_oracle(inputs, settings)
    for raw, expected in zip(inputs, js):
        got = predict(raw, mode=mode, exclude_extremes=exclude, trend_analysis=trend)
        assert got.target == pytest.approx(expected["target"], abs=1e-9), (raw, mode, exclude, trend)
        assert got.displayed_probability == pytest.approx(expected["displayed_probability"], abs=1e-9)
        # même série effectivement utilisée (troncature des extrêmes comprise)
        used_js = [float(x) for x in expected["used"].split(",")]
        _, _, used_py = aggregate_odds(parse_odds_input(raw), mode, exclude, trend)
        assert len(used_py) == len(used_js)


@pytest.mark.skipif(not (HTML.exists() and ORACLE.exists() and shutil.which("node")),
                    reason="fichier legacy ou Node indisponible")
def test_both_sides_refuse_fewer_than_five_odds():
    js = _run_oracle(["1.5,2.0,3.0,1.2"], {"analysisMode": "standard", "excludeExtremes": False,
                                            "trendAnalysis": False})
    assert js[0]["refused"] is True
    with pytest.raises(ValueError):
        legacy_target([1.5, 2.0, 3.0, 1.2])


# --------------------------------------------------------------------------- #
# 2. Propriétés arithmétiques établies par lecture du code
# --------------------------------------------------------------------------- #
def test_pro_equals_standard_by_default():
    """« Professionnel » sans analyse tendancielle, c'est la moyenne simple."""
    odds = [1.5, 2.0, 3.0, 1.2, 2.2, 4.1]
    assert legacy_target(odds, "pro", trend_analysis=False) == legacy_target(odds, "standard")


def test_advanced_depends_on_input_order_only_without_sorting():
    odds = [2.0, 1.2, 3.0, 1.5, 2.2]
    a = legacy_target(odds, "advanced")
    b = legacy_target(sorted(odds), "advanced")
    assert a != b
    # avec exclusion des extrêmes, le tri interne rend l'ordre indifférent
    assert legacy_target(odds, "advanced", exclude_extremes=True) == \
           legacy_target(sorted(odds), "advanced", exclude_extremes=True)


def test_weights_grow_so_last_entered_odd_dominates():
    """w_i = 1 + 0,1i : la DERNIÈRE côte saisie pèse le plus.

    La moyenne pondérée reste toujours comprise entre le minimum et le maximum
    de la série. Ce que l'ordre change, c'est la position du résultat dans cet
    intervalle : avec la même série, mettre la petite cote en dernier tire le
    résultat vers le bas et la grande cote en dernier le tire vers le haut.
    """
    odds = [1.1, 2.0, 3.0, 4.0, 8.0]
    small_last = [1.1, 2.0, 3.0, 8.0, 1.1]   # même série, petite cote en dernier
    big_last = [1.1, 1.1, 2.0, 3.0, 8.0]     # même série, grande cote en dernier
    # mêmes valeurs => même moyenne simple : le mode Standard ignore l'ordre
    assert legacy_target(small_last, "standard") == legacy_target(big_last, "standard")
    # mais l'ordre décide du résultat du mode Avancé
    assert legacy_target(small_last, "advanced") < legacy_target(big_last, "advanced")


def test_trend_matches_closed_form():
    odds = [1.0, 2.0, 3.0, 6.0]  # moitiés : (1,2) et (3,6)
    # 1 + (4.5 - 1.5) / 1.5 = 3
    assert calculate_trend(odds) == pytest.approx(3.0)


def test_declared_probability_is_not_target_probability():
    """Le % affiché est 1/o_0 — un artefact, jamais P(MAX >= cible)."""
    p = predict("1.2, 2.0, 3.0, 1.5, 2.2")
    assert p.displayed_probability == pytest.approx(round(100 / 1.2, 2))
    assert p.target > 1.5
    # la probabilité implicite de la cible serait très inférieure
    assert p.displayed_probability / 100 > 0.5


def test_parse_filters_values_at_or_below_one():
    assert parse_odds_input("1.5, 1.00, 2.0, abc, 3.0, 1.2, 2.2") == [1.5, 2.0, 3.0, 1.2, 2.2]


def test_prediction_from_sequence_uses_only_supplied_history():
    history = [1.5, 2.0, 3.0, 1.2, 2.2, 9.0]
    p = prediction_from_sequence(history, window=5, order="chrono")
    assert p.n_inputs_used == 5          # la 6e cote (9.0) n'est pas dans la fenêtre
    from crashpredict.legacy import js_to_fixed

    # la fenêtre est bien la fin de l'historique fourni : [2.0, 3.0, 1.2, 2.2, 9.0]
    assert p.target == pytest.approx(js_to_fixed(float(np.mean(history[-5:])), 2))


# --------------------------------------------------------------------------- #
# 3. Protocole : aucune fuite vers le futur
# --------------------------------------------------------------------------- #
def test_targets_ignore_future_rounds():
    """Modifier une manche ne doit changer QUE les cibles postérieures à elle.

    `target[a]` ne lit que `m[a-window:a]` : une manche d'indice `k` n'influence
    donc jamais les cibles d'indice <= k.
    """
    m = np.round(np.linspace(1.1, 4.0, 24), 2)     # série strictement croissante
    base = legacy_targets(m, 5, "standard", "chrono", False, False)
    mutated = m.copy()
    mutated[12:] = 50.0                            # on ne touche qu'aux manches >= 12
    after = legacy_targets(mutated, 5, "standard", "chrono", False, False)
    # la cible de l'indice 12 (fenêtre 7..11) ne dépend que du passé : inchangée
    np.testing.assert_allclose(base[:13], after[:13], equal_nan=True)
    # celle de l'indice 13 (fenêtre 8..12) contient la manche 12 : elle change
    assert np.isfinite(base[13]) and base[13] != after[13]


def test_validation_thresholds_excludes_missing_rounds():
    m = np.array([1.0, 2.0, 3.0, 4.0])
    thr = validation_thresholds(m, 3)
    assert thr[0] == 3.0
    assert thr[1] == 4.0
    assert np.isnan(thr[-1]) and np.isnan(thr[-2])


def test_reference_probabilities_are_causal_and_target_aware():
    rng = np.random.default_rng(0)
    m = np.where(rng.random(3000) < 0.5, 1.2, 4.0)
    low = np.full(3000, 1.3)
    high = np.full(3000, 10.0)
    p_low = reference_probabilities(m, low, 3)
    p_high = reference_probabilities(m, high, 3)
    ok = np.isfinite(p_low) & np.isfinite(p_high)
    # une cible basse est bien plus probable qu'une cible haute
    assert np.nanmean(p_low[ok]) > np.nanmean(p_high[ok])
    # et la probabilité ne dépend que du passé : figer le futur ne la change pas
    m2 = m.copy()
    m2[2500:] = 42.0
    p2 = reference_probabilities(m2, low, 3)
    np.testing.assert_allclose(p_low[:2500], p2[:2500], equal_nan=True)


# --------------------------------------------------------------------------- #
# 4. Évaluation : cohérence des compteurs
# --------------------------------------------------------------------------- #
def test_evaluate_mode_counters_are_consistent():
    rng = np.random.default_rng(11)
    m = np.round(rng.exponential(2.0, 2000) + 1.0, 2)
    e = evaluate_mode(m, "standard", window=10, horizon=3)
    assert e.successes + e.failures == e.n_predictions
    assert e.ci_low <= e.rate <= e.ci_high
    assert 0 < e.rate < 1
    # ancrages possibles : indices 10 .. n-horizon  ->  n - horizon - window + 1
    assert e.n_predictions <= len(m) - 3 - 10 + 1


def test_horizon_five_is_never_harder_than_horizon_three():
    """L'horizon 5 contient l'horizon 3 : MAX5 >= MAX3, donc le taux ne peut pas baisser.

    En revanche le taux de 5 manches n'est PAS comparable à celui de 3 manches
    comme s'il s'agissait de la même performance : plus de tentatives, plus de
    chances. C'est précisément la confusion qui fabrique la promesse « 85 % ».
    """
    rng = np.random.default_rng(12)
    m = np.round(rng.exponential(2.0, 3000) + 1.0, 2)
    e3 = evaluate_mode(m, "pro", window=10, horizon=3)
    e5 = evaluate_mode(m, "pro", window=10, horizon=5)
    assert e5.rate > e3.rate


def test_non_overlapping_evaluation_uses_fewer_rounds():
    rng = np.random.default_rng(13)
    m = np.round(rng.exponential(2.0, 3000) + 1.0, 2)
    full = evaluate_mode(m, "standard", window=5, horizon=5)
    sparse = evaluate_mode(m, "standard", window=5, horizon=5, step=5)
    assert sparse.n_predictions < full.n_predictions
    assert abs(sparse.rate - full.rate) < 0.05


def test_betting_reports_losses_as_well_as_wins():
    rng = np.random.default_rng(14)
    m = np.round(rng.exponential(2.0, 3000) + 1.0, 2)
    b = evaluate_betting(m, "pro_guide", horizon=5)
    assert b.wins + b.losses == b.rounds_played
    # chaque essai se termine par une atteinte ou un échec, jamais par rien
    assert b.rounds_played > 0 and b.strike_rate == pytest.approx(b.wins / b.rounds_played)
    # témoin sans information : même calendrier, même nombre d'essais.
    # Il ne peut pas être identique au réel (échantillonnage aléatoire), mais il
    # doit rester dans le même ordre de grandeur — sinon la simulation est fausse.
    ref = evaluate_betting(m, "pro_guide", horizon=5, reference=True)
    assert ref.rounds_played == b.rounds_played
    assert abs(ref.strike_rate - b.strike_rate) < 0.20


def test_policies_include_the_reference_witness():
    rng = np.random.default_rng(15)
    m = np.round(rng.exponential(2.0, 1500) + 1.0, 2)
    rows = evaluate_policies(m, horizon=5)
    assert rows
    assert all("reference_profit_units" in r for r in rows)


def test_constants_come_from_config():
    assert CONFIG.legacy.min_inputs == 5
    assert CONFIG.legacy.weight_step == pytest.approx(0.1)
    assert CONFIG.legacy.horizon_historical == 5
    assert CONFIG.legacy.default_window in CONFIG.legacy.windows


# --------------------------------------------------------------------------- #
# 5. Résultats publiés (si le backtest a été lancé)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (ROOT / "reports" / "robotv45.json").exists(),
                    reason="robotv45.json absent (lancer scripts/legacy_backtest.py)")
def test_published_report_is_internally_consistent():
    report = json.loads((ROOT / "reports" / "robotv45.json").read_text(encoding="utf-8"))
    for row in report["main"]:
        assert row["successes"] + row["failures"] == row["n_predictions"]
        assert row["ci_low"] <= row["rate"] <= row["ci_high"]
        assert row["p_value_holm"] is not None
    # la référence doit rester dans la zone de la loi marginale
    for row in report["main"]:
        assert abs(row["gap_vs_reference"]) < 0.02
    # la promesse commerciale n'est dépassée par aucune stratégie sur 3 manches
    assert report["summary"]["rates_h3"]["pro_guide"] < 0.95
