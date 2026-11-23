"""ROBOT V4.5 : ce que la génération précédente faisait, et ce qu'elle vaut.

Cette page ne prend aucune décision : elle lit `reports/robotv45.json` (produit
par `scripts/legacy_backtest.py`) et l'affiche. Un bouton permet de recalculer
l'évaluation à la demande.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from ... import db
from ...config import CONFIG, REPORT_DIR
from ...legacy import LEGACY_STRATEGIES, predict, round_of_success
from ..components import dataframe, kpi_row, pct, warn

LABELS = {
    "standard": "Standard — moyenne simple",
    "advanced": "Avancé — pondération 1 + 0,1·i",
    "pro": "Professionnel — sans tendance (= Standard)",
    "pro_trend": "Professionnel — avec analyse tendancielle",
    "pro_guide": "Réglage conseillé par le guide",
    "pro_guide_skip2": "Guide + report au 3e round",
    "pro_guide_martingale": "Guide + martingale",
}


def _load() -> dict | None:
    path = REPORT_DIR / "robotv45.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def render() -> None:
    st.markdown(
        "<div class='cp-hero'><h1>🤖 ROBOT V4.5 — génération précédente</h1>"
        "<p>Rétro-ingénierie du robot d'origine, reproduction fidèle, et évaluation "
        "expérimentale sur les mêmes manches que les modèles actuels.</p></div>",
        unsafe_allow_html=True,
    )

    report = _load()
    left, right = st.columns([1, 2])
    with left:
        if st.button("🔄 Recalculer l'évaluation complète", use_container_width=True):
            with st.spinner("Évaluation des 7 stratégies sous 2 horizons (≈ 9 min)…"):
                from ...legacy.run import persist, run_full_evaluation

                prod = run_full_evaluation()
                persist(prod)
            st.success("Évaluation terminée.")
            report = _load()
    with right:
        st.caption(
            "Le calcul est identique à `scripts/legacy_backtest.py` ; le résultat est écrit "
            "dans `reports/robotv45.json` et dans la table `metrics` (run dédié)."
        )

    if not report:
        warn("Aucune évaluation enregistrée. Lancez <code>scripts/legacy_backtest.py</code> "
             "ou cliquez sur le bouton ci-dessus.", "warn")
        return

    # ---- ce que le robot fait réellement ---------------------------------- #
    st.markdown("#### Ce que le robot calcule (reproduit ligne à ligne)")
    st.markdown(
        """
| Mode | Formule exacte | Remarque issue du code d'origine |
|---|---|---|
| **Standard** | `moyenne(cotes)` | ignore l'ordre de saisie |
| **Avancé** | `Σ(cᵢ·(1+0,1·i)) / Σ(1+0,1·i)` | pondère selon la **position dans le champ de saisie** ; les cotes ne sont triées que si « exclure les extrêmes » est coché |
| **Professionnel** | `moyenne(cotes) × tendance` | avec `trendAnalysis: false` (le défaut), `tendance = 1` ⇒ **arithmétiquement identique au mode Standard** |

Le pourcentage affiché par l'interface est `1/o₀` (la première côte saisie), **pas** la
probabilité que la cible soit atteinte. La validation est un clic humain (« ✓ Validé », puis
un numéro de manche de 1 à 5) : le robot ne vérifie jamais le résultat lui-même.
        """
    )

    with st.expander("🧪 Essayer le robot d'origine sur une saisie", expanded=False):
        raw = st.text_input("Cotes séparées par des virgules (≥ 5)", "2.10, 1.45, 3.20, 1.80, 2.60")
        c1, c2, c3 = st.columns(3)
        mode = c1.selectbox("Mode", ["standard", "advanced", "pro"], format_func=lambda m: LABELS[m])
        exclude = c2.checkbox("Exclure les valeurs extrêmes", value=True)
        trend = c3.checkbox("Analyse tendancielle", value=False)
        try:
            p = predict(raw, mode=mode, exclude_extremes=exclude, trend_analysis=trend)
            kpi_row(
                [
                    {"label": "Côte probable", "value": f"{p.target:.2f}x", "accent": "#0984E3"},
                    {"label": "% affiché par le robot", "value": f"{p.displayed_probability:.2f} %",
                     "sub": f"= 1/{p.displayed_odds:g}", "accent": "#E17055"},
                    {"label": "Cotes utilisées", "value": str(p.n_inputs_used),
                     "sub": f"tendance appliquée : ×{p.trend_applied:.3f}"},
                ]
            )
            st.caption(
                "Ce « % affiché » est un artefact d'interface : il ne décrit pas la probabilité "
                "que la cible 3.20x soit atteinte dans les manches suivantes."
            )
        except ValueError as exc:
            warn(str(exc), "warn")

    # ---- résultats mesurés ------------------------------------------------- #
    p = report["protocol"]
    st.markdown("#### Résultats mesurés sur les manches réelles")
    st.caption(
        f"{p['n_rounds']:,} manches ({p['first_round']} → {p['last_round']}) · fenêtre de saisie "
        f"{p['window']} cotes passées · ordre « {p['order']} » · première prédiction manche "
        f"{p['first_anchor_round']} · aucune manche future utilisée.".replace(",", " ")
    )

    horizon = st.radio("Règle de validation", [3, 5], horizontal=True,
                       format_func=lambda h: ("3 manches suivantes (règle du cahier des charges)"
                                              if h == 3 else "5 manches suivantes (promesse historique du robot)"))
    rows = [r for r in report["main"] if r["horizon"] == horizon]
    table = pd.DataFrame(
        {
            "Stratégie": [LABELS.get(r["strategy"], r["strategy"]) for r in rows],
            "Prédictions": [r["n_predictions"] for r in rows],
            "Succès": [r["successes"] for r in rows],
            "Échecs": [r["failures"] for r in rows],
            "Taux": [r["rate"] for r in rows],
            "IC 95 % bas": [r["ci_low"] for r in rows],
            "IC 95 % haut": [r["ci_high"] for r in rows],
            "Loi marginale": [r["expected_rate"] for r in rows],
            "Écart (pts)": [r["gap_vs_reference"] * 100 for r in rows],
            "p (Holm)": [r["p_value_holm"] for r in rows],
            "Verdict": [r["verdict"] for r in rows],
        }
    )
    dataframe(
        table,
        column_config={
            "Taux": st.column_config.NumberColumn(format="%.2f %%"),
            "IC 95 % bas": st.column_config.NumberColumn(format="%.2f %%"),
            "IC 95 % haut": st.column_config.NumberColumn(format="%.2f %%"),
            "Loi marginale": st.column_config.NumberColumn(format="%.2f %%"),
            "Écart (pts)": st.column_config.NumberColumn(format="%+.2f"),
            "p (Holm)": st.column_config.NumberColumn(format="%.4f"),
        },
        hide_index=True,
    )

    best = max(rows, key=lambda r: r["rate"])
    kpi_row(
        [
            {"label": "Meilleure stratégie", "value": f"{best['rate']*100:.1f} %",
             "sub": LABELS.get(best["strategy"], best["strategy"]), "accent": "#00B894"},
            {"label": "Loi marginale", "value": f"{best['expected_rate']*100:.1f} %",
             "sub": "même cible, aucune information"},
            {"label": "Promesse du guide", "value": "85–95 %",
             "sub": "à 3 manches : non reproduite", "accent": "#E17055"},
        ]
    )

    # ---- l'illusion -------------------------------------------------------- #
    st.markdown("#### D'où viennent les 85–95 % annoncés")
    lab = pd.DataFrame(
        {
            "Part des échecs marqués « échoué »": [x["honest_failure_rate"] for x in report["illusion"]["labelling"]],
            "Taux affiché (5 manches)": [x["displayed_accuracy"] for x in report["illusion"]["labelling"]],
            "Taux affiché (3 manches)": [x["displayed_accuracy"] for x in report["illusion"]["labelling_h3"]],
        }
    )
    st.line_chart(lab.set_index("Part des échecs marqués « échoué »"))
    st.caption(
        "Le robot ne clôture que ce que l'utilisateur décide de clôturer. Une prédiction laissée "
        "en attente sort du dénominateur. En ne marquant comme échoués qu'une partie des échecs "
        "réels — sans rien falsifier — on obtient exactement la fourchette annoncée."
    )

    rounds = pd.DataFrame([r for r in report["illusion"]["rounds"] if r["round"]])
    if not rounds.empty:
        st.markdown("##### Manche à laquelle la cible est atteinte (conseil « commencer au 3e round »)")
        st.bar_chart(rounds.set_index("round")["successes"])
        miss = next((r for r in report["illusion"]["rounds"] if r["round"] is None), None)
        if miss:
            st.caption(
                f"{rounds['successes'].sum():,} atteintes réparties sur les 5 manches contre "
                f"{miss['successes']:,} échecs complets. Attendre le 3e round supprime mécaniquement "
                "les atteintes des manches 1 et 2, qui sont les plus fréquentes.".replace(",", " ")
            )

    # ---- politique de mise ------------------------------------------------- #
    st.markdown("#### La stratégie de mise recommandée par le guide")
    bet = pd.DataFrame(
        {
            "Stratégie": [LABELS.get(b["strategy"], b["strategy"]) for b in report["betting"]],
            "Politique": [b["policy"] for b in report["betting"]],
            "Essais": [b["rounds_played"] for b in report["betting"]],
            "Taux d'atteinte": [b["strike_rate"] for b in report["betting"]],
            "Profit": [b["profit_units"] for b in report["betting"]],
            "Décroissance max": [b["max_drawdown"] for b in report["betting"]],
            "Ruines": [b["ruin_events"] for b in report["betting"]],
            "Témoin sans information": [b["reference_profit_units"] for b in report["betting"]],
        }
    )
    dataframe(
        bet,
        column_config={
            "Taux d'atteinte": st.column_config.NumberColumn(format="%.2f %%"),
            "Profit": st.column_config.NumberColumn(format="%+.3f"),
            "Décroissance max": st.column_config.NumberColumn(format="%.3f"),
            "Témoin sans information": st.column_config.NumberColumn(format="%+.3f"),
        },
        hide_index=True,
    )
    st.caption(
        "Profit en multiples de la bankroll initiale (mise de base 2 %). Le témoin rejoue les mêmes "
        "cibles, mises et dates en tirant l'atteinte avec la loi marginale : il n'a aucune information. "
        "Un profit réel proche du témoin signifie « chance », pas « compétence »."
    )

    # ---- comparaison avec la génération actuelle --------------------------- #
    st.markdown("#### Comparaison avec les modèles ML")
    ml = report.get("ml", {})
    if ml.get("available"):
        ov = report.get("overlap", {})
        st.caption(
            f"Comparaison appariée sur {ml['n_rounds']:,} manches (plage du dernier walk-forward, "
            f"soit {ov.get('share_of_series', float('nan'))*100:.1f} % de la série) : "
            "mêmes manches, mêmes cibles, donc comparaison valide.".replace(",", " ")
        )
        mlt = pd.DataFrame(
            {
                "Cible": [f"{r['target']:g}x" for r in ml["table"]],
                "Modèle": [r["model"] for r in ml["table"]],
                "n": [r["n"] for r in ml["table"]],
                "Taux observé": [r["observed_rate"] for r in ml["table"]],
                "Probabilité annoncée": [r["mean_probability"] for r in ml["table"]],
                "Brier": [r["brier"] for r in ml["table"]],
                "Brier loi marginale": [r["brier_reference"] for r in ml["table"]],
                "Δ Brier": [r["delta_vs_reference"] for r in ml["table"]],
                "p (Holm)": [r["p_holm"] for r in ml["table"]],
                "Verdict": [r["verdict"] for r in ml["table"]],
            }
        )
        dataframe(
            mlt,
            column_config={
                "Taux observé": st.column_config.NumberColumn(format="%.2f %%"),
                "Probabilité annoncée": st.column_config.NumberColumn(format="%.2f %%"),
                "Brier": st.column_config.NumberColumn(format="%.5f"),
                "Brier loi marginale": st.column_config.NumberColumn(format="%.5f"),
                "Δ Brier": st.column_config.NumberColumn(format="%+.6f"),
                "p (Holm)": st.column_config.NumberColumn(format="%.4f"),
            },
            hide_index=True,
        )
        kpi_row(
            [
                {"label": "Modèles meilleurs que la loi marginale", "value": str(ml["n_better"]), "accent": "#00B894"},
                {"label": "Modèles significativement pires", "value": str(ml["n_worse"]), "accent": "#E17055"},
                {"label": "Sans différence démontrée", "value": str(ml["n_null"])},
            ]
        )
    else:
        warn("Aucune probabilité ML enregistrée : lancez un entraînement pour activer la comparaison.", "warn")

    # ---- conclusion -------------------------------------------------------- #
    st.markdown("#### Conclusion")
    st.info(report["summary"]["conclusion"])
    st.caption(
        "Correction de Holm appliquée sur toute la famille de tests (stratégies × horizons, puis "
        "sensibilité fenêtre × ordre). Aucun chiffre de cette page n'est saisi à la main : "
        "tout provient de reports/robotv45.json, régénérable par scripts/legacy_backtest.py."
    )

    with st.expander("📉 Sensibilité : fenêtre de saisie, ordre des cotes, sous-périodes"):
        sens = pd.DataFrame(
            [
                {
                    "Horizon": d["horizon"],
                    "Fenêtre": d["window"],
                    "Ordre": d["order"],
                    "Stratégie": d["strategy"],
                    "n": d["n_predictions"],
                    "Taux": d["rate"],
                    "Écart (pts)": d["gap_vs_reference"] * 100,
                    "p (Holm)": d.get("p_value_holm_sensitivity"),
                }
                for d in report.get("sensitivity", [])
            ]
        )
        if not sens.empty:
            dataframe(
                sens,
                column_config={
                    "Taux": st.column_config.NumberColumn(format="%.2f %%"),
                    "Écart (pts)": st.column_config.NumberColumn(format="%+.2f"),
                    "p (Holm)": st.column_config.NumberColumn(format="%.4f"),
                },
                hide_index=True,
            )
            st.caption(
                "Si un avantage réel existait, il apparaîtrait pour toutes les fenêtres et tous les "
                "ordres. Ici les écarts significatifs changent de signe selon la fenêtre : "
                "c'est la signature d'un choix a posteriori, pas d'un edge."
            )
        per = pd.DataFrame(
            [
                {
                    "Sous-période": d["period"],
                    "Horizon": d["horizon"],
                    "Stratégie": d["strategy"],
                    "n": d["n_predictions"],
                    "Taux": d["rate"],
                    "Écart (pts)": d["gap_vs_reference"] * 100,
                    "p": d["p_value_vs_reference"],
                }
                for d in report.get("periods", [])
            ]
        )
        if not per.empty:
            st.markdown("##### Stabilité dans le temps (tiers de la série)")
            dataframe(
                per,
                column_config={
                    "Taux": st.column_config.NumberColumn(format="%.2f %%"),
                    "Écart (pts)": st.column_config.NumberColumn(format="%+.2f"),
                    "p": st.column_config.NumberColumn(format="%.4f"),
                },
                hide_index=True,
            )
