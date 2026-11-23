"""Dashboard : statistiques principales, derniers coefficients, prédiction courante."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ... import db
from ...prediction import current_analysis
from ...prediction.engine import strategy_table
from ...stats import charts, dashboard_stats, prediction_history
from ..components import badge, dataframe as df_box, kpi_row, pct, warn


@st.cache_data(ttl=30, show_spinner=False)
def _load(n: int = 30) -> dict:
    rows = db.get_rounds(limit=800, ascending=False)
    st_rows = db.get_rounds()
    analysis = current_analysis()
    return {
        "recent": [
            {"manche": r["round_id"], "coefficient": round(r["multiplier"], 2),
             "horodatage": r["ts"], "source": r["ts_source"]}
            for r in rows[:n]
        ],
        "series": [r["multiplier"] for r in st_rows],
        "round_ids": [r["round_id"] for r in st_rows],
        "analysis": analysis,
    }


def render() -> None:

    st.markdown(
        "<div class='cp-hero'><h1>Dashboard</h1>"
        "<p>Vue d'ensemble du robot : données vérifiées, prédiction courante, performances mesurées.</p></div>",
        unsafe_allow_html=True,
    )
    data = _load()
    analysis = data["analysis"]
    if analysis.get("error"):
        warn("La base est vide. Importez les données depuis la page <b>Données</b>.", "bad")
        return

    stats = dashboard_stats(backtest=True)
    fp = analysis["data_fingerprint"]

    kpi_row(
        [
            {"label": "Manches vérifiées", "value": f"{fp['n']:,}".replace(",", " "),
             "sub": f"{fp['first']} → {fp['last']}"},
            {"label": "Prédictions évaluées", "value": f"{stats['n_predictions']:,}".replace(",", " "),
             "sub": f"{stats['n_success']} succès · {stats['n_failure']} échecs"},
            {"label": "Taux de réussite (3 manches)", "value": pct(stats.get("success_rate")),
             "sub": "modèle déployé " + str(stats.get("deployed_model") or "n/d"), "accent": "#00B894"},
            {"label": "Série en cours", "value": f"{stats.get('current_success_series', 0)} ✅ / {stats.get('current_failure_series', 0)} ❌",
             "sub": f"record {stats.get('best_series', 0)} succès consécutifs", "accent": "#FDCB6E"},
        ]
    )
    st.write("")

    left, right = st.columns([1.15, 1])
    with left:
        st.plotly_chart(
            charts.rounds_history(data["series"], data["round_ids"], last=300),
            width="stretch",
        )
    with right:
        st.markdown("#### Prédiction courante")
        latest = prediction_history(limit=1, backtest=False)
        if latest.empty:
            st.info("Aucune prédiction en direct enregistrée. Lancez-en une depuis **🎯 Prédiction**.")
        else:
            row = latest.iloc[0]
            st.markdown(
                f"**Cible** : `{row['target']:.2f}x` &nbsp;·&nbsp; "
                f"**Probabilité** : `{row['probability']*100:.1f} %` &nbsp;·&nbsp; {badge(row['status'])}",
                unsafe_allow_html=True,
            )
            st.caption(f"Créée le {row['created_at']} · ancrage manche {row['anchor_round_id']}")
            st.caption(f"Fenêtre de validation : {row['window_round_ids']}")
        st.markdown("#### Derniers coefficients")
        df_box(pd.DataFrame(data["recent"]), height=250)

    st.write("")
    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        st.markdown("#### Contexte statistique")
        a = analysis
        st.markdown(
            f"- moyenne 100 manches : **{a['mean_100']:.2f}x**\n"
            f"- moyenne 1 000 manches : **{a['mean_1000']:.2f}x**\n"
            f"- part ≥ 2x (1 000) : **{a['share_ge_2_1000']*100:.1f} %**\n"
            f"- série < 1.5x : **{a['streak_low_1_5']}** · série ≥ 2x : **{a['streak_high_2']}**"
        )
        drift = a["drift_test"]
        st.markdown(
            f"<div class='cp-warn'>Test de dérive (seuil 2x) : {drift['rate_short']*100:.1f} % sur {drift['n_short']} manches "
            f"contre {drift['rate_long']*100:.1f} % sur {drift['n_long']} — p = {drift['p_value']:.3f} → "
            f"<b>{drift['verdict']}</b></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown("#### Probabilité par cible")
        grid = a["grid"]
        df_box(
            pd.DataFrame(
                [{"cible": f"{g['target']:.2f}x", "probabilité": f"{g['probability']*100:.1f} %"} for g in grid]
            ),
            height=250,
        )
    with c3:
        st.markdown("#### Stratégies proposées")
        strat_rows = [
            {
                "stratégie": entry.get("strategie", entry.get("stratégie")),
                "cible": f"{entry['cible']:.2f}x",
                "probabilité": f"{entry['probabilité']*100:.1f} %",
                "niveau": entry["niveau"],
            }
            for entry in strategy_table()
        ]
        df_box(pd.DataFrame(strat_rows))
    st.markdown("---")
    warn(
        "Les taux affichés proviennent du <b>backtest walk-forward</b> (chaque prédiction n'utilise que le passé). "
        "L'écart entre taux observé et probabilité annoncée est la seule mesure du sérieux du robot ; "
        "il est détaillé dans <b>🔬 Backtesting</b> et <b>📄 Rapport</b>.",
        "warn",
    )
