"""Historique complet des prédictions, avec validation automatique."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ... import db
from ...stats import charts, dashboard_stats, prediction_history, streaks
from ...validation import close_pending_predictions
from ..components import dataframe, kpi_row, pct, warn


def render() -> None:
    st.markdown(
        "<div class='cp-hero'><h1>Historique des prédictions</h1>"
        "<p>Chaque prédiction est conservée avec son identifiant, ses 3 manches de validation, "
        "son résultat et son statut.</p></div>",
        unsafe_allow_html=True,
    )
    close_pending_predictions()

    backtest = st.toggle(
        "Afficher les prédictions du backtest (1 par manche)",
        value=False,
        help="Le backtest produit une prédiction par manche : les taux sont valides mais les échantillons ne sont pas indépendants.",
    )
    stats = dashboard_stats(backtest=backtest)
    kpi_row(
        [
            {"label": "Prédictions", "value": f"{stats['n_predictions']:,}".replace(",", " ")},
            {"label": "Succès", "value": str(stats["n_success"]), "accent": "#00B894"},
            {"label": "Échecs", "value": str(stats["n_failure"]), "accent": "#E17055"},
            {"label": "Taux", "value": pct(stats.get("success_rate")), "accent": "#FDCB6E"},
            {"label": "En attente", "value": str(stats["n_pending"]), "accent": "#0984E3"},
        ]
    )
    st.write("")

    df = prediction_history(limit=2000, backtest=backtest)
    if df.empty:
        warn("Aucune prédiction enregistrée. Lancez une analyse depuis <b>🎯 Prédiction</b>.", "warn")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        f_status = st.multiselect("Statut", sorted(df["status"].unique()), default=sorted(df["status"].unique()))
    with c2:
        f_strategy = st.multiselect("Stratégie", sorted(df["strategy"].dropna().unique()),
                                    default=sorted(df["strategy"].dropna().unique()))
    with c3:
        targets = sorted(df["target"].unique())
        f_target = st.multiselect("Cible", targets, default=targets)

    view = df[df["status"].isin(f_status) & df["strategy"].isin(f_strategy) & df["target"].isin(f_target)]

    st.markdown(f"#### {len(view):,} prédiction(s) affichée(s)".replace(",", " "))
    table = view.copy()
    table["probability"] = table["probability"].map(lambda v: f"{v*100:.1f} %" if pd.notna(v) else "n/d")
    table["target"] = table["target"].map(lambda v: f"{v:.2f}x")
    table["status"] = table["status"].map(
        lambda s: {"SUCCESS": "✅ SUCCÈS", "FAILURE": "❌ ÉCHEC", "PENDING": "⏳ EN ATTENTE"}.get(s, s)
    )
    table["actual_max"] = table["actual_max"].map(lambda v: f"{v:.2f}x" if pd.notna(v) else "—")
    table.columns = ["ID", "Date/heure (UTC)", "Stratégie", "Cible", "Probabilité", "Statut",
                     "Manche d'ancrage", "Manches de validation", "Coefficients observés", "MAX observé", "Modèle"]
    dataframe(table, height=430)

    st.markdown("#### Séries et records")
    s_succ = streaks(pd.DataFrame([dict(r) for r in db.predictions(limit=1_000_000, backtest=backtest)]), "success")
    s_fail = streaks(pd.DataFrame([dict(r) for r in db.predictions(limit=1_000_000, backtest=backtest)]), "failure")
    kpi_row(
        [
            {"label": "Série actuelle de succès", "value": str(s_succ["current"]), "accent": "#00B894",
             "sub": f"record : {s_succ['best']}"},
            {"label": "Série actuelle d'échecs", "value": str(s_fail["current"]), "accent": "#E17055",
             "sub": f"record : {s_fail['best']}"},
        ]
    )

    st.markdown("#### Chronologie des prédictions")
    timeline = view.copy()
    timeline["anchor_round_id"] = pd.to_numeric(timeline["anchor_round_id"], errors="coerce")
    st.plotly_chart(charts.predictions_timeline(timeline.dropna(subset=["anchor_round_id"]).tail(500)),
                    width="stretch")

    st.markdown("#### Export")
    csv = view.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Télécharger l'historique (CSV)", csv, "predictions.csv", "text/csv")
