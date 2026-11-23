"""Backtesting : protocole, métriques hors échantillon, significativité, calibration."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ...backtesting.examples import protocol_explanation
from ...config import CONFIG
from ...stats import charts, dashboard_stats, performance_by_period, performance_by_target
from ...stats.dashboard import calibration_of_last_run, last_metrics_run, probability_samples
from ..components import dataframe, kpi_row, pct, warn


def render() -> None:
    st.markdown(
        "<div class='cp-hero'><h1>Backtesting walk-forward</h1>"
        "<p>Protocole chronologique strict : le modèle ne voit jamais le futur. "
        "Les résultats ci-dessous sont la seule preuve de valeur du robot.</p></div>",
        unsafe_allow_html=True,
    )

    with st.expander("📐 Protocole exact (à lire avant d'interpréter un chiffre)", expanded=False):
        st.markdown(protocol_explanation())

    run_id, metrics, skill, summary = last_metrics_run()
    if not run_id:
        warn("Aucun backtest enregistré : lancez un entraînement depuis <b>🧠 Entraînement</b>.", "warn")
        return

    st.markdown(f"**Run** `{run_id}` · horizon {CONFIG.rules.horizon} manches · purge {CONFIG.backtest.purge} manches "
                f"· fenêtre d'entraînement max {CONFIG.backtest.max_train_window:,} manches".replace(",", " "))

    stats = dashboard_stats(backtest=True)
    kpi_row(
        [
            {"label": "Prédictions évaluées", "value": f"{stats['n_predictions']:,}".replace(",", " "),
             "sub": "une par manche et par cible"},
            {"label": "Taux observé", "value": pct(stats.get("success_rate")), "accent": "#00B894"},
            {"label": "Brier global", "value": f"{stats.get('brier', float('nan')):.5f}" if stats.get("brier") else "n/d",
             "sub": "erreur quadratique des probabilités"},
            {"label": "Modèle déployé", "value": str(stats.get("deployed_model")), "accent": "#0984E3"},
        ]
    )

    st.markdown("#### Le robot bat-il la baseline ?")
    if skill is not None and not skill.empty:
        piv = skill.pivot_table(index=["model_name", "target"], columns="metric", values="value").reset_index()
        if "delta_brier_vs_baseline" in piv.columns and "ci_low" not in piv.columns:
            # reconstruire les IC depuis la table skill brute
            pass
        piv.columns = [c if c in ("model_name", "target") else c for c in piv.columns]
        display = piv.rename(columns={
            "model_name": "modèle", "target": "cible",
            "delta_brier_vs_baseline": "Δ Brier", "p_value_holm": "p (Holm)",
        })
        if "Δ Brier" in display.columns:
            display["verdict"] = display.apply(
                lambda r: "meilleur (significatif)" if (r.get("Δ Brier", 0) or 0) > 0 and (r.get("p (Holm)", 1) or 1) < 0.05
                else ("pire (significatif)" if (r.get("Δ Brier", 0) or 0) < 0 and (r.get("p (Holm)", 1) or 1) < 0.05
                      else "pas de différence démontrée"),
                axis=1,
            )
            st.dataframe(display.style.format({"Δ Brier": "{:+.6f}", "p (Holm)": "{:.4f}"}),
                         width="stretch", hide_index=True)
        any_significant = (
            display["verdict"].str.contains("significatif").any() if "verdict" in display.columns else False
        )
        if any_significant:
            better = display[display["verdict"] == "meilleur (significatif)"]
            if better.empty:
                warn(
                    "Aucun modèle ne bat la baseline de façon significative ; certains font "
                    "<b>significativement pire</b>. Le robot déploie donc la loi empirique.",
                    "bad",
                )
            else:
                warn(f"{len(better)} configuration(s) battent la baseline de façon significative.", "good")
        else:
            warn(
                "Aucun modèle ne se distingue significativement de la baseline empirique "
                "(correction de Holm, IC 95 % bootstrap). C'est le résultat honnête : "
                "les coefficients ne montrent pas de structure exploitable.",
                "warn",
            )

        st.markdown("##### Visualisation du gain (IC 95 %)")
        from ...stats.dashboard import _load_predictions  # noqa: F401
        from ...config import MODEL_DIR
        path = MODEL_DIR / "last_skill.pkl"
        if path.exists():
            raw = pd.read_pickle(path)
            st.plotly_chart(charts.skill_plot(raw), width="stretch")

    st.markdown("#### Taux de réussite par cible (walk-forward)")
    by_target = performance_by_target(backtest=True)
    if not by_target.empty:
        st.plotly_chart(charts.success_rate_by_target(by_target), width="stretch")
        show = by_target.copy()
        for col in ("taux", "prob_moyenne", "écart vs prob"):
            show[col] = show[col].map(lambda v: f"{v*100:+.2f} %" if col == "écart vs prob" else f"{v*100:.2f} %")
        show["brier"] = show["brier"].map(lambda v: f"{v:.6f}")
        dataframe(show.rename(columns={"target": "cible", "prob_moyenne": "probabilité annoncée", "fiable": "échantillon suffisant"}))
        st.caption(
            "« fiable » = au moins 30 prédictions. Un écart positif entre taux observé et probabilité annoncée "
            "signifie que le robot est trop pessimiste ; un écart négatif qu'il est trop optimiste."
        )

    st.markdown("#### Calibration")
    calib_target = st.selectbox("Cible pour la courbe de calibration", CONFIG.targets.fixed, format_func=lambda v: f"{v:.2f}x")
    calib = calibration_of_last_run(target=float(calib_target))
    if calib:
        st.plotly_chart(charts.calibration_plot(calib), width="stretch")
        st.caption("Points proches de la diagonale = probabilités honnêtes. Taille du point = nombre d'observations.")
    else:
        st.info("Pas encore de données de calibration pour cette cible.")

    st.markdown("#### Stabilité dans le temps")
    periods = performance_by_period(backtest=True)
    if not periods.empty:
        st.plotly_chart(charts.performance_over_time(periods), width="stretch")
        dataframe(periods.rename(columns={
            "bloc": "bloc", "manche_min": "manche min", "manche_max": "manche max", "n": "n",
            "taux": "taux", "prob_moyenne": "probabilité annoncée",
        }))
    else:
        st.info("Données insuffisantes pour l'analyse par période.")

    samples = probability_samples()
    if not samples.empty:
        st.markdown("#### Échantillon brut des probabilités (auditable)")
        st.caption("Chaque ligne = une manche, un modèle, une cible, la probabilité annoncée et la vérité observée.")
        st.dataframe(samples.head(500), width="stretch", hide_index=True)
        st.download_button(
            "⬇️ Télécharger toutes les probabilités (CSV)",
            samples.to_csv(index=False).encode("utf-8"),
            "probabilites_backtest.csv",
            "text/csv",
        )
