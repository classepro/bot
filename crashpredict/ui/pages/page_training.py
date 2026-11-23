"""Entraînement : lancer, comparer, visualiser, réentraîner, sauvegarder."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ... import db
from ...config import CONFIG
from ...models.registry import available_models
from ...stats import charts, dashboard_stats
from ...stats.dashboard import feature_importances, last_metrics_run, probability_samples
from ...training import DEFAULT_MODELS, run_training
from ..components import dataframe, kpi_row, pct, warn


def render() -> None:
    st.markdown(
        "<div class='cp-hero'><h1>Entraînement & comparaison de modèles</h1>"
        "<p>Walk-forward : chaque prédiction n'utilise que le passé, avec purge de 3 manches. "
        "Aucun mélange aléatoire des données.</p></div>",
        unsafe_allow_html=True,
    )

    fp = db.data_fingerprint()
    if not fp["n"]:
        warn("Base vide : importez des données d'abord.", "bad")
        return

    with st.form("train_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            targets = st.multiselect(
                "Cibles", CONFIG.targets.grid, default=CONFIG.targets.fixed,
                format_func=lambda v: f"{v:.2f}x",
            )
            models = st.multiselect("Modèles", available_models(), default=DEFAULT_MODELS)
        with c2:
            initial_train = st.number_input("Manches d'entraînement initial", 5_000, 80_000, CONFIG.backtest.initial_train, 5_000)
            step = st.number_input("Pas de réentraînement (manches)", 1_000, 20_000, CONFIG.backtest.step, 1_000)
        with c3:
            max_folds = st.number_input("Nombre de folds max (0 = tous)", 0, 40, 0, 1,
                                        help="Chaque fold réentraîne tous les modèles sur toutes les cibles.")
            ablation = st.checkbox("Analyse d'ablation des features", value=False,
                                   help="Mesure l'apport réel de chaque famille de variables (2 folds, plus lent).")
        submitted = st.form_submit_button("🚀 Lancer l'entraînement + backtest", type="primary")

    if submitted:
        if not targets or not models:
            st.error("Sélectionnez au moins une cible et un modèle.")
        else:
            progress = st.progress(0.0, text="Préparation…")
            log = st.empty()

            def cb(i: int, n: int, msg: str) -> None:
                progress.progress(min(i / max(n, 1), 1.0), text=msg)
                log.caption(f"fold {i}/{n}")

            try:
                summary = run_training(
                    targets=targets, models=models, initial_train=int(initial_train),
                    step=int(step), max_folds=int(max_folds) or None, with_ablation=ablation,
                    ablation_target=float(targets[0]), progress=cb,
                )
            except Exception as exc:
                st.error(f"Échec de l'entraînement : {exc!r}")
            else:
                progress.progress(1.0, text="Terminé")
                st.success(
                    f"Entraînement terminé en {summary.get('duration_seconds')} s — "
                    f"{summary.get('n_folds')} folds. Modèle déployé : {summary.get('deployed_model')}."
                )

    run_id, metrics, skill, summary = last_metrics_run()
    if not run_id:
        st.info("Aucun entraînement enregistré pour l'instant.")
        return

    st.markdown("---")
    st.markdown(f"#### Dernier run `{run_id}`")
    if summary:
        kpi_row(
            [
                {"label": "Folds", "value": str(summary.get("n_folds"))},
                {"label": "Modèle déployé", "value": str(summary.get("deployed_model")), "accent": "#00B894"},
                {"label": "Durée", "value": f"{summary.get('duration_seconds')} s"},
                {"label": "Manches testées",
                 "value": f"{summary.get('last_test_round', 0) - summary.get('first_test_round', 0):,}".replace(",", " "),
                 "sub": f"{summary.get('first_test_round')} → {summary.get('last_test_round')}"},
            ]
        )

    if not metrics.empty:
        from ...stats import performance_by_model
        perf = performance_by_model(backtest=True)
        st.markdown("#### Comparaison des modèles (hors échantillon)")
        if not perf.empty:
            st.plotly_chart(charts.model_comparison(perf), width="stretch")
            show = perf.copy()
            for col in ("brier", "auc", "delta_brier_moyen"):
                if col in show.columns:
                    show[col] = show[col].map(lambda v: f"{v:.6f}" if pd.notna(v) else "n/d")
            dataframe(show.rename(columns={"model_name": "modèle"}))
        st.caption(
            "Δ Brier moyen > 0 signifie que le modèle fait mieux que la baseline empirique. "
            "La significativité (correction de Holm) est vérifiée dans l'onglet Backtesting."
        )

    if not skill.empty:
        piv = skill.pivot_table(index=["model_name", "target"], columns="metric", values="value").reset_index()
        piv.columns = ["modèle", "cible", "Δ Brier", "p-value (Holm)"]
        piv["verdict"] = piv.apply(
            lambda r: "significatif" if (r["Δ Brier"] > 0 and r["p-value (Holm)"] < 0.05) else "non significatif",
            axis=1,
        )
        st.markdown("#### Tests de significativité")
        st.dataframe(piv.style.format({"Δ Brier": "{:.6f}", "p-value (Holm)": "{:.4f}"}), width="stretch")

    samples = probability_samples()
    if not samples.empty:
        st.markdown("#### Calibration du modèle déployé")
        from ...backtesting.metrics import calibration_table
        from ...stats import charts as _charts
        target_cal = st.selectbox("Cible pour la calibration", sorted(samples["target"].unique()),
                                  format_func=lambda v: f"{v:.2f}x", index=0)
        sub = samples[(samples["target"] == target_cal) & samples["p"].notna()]
        if not sub.empty:
            st.plotly_chart(_charts.calibration_plot(calibration_table(sub["y"], sub["p"])), width="stretch")
            st.caption("Une calibration proche de la diagonale signifie que les probabilités annoncées correspondent "
                       "aux fréquences réellement observées hors échantillon.")

    imp = feature_importances()
    if not imp.empty:
        st.markdown("#### Variables importantes")
        st.plotly_chart(charts.importance_plot(imp), width="stretch")

    if summary and summary.get("ablation"):
        st.markdown("#### Ablation des features (quel groupe apporte quoi ?)")
        abl = pd.DataFrame(summary["ablation"])
        st.dataframe(abl.style.format({"brier": "{:.6f}", "auc": "{:.4f}"}), width="stretch")
        st.caption(
            "Si retirer un groupe ne dégrade pas le Brier, ce groupe n'apporte aucune information utile. "
            "C'est un test de sobriété, pas une performance."
        )

    st.markdown("#### Runs précédents")
    runs = db.list_runs(limit=10)
    if runs:
        dataframe(
            pd.DataFrame(
                [
                    {"run": r["id"], "date": r["created_at"], "statut": r["status"],
                     "fichier": json.loads(r["data_fingerprint"])["n"] if r["data_fingerprint"] else None}
                    for r in runs
                ]
            )
        )
