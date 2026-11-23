"""Module Prédiction : lancer une analyse, obtenir cible + probabilité + explication."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ... import db
from ...config import CONFIG
from ...prediction import current_analysis, make_prediction, strategy_table
from ...prediction.engine import PredictionEngine
from ...prediction.targets import confidence_level, probability_grid
from ...stats import charts
from ...validation import close_pending_predictions
from ..components import badge, dataframe as df_box, kpi_row, pct, warn


def render() -> None:

    st.markdown(
        "<div class='cp-hero'><h1>Prédiction</h1>"
        "<p>Le robot propose un coefficient cible, sa probabilité de validation sur les 3 manches suivantes, "
        "et l'explication chiffrée de cette probabilité.</p></div>",
        unsafe_allow_html=True,
    )

    closed = close_pending_predictions()
    if closed:
        st.toast(f"{closed} prédiction(s) validée(s) automatiquement sur les 3 manches.", icon="✅")

    analysis = current_analysis()
    if analysis.get("error"):
        warn("Base vide : importez des données d'abord.", "bad")
        return

    engine = PredictionEngine.from_storage()
    kpi_row(
        [
            {"label": "Dernière manche", "value": f"{analysis['last_multiplier']:.2f}x",
             "sub": f"manche {analysis['last_round_id']}"},
            {"label": "Historique utilisé", "value": f"{analysis['n_rounds']:,}".replace(",", " "),
             "sub": "manches vérifiées"},
            {"label": "Méthode de probabilité", "value": engine.model_name,
             "sub": "sélectionnée par le backtest", "accent": "#0984E3"},
            {"label": "Cible optimisée (80 %)", "value": f"{analysis['optimized']['target']:.2f}x"
             if analysis["optimized"].get("target") else "n/d",
             "sub": pct(analysis["optimized"].get("probability")), "accent": "#00B894"},
        ]
    )

    st.write("")
    left, right = st.columns([1, 1.1])
    with left:
        st.markdown("#### Lancer une prédiction")
        mode = st.radio(
            "Stratégie",
            ["Cible fixe", "Taux de réussite visé"],
            horizontal=True,
            help="La cible fixe utilise un seuil standard ; le taux visé inverse la loi empirique.",
        )
        target = None
        strategy = None
        desired = None
        if mode == "Cible fixe":
            preset = st.selectbox(
                "Cible",
                [f"{t:.2f}x" for t in CONFIG.targets.fixed] + ["Personnalisée"],
                index=1,
            )
            if preset == "Personnalisée":
                target = st.number_input("Coefficient cible", min_value=1.01, max_value=1000.0, value=3.0, step=0.05)
                strategy = "personnalisée"
            else:
                target = float(preset.rstrip("x"))
                strategy = {
                    1.5: "prudente", 2.0: "équilibrée", 2.5: "ambitieuse", 5.0: "très ambitieuse"
                }.get(target, "fixe")
        else:
            desired = st.slider("Taux de réussite visé", 0.5, 0.99, 0.8, 0.01,
                                help="La cible est déduite de l'historique, pas choisie au hasard.")
            strategy = f"taux visé {desired*100:.0f} %"

        expected = probability_grid(
            [float(r["multiplier"]) for r in db.get_rounds()],
            targets=[target] if target else CONFIG.targets.grid,
        )
        if target:
            p = expected[0]["probability"]
            st.markdown(
                f"Probabilité estimée pour cette cible : **{pct(p)}** "
                f"(n = {expected[0]['n_rounds']:,} manches)".replace(",", " "),
                unsafe_allow_html=True,
            )
        if st.button("🧠 Lancer l'analyse et enregistrer la prédiction", type="primary", width="stretch"):
            try:
                result = make_prediction(target=target, strategy=strategy, desired_rate=desired, persist=True)
            except RuntimeError as exc:
                st.error(str(exc))
            else:
                st.session_state["last_prediction"] = result
                st.success(
                    f"Prédiction {result['id']} enregistrée : cible {result['target']:.2f}x, "
                    f"probabilité {pct(result['probability'])} — statut EN ATTENTE."
                )

        if st.session_state.get("last_prediction"):
            r = st.session_state["last_prediction"]
            st.markdown("---")
            st.markdown(
                f"#### Prédiction {r['id']}\n"
                f"{badge('PENDING')} &nbsp; cible **{r['target']:.2f}x** &nbsp;·&nbsp; "
                f"probabilité **{pct(r['probability'])}** ({r['confidence']}) &nbsp;·&nbsp; "
                f"méthode `{r['model_name']}`\n\n"
                f"Manches de validation : `{', '.join(str(i) for i in r['window_round_ids'])}`",
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("#### Probabilité selon la cible")
        grid = probability_grid([float(r["multiplier"]) for r in db.get_rounds()])
        st.plotly_chart(charts.target_probability_curve(grid), width="stretch")

    st.markdown("#### Taux de remplissage et gain espéré par cible")
    st.plotly_chart(charts.fill_rate_curve([float(r["multiplier"]) for r in db.get_rounds()]), width="stretch")
    st.caption(
        "Lecture : la barre est la part des fenêtres de 3 manches qui atteignent la cible (règle de validation). "
        "La courbe est le gain espéré par unité misée si l'on jouait chaque manche à cette cible : "
        "en dessous de 1.00, la marge maison l'emporte. C'est une mesure descriptive, pas une recommandation de jeu."
    )

    st.markdown("#### Les quatre stratégies + cible optimisée")
    strat_df = pd.DataFrame(strategy_table()).rename(columns={"strategie": "stratégie"})
    strat_df["probabilité"] = strat_df["probabilité"].map(lambda v: f"{v*100:.1f} %")
    strat_df["succès_attendus"] = strat_df["succès_attendus"].map(lambda v: f"{v:,}".replace(",", " "))
    strat_df.columns = ["Stratégie", "Cible", "Probabilité estimée", "Manches analysées", "Succès attendus", "Niveau"]
    df_box(strat_df)

    st.markdown("#### Prédictions proposées (enregistrées en un clic)")
    cols = st.columns(len(CONFIG.targets.fixed) + 1)
    for col, target in zip(cols, CONFIG.targets.fixed):
        with col:
            est = probability_grid([float(r["multiplier"]) for r in db.get_rounds()], targets=[target])[0]
            st.markdown(
                f"<div class='cp-card'><div class='cp-label'>Cible {target:.2f}x</div>"
                f"<div class='cp-value'>{est['probability']*100:.1f} %</div>"
                f"<div class='cp-sub'>produit {est['probability']*target:.2f} unités</div></div>",
                unsafe_allow_html=True,
            )
            if st.button(f"Enregistrer {target:.2f}x", key=f"save_{target}"):
                try:
                    res = make_prediction(target=target)
                except RuntimeError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["last_prediction"] = res
                    st.success(f"{res['id']} enregistrée")
                    st.rerun()
    with cols[-1]:
        opt = analysis["optimized"]
        if opt.get("target"):
            st.markdown(
                f"<div class='cp-card'><div class='cp-label'>Optimisée 80 %</div>"
                f"<div class='cp-value'>{opt['probability']*100:.1f} %</div>"
                f"<div class='cp-sub'>cible {opt['target']:.2f}x</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("#### Explication des variables")
    with st.expander("Pourquoi cette probabilité ? (détail complet)", expanded=False):
        a = analysis
        st.markdown(
            f"""
**1. Loi marginale (ce qui détermine réellement la probabilité).**
P(validation) = 1 − (1 − P(une manche ≥ cible))³, calculée sur les
{CONFIG.probabilities.long_window:,} dernières manches. Plus la cible est haute, plus la probabilité chute.

**2. Contexte récent (affiché, mais sans effet démontré).**
- moyenne des 100 dernières manches : `{a['mean_100']:.2f}x`
- part de manches ≥ 2x sur 1 000 : `{a['share_ge_2_1000']*100:.1f} %`
- série actuelle sous 1.5x : `{a['streak_low_1_5']}` — série actuelle ≥ 2x : `{a['streak_high_2']}`
- test de dérive (binomial) : {a['drift_test']['verdict']} (p = {a['drift_test']['p_value']:.3f})

**3. Avertissement.**
Ces variables sont affichées parce qu'elles sont demandées, mais le backtest montre qu'elles n'apportent
aucun gain de Brier significatif : la probabilité annoncée est une fréquence historique de validation,
pas une prédiction du coefficient exact de la prochaine manche.
""".replace(",", " ")
        )
        from ...stats import feature_importances
        imp = feature_importances()
        if not imp.empty:
            st.plotly_chart(charts.importance_plot(imp), width="stretch")
        else:
            st.info("Importances disponibles après un entraînement complet (page 🧠 Entraînement).")

    st.markdown("---")
    st.markdown("#### Dernières prédictions en direct")
    from ...stats import prediction_history
    df = prediction_history(limit=15, backtest=False)
    if df.empty:
        st.info("Aucune prédiction en direct pour l'instant.")
    else:
        df["probability"] = df["probability"].map(lambda v: f"{v*100:.1f} %" if pd.notna(v) else "n/d")
        df["status"] = df["status"].map(lambda s: {"SUCCESS": "✅ SUCCÈS", "FAILURE": "❌ ÉCHEC", "PENDING": "⏳ EN ATTENTE"}.get(s, s))
        df_box(df)
