"""Rapport scientifique : ce que le système démontre, ce qu'il ne démontre pas."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ... import db
from ...config import CONFIG
from ...stats import charts, dashboard_stats, performance_by_target
from ...stats.dashboard import last_metrics_run
from ..components import dataframe, kpi_row, pct, warn


def render() -> None:
    st.markdown(
        "<div class='cp-hero'><h1>Rapport scientifique</h1>"
        "<p>Résultats réels, y compris négatifs. Aucun chiffre n'est présenté sans son dénominateur, "
        "sa fenêtre et sa méthode.</p></div>",
        unsafe_allow_html=True,
    )
    run_id, metrics, skill, summary = last_metrics_run()
    fp = db.data_fingerprint()

    st.markdown("## 1. Données")
    st.markdown(
        f"""
- **{fp['n']:,} manches** consécutives ({fp['first']} → {fp['last']}), **{fp['gaps']} trou**,
  **{fp['duplicates']} doublon**.
- Chaîne provably-fair : **{fp['chain_ok']:,} transitions vérifiées, {fp['chain_bad']} échec**.
- Horodatage : la source historique n'en contient pas ; les temps sont estimés (`ts_source='estimated'`)
  tant que le collecteur live ne les remplace pas (`ts_source='live'`).
""".replace(",", " ")
    )

    if not run_id:
        warn("Aucun backtest enregistré : le rapport de performance sera disponible après un entraînement.", "warn")
        return

    st.markdown("## 2. Protocole")
    st.markdown(
        f"Walk-forward, purge de {CONFIG.rules.horizon} manches, réentraînement tous les "
        f"{CONFIG.backtest.step:,} manches, fenêtre d'entraînement ≤ {CONFIG.backtest.max_train_window:,} manches, "
        f"test sur les manches **{summary.get('first_test_round')} → {summary.get('last_test_round')}**.".replace(",", " ")
    )

    st.markdown("## 3. Résultats")
    stats = dashboard_stats(backtest=True)
    kpi_row(
        [
            {"label": "Prédictions évaluées", "value": f"{stats['n_predictions']:,}".replace(",", " ")},
            {"label": "Taux observé", "value": pct(stats.get("success_rate")), "accent": "#00B894"},
            {"label": "Brier", "value": f"{stats.get('brier'):.5f}" if stats.get("brier") else "n/d"},
            {"label": "Modèle déployé", "value": str(stats.get("deployed_model"))},
        ]
    )

    if skill is not None and not skill.empty:
        piv = skill.pivot_table(index=["model_name", "target"], columns="metric", values="value").reset_index()
        display = piv.rename(columns={"model_name": "modèle", "target": "cible",
                                      "delta_brier_vs_baseline": "Δ Brier", "p_value_holm": "p (Holm)"})
        if "Δ Brier" in display.columns:
            display["verdict"] = display.apply(
                lambda r: "meilleur (significatif)" if (r.get("Δ Brier", 0) or 0) > 0 and (r.get("p (Holm)", 1) or 1) < 0.05
                else ("pire (significatif)" if (r.get("Δ Brier", 0) or 0) < 0 and (r.get("p (Holm)", 1) or 1) < 0.05
                      else "non démontré"),
                axis=1,
            )
            st.markdown("### 3.1 Modèles vs baseline empirique")
            st.dataframe(display.style.format({"Δ Brier": "{:+.6f}", "p (Holm)": "{:.4f}"}),
                         width="stretch", hide_index=True)
            better = display[display["verdict"] == "meilleur (significatif)"]
            worse = display[display["verdict"] == "pire (significatif)"]
            if better.empty:
                warn(
                    "<b>Conclusion honnête : aucun modèle ne démontre un avantage significatif</b> sur la loi "
                    "empirique. " + (f"{len(worse)} configuration(s) font significativement pire." if len(worse) else "")
                    + " Le robot déploie donc la loi empirique et affiche des probabilités calibrées, sans prétendre "
                    "prédire les manches.",
                    "warn",
                )
            else:
                warn(f"{len(better)} configuration(s) présentent un avantage significatif après correction de Holm.", "good")

    by_target = performance_by_target(backtest=True)
    if not by_target.empty:
        st.markdown("### 3.2 Taux par cible")
        view = by_target.copy()
        view["taux"] = view["taux"].map(lambda v: f"{v*100:.2f} %")
        view["prob_moyenne"] = view["prob_moyenne"].map(lambda v: f"{v*100:.2f} %")
        view["écart vs prob"] = view["écart vs prob"].map(lambda v: f"{v*100:+.2f} pts")
        view["brier"] = view["brier"].map(lambda v: f"{v:.6f}")
        st.dataframe(view.rename(columns={"target": "cible", "prob_moyenne": "probabilité annoncée",
                                          "fiable": "fiable"}), width="stretch", hide_index=True)

    st.markdown("## 4. Ce que le système démontre / ne démontre pas")
    st.markdown(
        """
**Démontré**
- Les probabilités annoncées sont calibrées sur des données jamais utilisées à l'entraînement
  (comparer « probabilité annoncée » et « taux observé » ci-dessus).
- La règle de validation est unique et appliquée de façon identique en backtest et en direct.
- Les données sont authentiques et vérifiables cryptographiquement.

**Non démontré**
- Qu'un historique de coefficients permette de prévoir la manche suivante. Sur ces données,
  les manches sont statistiquement indiscernables d'un tirage indépendant (autocorrélation ≈ 0,
  taux stables par blocs, pas de clustering significatif).
- Qu'une stratégie génère un gain à long terme : la marge maison (~1 % par manche sur cette
  implémentation) rend l'espérance négative si le RNG est sain.

**Conséquence pratique**
Le robot est un **outil de mesure** : il transforme une intuition (« ça va payer ») en probabilité
vérifiable et en historique auditable. Il ne transforme pas un jeu à espérance négative en
investissement.
""".replace("≈", "≈")
    )

    st.markdown("## 5. Rapport généré (HTML / Markdown)")
    from ...config import REPORT_DIR
    from ...reporting import build_report, render_html

    html_path = REPORT_DIR / "rapport_performance.html"
    md_path = REPORT_DIR / "rapport_performance.md"
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Régénérer le rapport"):
            from ...reporting import write_reports

            write_reports()
            st.success("Rapport régénéré depuis la base.")
            html_path = REPORT_DIR / "rapport_performance.html"
            md_path = REPORT_DIR / "rapport_performance.md"
    with c2:
        st.caption("Le rapport est aussi régénéré automatiquement après chaque entraînement.")
    if html_path.exists():
        st.download_button("⬇️ Télécharger le rapport HTML", html_path.read_bytes(), "rapport_performance.html", "text/html")
    if md_path.exists():
        with st.expander("Aperçu Markdown du rapport", expanded=False):
            st.markdown(md_path.read_text(encoding="utf-8"))
    else:
        st.info("Aucun rapport généré pour l'instant : lancez un entraînement ou cliquez sur « Régénérer ».")

    st.markdown("## 6. Exports")
    payload = {
        "run_id": run_id,
        "summary": summary,
        "data_fingerprint": fp,
        "metrics": metrics.to_dict(orient="records") if metrics is not None and not metrics.empty else [],
        "skill": skill.to_dict(orient="records") if skill is not None and not skill.empty else [],
    }
    st.download_button(
        "⬇️ Télécharger le rapport (JSON)",
        json.dumps(payload, indent=2, default=str).encode("utf-8"),
        f"rapport_{run_id}.json",
        "application/json",
    )
