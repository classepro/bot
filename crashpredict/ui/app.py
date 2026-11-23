"""Application Streamlit — point d'entrée.

Lancement :
    streamlit run crashpredict/ui/app.py --server.address 0.0.0.0 --server.port 8501

L'interface est un pur consommateur des modules métier :
aucune décision de prédiction n'est prise dans ce fichier.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crashpredict import db
from crashpredict.ui.theme import STREAMLIT_CSS
from crashpredict.ui.pages import (
    page_backtesting,
    page_dashboard,
    page_data,
    page_history,
    page_legacy,
    page_prediction,
    page_report,
    page_training,
)

st.set_page_config(
    page_title="Crash Game AI Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(STREAMLIT_CSS, unsafe_allow_html=True)

db.init_db()

PAGES = {
    "📊 Dashboard": page_dashboard.render,
    "🎯 Prédiction": page_prediction.render,
    "🗂 Historique": page_history.render,
    "🧠 Entraînement": page_training.render,
    "🔬 Backtesting": page_backtesting.render,
    "🤖 ROBOT V4.5": page_legacy.render,
    "🗄 Données": page_data.render,
    "📄 Rapport scientifique": page_report.render,
}


def main() -> None:
    with st.sidebar:
        st.markdown(
            "<div class='cp-hero' style='padding:16px 18px'>"
            "<h1 style='font-size:1.15rem'>📈 Crash AI Predictor</h1>"
            "<p style='font-size:.78rem'>Analyse & validation — pas de promesse de gain</p></div>",
            unsafe_allow_html=True,
        )
        choice = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
        st.markdown("---")
        fp = db.data_fingerprint()
        chain_label = (
            "<span style='color:#00B894'>vérifiée</span>"
            if fp["chain_bad"] == 0 and fp["chain_ok"]
            else "<span style='color:#FDCB6E'>partielle</span>"
        )
        st.markdown(
            f"<div class='cp-mono'>manches : {fp['n']:,}<br>"
            f"plage : {fp['first']} → {fp['last']}<br>"
            f"trous : {fp['gaps']}<br>"
            f"chaîne SHA256 : {chain_label}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.caption(
            "⚖️ Outil d'analyse statistique. Les jeux de crash comportent une marge maison ; "
            "aucun modèle ne garantit un gain. Utilisation responsable uniquement."
        )

    PAGES[choice]()


if __name__ == "__main__":
    main()
else:
    main()
