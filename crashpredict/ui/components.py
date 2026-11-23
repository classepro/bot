"""Composants visuels réutilisables (cartes, badges, tableaux)."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def card(label: str, value: str, sub: str = "", accent: str = "#6C5CE7") -> str:
    return (
        f"<div class='cp-card'><div class='cp-label'>{label}</div>"
        f"<div class='cp-value' style='color:{accent}'>{value}</div>"
        f"<div class='cp-sub'>{sub}</div></div>"
    )


def badge(status: str) -> str:
    mapping = {
        "SUCCESS": ("cp-success", "✅ SUCCÈS"),
        "FAILURE": ("cp-failure", "❌ ÉCHEC"),
        "PENDING": ("cp-pending", "⏳ EN ATTENTE"),
    }
    cls, label = mapping.get(status, ("cp-info", status))
    return f"<span class='cp-pill {cls}'>{label}</span>"


def kpi_row(items: list[dict[str, Any]]) -> None:
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            st.markdown(
                card(item["label"], item["value"], item.get("sub", ""), item.get("accent", "#6C5CE7")),
                unsafe_allow_html=True,
            )


def status_summary(status: str, probability: float | None, target: float | None = None) -> None:
    left, right = st.columns([2, 1])
    with left:
        txt = f"Statut : {badge(status)}"
        if target is not None:
            txt += f" &nbsp; cible <b>{target:.2f}x</b>"
        if probability is not None:
            txt += f" &nbsp; probabilité <b>{probability*100:.1f} %</b>"
        st.markdown(txt, unsafe_allow_html=True)
    with right:
        if probability is not None:
            st.progress(min(max(probability, 0.0), 1.0))


def warn(text: str, kind: str = "warn") -> None:
    st.markdown(f"<div class='cp-{kind}'>{text}</div>", unsafe_allow_html=True)


def dataframe(df: pd.DataFrame, height=None, **kwargs) -> None:
    if df is None or df.empty:
        st.info("Aucune donnée disponible pour le moment.")
        return
    opts = {"width": "stretch", "hide_index": True, **kwargs}
    if isinstance(height, int) and height >= 40:
        opts["height"] = height
    st.dataframe(df, **opts)


def pct(value: float | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/d"
    return f"{value*100:.{digits}f} %"


def num(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/d"
    return f"{value:,.{digits}f}".replace(",", " ")
