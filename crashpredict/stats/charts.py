"""Graphiques Plotly (aucune donnée simulée : tout vient de la base)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = ["#6C5CE7", "#00B894", "#FDCB6E", "#E17055", "#0984E3", "#D63031", "#00CEC9", "#636E72"]
TEMPLATE = "plotly_dark"
LAYOUT = dict(template=TEMPLATE, margin=dict(l=40, r=20, t=50, b=40), hovermode="x unified")


def rounds_history(multipliers, round_ids=None, last: int = 500) -> go.Figure:
    m = np.asarray(multipliers, dtype=float)[-last:]
    x = np.asarray(round_ids)[-last:] if round_ids is not None else np.arange(len(m))
    fig = go.Figure(go.Scatter(x=x, y=m, mode="lines+markers", line=dict(color=PALETTE[0], width=1),
                               marker=dict(size=4), name="coefficient"))
    fig.add_hline(y=2.0, line_dash="dot", line_color=PALETTE[2], annotation_text="2.00x")
    fig.add_hline(y=5.0, line_dash="dot", line_color=PALETTE[3], annotation_text="5.00x")
    fig.update_layout(title=f"Coefficients des {len(m)} dernières manches (échelle log)", yaxis_type="log", **LAYOUT)
    fig.update_yaxes(title="coefficient")
    fig.update_xaxes(title="manche")
    return fig


def distribution(multipliers, bins: int = 60) -> go.Figure:
    m = np.asarray(multipliers, dtype=float)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Distribution (log)", "Fonction de survie empirique"))
    fig.add_trace(go.Histogram(x=np.log10(np.maximum(m, 1.0)), nbinsx=bins, marker_color=PALETTE[0]), row=1, col=1)
    grid = np.logspace(0, np.log10(max(m.max(), 2.0)), 200)
    surv = [float(np.mean(m >= t)) for t in grid]
    fig.add_trace(go.Scatter(x=grid, y=surv, mode="lines", line=dict(color=PALETTE[1]), name="P(X ≥ t)"), row=1, col=2)
    for t in (1.5, 2, 5, 10):
        fig.add_vline(x=t, line_dash="dot", line_color=PALETTE[2], row=1, col=2)
    fig.update_xaxes(title="log10(coefficient)", row=1, col=1)
    fig.update_xaxes(title="coefficient (log)", type="log", row=1, col=2)
    fig.update_yaxes(title="P(X ≥ t)", row=1, col=2)
    fig.update_layout(title="Distribution des coefficients — données réelles", **{**LAYOUT, "hovermode": "closest"})
    return fig


def target_probability_curve(grid_rows: list[dict]) -> go.Figure:
    df = pd.DataFrame(grid_rows)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["target"], y=df["probability"], mode="lines+markers",
                             line=dict(color=PALETTE[1], width=3), name="P(validation 3 manches)"))
    fig.add_hline(y=0.8, line_dash="dash", line_color=PALETTE[2], annotation_text="80 %")
    fig.add_hline(y=0.6, line_dash="dash", line_color=PALETTE[3], annotation_text="60 %")
    fig.update_layout(title="Probabilité de validation par cible (loi empirique)", xaxis_title="cible (x)",
                      yaxis_title="probabilité", **LAYOUT)
    return fig


def success_rate_by_target(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["target"].astype(str), y=df["taux"], name="taux observé",
                         marker_color=PALETTE[1], text=[f"{v*100:.1f}%" for v in df["taux"]]))
    fig.add_trace(go.Scatter(x=df["target"].astype(str), y=df["prob_moyenne"], mode="markers+lines",
                             name="probabilité annoncée", marker=dict(size=12, color=PALETTE[0])))
    fig.update_layout(title="Taux de réussite par cible vs probabilité annoncée", yaxis_tickformat=".0%", **LAYOUT)
    return fig


def model_comparison(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "brier" in df.columns:
        fig.add_trace(go.Bar(x=df["model_name"], y=df["brier"], name="Brier (plus bas = mieux)",
                             marker_color=PALETTE[4]))
    if "auc" in df.columns:
        fig.add_trace(go.Scatter(x=df["model_name"], y=df["auc"], mode="markers", yaxis="y2",
                                 marker=dict(size=14, color=PALETTE[2]), name="AUC"))
    fig.update_layout(title="Comparaison des modèles (hors échantillon)", yaxis=dict(title="Brier"),
                      yaxis2=dict(title="AUC", overlaying="y", side="right", range=[0.4, 0.6]), **LAYOUT)
    return fig


def skill_plot(skill: pd.DataFrame) -> go.Figure:
    """skill : colonnes model / target / delta_brier / ci_low / ci_high (WalkForwardResult)
    ou model_name / target / value / ci_low / ci_high (format base de données)."""
    df = skill.copy()
    if "delta_brier" in df.columns:
        df = df.rename(columns={"model": "model_name", "delta_brier": "value"})
    if "model_name" not in df.columns and "model" in df.columns:
        df = df.rename(columns={"model": "model_name"})
    if "value" not in df.columns and "delta_brier_vs_baseline" in df.columns:
        df = df.rename(columns={"delta_brier_vs_baseline": "value"})
    df = df.dropna(subset=["value", "ci_low", "ci_high"])
    df["label"] = df["model_name"].astype(str) + " · " + df["target"].astype(str) + "x"
    df["err_lo"] = df["value"] - df["ci_low"]
    df["err_hi"] = df["ci_high"] - df["value"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["value"], y=df["label"], mode="markers",
                             error_x=dict(type="data", symmetric=False, array=df["err_hi"], arrayminus=df["err_lo"],
                                          color=PALETTE[7]),
                             marker=dict(size=10, color=[PALETTE[1] if v > 0 else PALETTE[3] for v in df["value"]]),
                             name="Δ Brier (IC 95 %)"))
    fig.add_vline(x=0, line_color="white", line_width=1)
    fig.update_layout(title="Gain de Brier vs baseline (positif = meilleur) avec IC 95 %",
                      xaxis_title="Δ Brier (baseline − modèle)", **LAYOUT)
    return fig


def calibration_plot(table: list[dict]) -> go.Figure:
    df = pd.DataFrame(table)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#888"), name="parfait"))
    fig.add_trace(go.Scatter(x=df["predicted"], y=df["observed"], mode="markers+lines",
                             marker=dict(size=np.clip(df["n"], 6, 26), color=PALETTE[0]), name="observé"))
    fig.update_layout(title="Calibration (probabilité annoncée vs fréquence observée)",
                      xaxis_title="probabilité annoncée", yaxis_title="fréquence observée",
                      xaxis_range=[0, 1], yaxis_range=[0, 1], **LAYOUT)
    return fig


def performance_over_time(periods: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=periods["manche_max"], y=periods["taux"], mode="lines+markers",
                             name="taux observé", line=dict(color=PALETTE[1])))
    fig.add_trace(go.Scatter(x=periods["manche_max"], y=periods["prob_moyenne"], mode="lines",
                             name="probabilité annoncée", line=dict(color=PALETTE[0], dash="dot")))
    fig.update_layout(title="Évolution des performances par bloc chronologique", yaxis_tickformat=".0%", **LAYOUT)
    return fig


def predictions_timeline(df: pd.DataFrame) -> go.Figure:
    d = df.copy()
    d["y"] = d["target"]
    color_map = {"SUCCESS": PALETTE[1], "FAILURE": PALETTE[3], "PENDING": PALETTE[2]}
    fig = go.Figure()
    for status, grp in d.groupby("status"):
        fig.add_trace(go.Scatter(x=grp["anchor_round_id"], y=grp["target"], mode="markers", name=status,
                                 marker=dict(color=color_map.get(status, "#888"), size=7, symbol="circle")))
    fig.update_layout(title="Prédictions et validations", xaxis_title="manche d'ancrage", yaxis_title="cible",
                      **LAYOUT)
    return fig


def importance_plot(imp: pd.DataFrame, top: int = 15) -> go.Figure:
    d = imp.groupby("feature")["importance"].mean().sort_values(ascending=True).tail(top).reset_index()
    fig = go.Figure(go.Bar(x=d["importance"], y=d["feature"], orientation="h", marker_color=PALETTE[0]))
    fig.update_layout(title="Variables les plus utilisées par le modèle (gain)", **LAYOUT)
    return fig


def fill_rate_curve(multipliers, targets=None) -> go.Figure:
    """Taux de remplissage : part des fenêtres de 3 manches dont le MAX atteint la cible.

    C'est la mesure honnête d'une stratégie « viser t et se contenter de la première
    manche qui l'atteint » : plus la cible est haute, plus rare est le remplissage.
    """
    m = np.asarray(multipliers, dtype=float)
    targets = targets or [1.2, 1.5, 2.0, 2.5, 3.0, 5.0, 7.5, 10.0, 20.0]
    if len(m) < 4:
        return go.Figure()
    max3 = np.maximum.reduce([m[:-2], m[1:-1], m[2:]])
    rates = [float(np.mean(max3 >= t)) for t in targets]
    expected_value = [float(np.mean(np.where(m >= t, t, 0))) for t in targets]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=[f"{t:g}x" for t in targets], y=rates, name="taux de remplissage (3 manches)",
                         marker_color=PALETTE[1]), secondary_y=False)
    fig.add_trace(go.Scatter(x=[f"{t:g}x" for t in targets], y=expected_value, mode="lines+markers",
                             name="gain espéré par manche (unités misées)",
                             line=dict(color=PALETTE[3], width=2)), secondary_y=True)
    fig.add_hline(y=1.0, line_dash="dash", line_color="#888", secondary_y=True,
                  annotation_text="seuil de rentabilité (1.00)")
    fig.update_layout(title="Taux de remplissage et gain espéré par cible — données réelles", **LAYOUT)
    fig.update_yaxes(title="taux de remplissage", tickformat=".0%", secondary_y=False)
    fig.update_yaxes(title="gain espéré par manche", secondary_y=True)
    return fig
