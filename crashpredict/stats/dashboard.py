"""Statistiques du robot : tout est recalculé depuis la base, jamais inventé.

Séparation stricte :
  * prédictions LIVE (is_backtest=0) : celles du robot en fonctionnement ;
  * prédictions de BACKTEST (is_backtest=1) : un échantillon par manche, utile
    pour les taux mais PAS indépendant (fenêtres glissantes).
Le tableau de bord affiche les deux et précise lequel on regarde.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import db
from ..config import CONFIG


def _load_predictions(backtest: bool = False) -> pd.DataFrame:
    rows = db.predictions(limit=1_000_000, backtest=backtest)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    return df


def dashboard_stats(backtest: bool = True) -> dict:
    df = _load_predictions(backtest=backtest)
    fp = db.data_fingerprint()
    out = {
        "n_rounds": fp["n"],
        "first_round": fp["first"],
        "last_round": fp["last"],
        "gaps": fp["gaps"],
        "chain_bad": fp["chain_bad"],
        "duplicates": fp["duplicates"],
        "n_predictions": int(len(df)),
        "n_success": int((df["status"] == "SUCCESS").sum()) if len(df) else 0,
        "n_failure": int((df["status"] == "FAILURE").sum()) if len(df) else 0,
        "n_pending": int((df["status"] == "PENDING").sum()) if len(df) else 0,
        "success_rate": float((df["status"] == "SUCCESS").mean()) if len(df) else None,
        "deployed_model": db.get_app_config("deployed_model"),
        "last_run": db.get_app_config("last_training_run"),
        "is_backtest": backtest,
    }
    if len(df):
        closed = df[df["status"].isin(["SUCCESS", "FAILURE"])]
        out["best_series"] = streaks(df, kind="success")["best"]
        out["worst_series"] = streaks(df, kind="failure")["best"]
        out["current_success_series"], out["current_failure_series"] = (
            streaks(df, kind="success")["current"], streaks(df, kind="failure")["current"]
        )
        if len(closed):
            best = closed[closed["status"] == "SUCCESS"].sort_values("target", ascending=False).head(1)
            out["best_validated_target"] = float(best["target"].iloc[0]) if len(best) else None
            out["brier"] = float(np.mean((closed["probability"] - (closed["status"] == "SUCCESS")) ** 2))
        else:
            out["best_validated_target"] = None
            out["brier"] = None
    return out


def streaks(df: pd.DataFrame, kind: str = "success") -> dict:
    """Séries consécutives (ordre chronologique) de succès ou d'échecs."""
    if df.empty:
        return {"current": 0, "best": 0}
    d = df.copy()
    d["_date"] = pd.to_datetime(d["created_at"], errors="coerce", utc=True)
    d = d.sort_values(["_date", "id"])
    wanted = "SUCCESS" if kind == "success" else "FAILURE"
    flags = (d["status"] == wanted).astype(int).tolist()
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    current = 0
    for f in reversed(flags):
        if f:
            current += 1
        else:
            break
    return {"current": current, "best": best}


def performance_by_target(backtest: bool = True, min_samples: int = 30) -> pd.DataFrame:
    df = _load_predictions(backtest=backtest)
    if df.empty:
        return pd.DataFrame()
    closed = df[df["status"].isin(["SUCCESS", "FAILURE"])].copy()
    if closed.empty:
        return pd.DataFrame()
    closed["success"] = (closed["status"] == "SUCCESS").astype(int)
    g = closed.groupby("target").agg(
        n=("success", "size"),
        succes=("success", "sum"),
        taux=("success", "mean"),
        prob_moyenne=("probability", "mean"),
        brier=("success", lambda s: np.nan),
    ).reset_index()
    g["brier"] = closed.groupby("target").apply(
        lambda x: float(np.mean((x["probability"] - (x["status"] == "SUCCESS")) ** 2)), include_groups=False
    ).values
    g["écart vs prob"] = g["taux"] - g["prob_moyenne"]
    g["fiable"] = g["n"] >= min_samples
    return g.sort_values("target")


def performance_by_model(backtest: bool = True) -> pd.DataFrame:
    df = _load_predictions(backtest=backtest)
    if df.empty:
        return pd.DataFrame()
    run_id = db.get_app_config("last_training_run")
    if run_id:
        rows = db.run_metrics(run_id)
        if rows:
            m = pd.DataFrame([dict(r) for r in rows])
            piv = m[m["metric"].isin(["brier", "auc", "delta_brier_moyen", "skill_vs_baseline"])]
            piv = piv.pivot_table(index="model_name", columns="metric", values="value", aggfunc="mean")
            return piv.reset_index()
    return pd.DataFrame()


def performance_by_period(backtest: bool = True, periods: int = 8) -> pd.DataFrame:
    df = _load_predictions(backtest=backtest)
    if df.empty:
        return pd.DataFrame()
    closed = df[df["status"].isin(["SUCCESS", "FAILURE"])].copy()
    if closed.empty:
        return pd.DataFrame()
    closed["_id"] = pd.to_numeric(closed["anchor_round_id"], errors="coerce")
    closed = closed.sort_values("_id")
    closed["bloc"] = pd.qcut(closed["_id"].rank(method="first"), periods, labels=False)
    closed["success"] = (closed["status"] == "SUCCESS").astype(int)
    g = closed.groupby("bloc").agg(
        manche_min=("_id", "min"), manche_max=("_id", "max"),
        n=("success", "size"), taux=("success", "mean"),
        prob_moyenne=("probability", "mean"),
    ).reset_index()
    return g


def prediction_history(limit: int = 500, backtest: bool = False) -> pd.DataFrame:
    df = _load_predictions(backtest=backtest)
    if df.empty:
        return pd.DataFrame()
    df["_date"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df = df.sort_values(["_date", "id"], ascending=False).head(limit)
    return df[["id", "created_at", "strategy", "target", "probability", "status",
               "anchor_round_id", "window_round_ids", "window_values", "actual_max", "model_name"]]


def last_metrics_run() -> tuple[str | None, pd.DataFrame, pd.DataFrame, dict | None]:
    run = db.latest_finished_run()
    if not run:
        return None, pd.DataFrame(), pd.DataFrame(), None
    rows = db.run_metrics(run["id"])
    df = pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
    summary = json.loads(run["summary"]) if run["summary"] else None
    metrics = df[~df["metric"].isin(["delta_brier_vs_baseline", "p_value_holm", "delta_brier_moyen", "auc"])] if not df.empty else df
    skill = df[df["metric"].isin(["delta_brier_vs_baseline", "p_value_holm"])] if not df.empty else df
    return run["id"], metrics, skill, summary


def feature_importances() -> pd.DataFrame:
    """Importances moyennes du dernier backtest (vide si non calculées)."""
    from ..config import MODEL_DIR
    path = MODEL_DIR / "last_importance.pkl"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_pickle(path)


def probability_samples(run_id: str | None = None) -> pd.DataFrame:
    """Échantillon (manche, modèle, cible, probabilité, vérité) du dernier backtest."""
    from ..config import MODEL_DIR
    parquet = MODEL_DIR / "last_probabilities.parquet"
    pkl = MODEL_DIR / "last_probabilities.pkl"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if pkl.exists():
        return pd.read_pickle(pkl)
    return pd.DataFrame()


def calibration_of_last_run(model_name: str | None = None, target: float | None = None) -> list[dict]:
    from ..backtesting.metrics import calibration_table
    from ..config import CONFIG

    df = probability_samples()
    if df.empty:
        return []
    model_name = model_name or db.get_app_config("deployed_model") or "baseline_rolling_long"
    target = target or CONFIG.targets.balanced
    sub = df[(df["model"] == model_name) & (df["target"] == target)].dropna(subset=["p", "y"])
    return calibration_table(sub["y"].to_numpy(), sub["p"].to_numpy())
