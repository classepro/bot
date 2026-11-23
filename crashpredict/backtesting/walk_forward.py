"""Backtesting chronologique walk-forward (aucun mélange aléatoire).

Schéma pour chaque cible `t` :

    |<------------ train (index < s) ----------->| purge |<-- test -->|
    s = initial_train, s+step, s+2*step, ...
    purge = horizon (3 manches) pour que la cible du dernier point
    d'entraînement (MAX des 3 suivantes) soit entièrement connue avant le test.

À chaque pas, les modèles sont RÉENTRAÎNÉS sur [0, s) seulement, puis évaluent
les manches [s, s+step). Les baselines empiriques utilisent la même information.

Sorties : probabilités par (modèle, cible, manche), métriques, tests de
significativité vs baseline, statistiques de la règle des 3 manches.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from ..config import CONFIG
from ..features import build_dataset, feature_columns
from ..models import build_model, empirical_validation_probability
from ..models.statistical import MarkovChain, ParetoTail
from .metrics import summarize_probabilities
from .significance import holm_correction, paired_skill_test


@dataclass
class WalkForwardResult:
    probabilities: pd.DataFrame          # round_id, model, target, y, p, fold
    metrics: pd.DataFrame
    skill: pd.DataFrame
    summary: dict = field(default_factory=dict)
    predictions: pd.DataFrame | None = None   # prédictions au format "1 prédiction = 1 cible"
    engine: str = "base"
    feature_importance: pd.DataFrame | None = None
    importance: pd.DataFrame | None = None

    def to_serializable(self) -> dict:
        return {
            "n_probabilities": int(len(self.probabilities)),
            "models": sorted(self.probabilities["model"].unique().tolist()),
            "targets": sorted(self.probabilities["target"].unique().tolist()),
            "folds": int(self.probabilities["fold"].nunique()),
            **self.summary,
        }


def _fit_predict(model_name: str, X_train: pd.DataFrame, y_train: np.ndarray,
                 X_test: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """Entraîne un modèle sklearn calibré et renvoie les probabilités du bloc de test."""
    model = build_model(model_name, random_state=seed)
    y = np.asarray(y_train, dtype=int)
    if len(np.unique(y)) < 2:
        return np.full(len(X_test), float(y.mean()))
    model.fit(X_train, y)
    proba = model.predict_proba(X_test)
    return np.asarray(proba)[:, 1] if proba.ndim == 2 else np.asarray(proba)


def rolling_empirical(history: np.ndarray, future_values: np.ndarray, target: float,
                      window: int, horizon: int) -> np.ndarray:
    """P(validation) recalculée à chaque manche du bloc de test, sans voir le futur.

    Causality: pour prédire la manche j du bloc, on ne connaît que l'historique
    plus les manches strictement antérieures à j (donc aucune des 3 manches
    qui servent à valider la prédiction).
    """
    h = np.asarray(history, dtype=float)
    f = np.asarray(future_values, dtype=float)
    if len(f) == 0:
        return np.array([])
    hist_hits = np.concatenate([(h >= target).astype(float), (f >= target).astype(float)])
    hist_cs = np.concatenate([[0.0], np.cumsum(hist_hits)])
    n_hist = len(h)
    n_total = len(h) + len(f)
    idx = np.arange(len(f))
    end_pos = n_hist + idx                      # fenêtre [end_pos-window, end_pos)
    start_pos = np.maximum(0, end_pos - window)
    hits = hist_cs[end_pos] - hist_cs[start_pos]
    n_used = np.maximum(1, end_pos - start_pos)
    p_round = np.clip(hits / n_used, 0, 1)
    return 1 - (1 - p_round) ** horizon


def training_slice_end(start: int, purge: int) -> int:
    """Fin des lignes utilisables pour l'entraînement quand le test commence à `start`.

    La ligne i du dataset contient la cible MAX(m[i+1..i+3]) : elle n'est
    connaissable qu'à la manche i+3. Pour un test démarrant à `start`, on
    n'entraîne donc que sur i <= start - purge - 1.
    """
    return max(0, start - purge)


def run_walk_forward(
    multipliers: pd.Series,
    round_ids: pd.Series | None = None,
    targets: Iterable[float] | None = None,
    models: Iterable[str] | None = None,
    initial_train: int | None = None,
    step: int | None = None,
    max_folds: int | None = None,
    horizon: int | None = None,
    seed: int = 42,
    progress=None,
    record_predictions: bool = True,
    engine: str = "base",
) -> WalkForwardResult:
    """Exécute le walk-forward complet. `progress(fold, n_folds, msg)` optionnel."""
    targets = list(targets or CONFIG.targets.fixed)
    models = list(models or ["lightgbm", "random_forest", "logistic", "xgboost"])
    horizon = horizon or CONFIG.rules.horizon
    initial_train = initial_train or CONFIG.backtest.initial_train
    step = step or CONFIG.backtest.step
    purge = horizon
    max_train = CONFIG.backtest.max_train_window

    mult = pd.Series(multipliers).astype(float).reset_index(drop=True)
    if round_ids is None:
        round_ids = pd.Series(np.arange(len(mult)), name="round_id")
    rid = pd.Series(round_ids).reset_index(drop=True)

    ds = build_dataset(mult, rid, targets=targets)
    fcols = feature_columns(ds)
    n = len(ds)
    usable_end = n - horizon  # après cette position, pas de futur -> pas de cible

    predicted_model = "baseline_rolling_long"
    starts = list(range(initial_train, usable_end, step))
    if not starts:
        raise ValueError(
            f"historique trop court pour un walk-forward (n={n}, initial_train={initial_train})"
        )
    if max_folds:
        starts = starts[:max_folds]

    rows: list[dict] = []
    fold_importance: list[pd.DataFrame] = []
    t0 = time.time()

    for fold, start in enumerate(starts):
        train_end = training_slice_end(start, purge)
        train_start = max(0, train_end - max_train) if max_train else 0
        test_end = min(start + step, usable_end)
        if train_end < 200:
            continue
        train = ds.iloc[train_start:train_end]
        test = ds.iloc[start:test_end]
        Xtr_all, Xte_all = train[fcols], test[fcols]

        for t in targets:
            ycol = f"y_{t}"
            ytr = train[ycol].to_numpy()
            yte = test[ycol].to_numpy()
            valid = np.isfinite(ytr)  # uniquement le train : le test est jugé séparément
            keep = np.isfinite(yte)
            if valid.sum() < 100 or keep.sum() < 10:
                continue
            Xtr, ytr = Xtr_all[valid], ytr[valid]
            Xte, yte = Xte_all[keep], yte[keep]
            te_rounds = test["round_id"].to_numpy()[keep]
            te_hist = train["multiplier"].to_numpy()

            # ---------------- baselines empiriques (mêmes données, même causalité) ----------------
            # Historique = tout ce qui est connu AVANT la fenêtre de test ; on y ajoute
            # les manches du bloc au fur et à mesure (après avoir prédit).
            tests_vals = test["multiplier"].to_numpy()[keep]
            base_preds = {
                "baseline_fixed": np.array([empirical_validation_probability(te_hist, t, horizon)[0]] * len(Xte)),
                "baseline_rolling_long": rolling_empirical(te_hist, tests_vals, t, CONFIG.probabilities.long_window, horizon),
                "baseline_rolling_short": rolling_empirical(te_hist, tests_vals, t, CONFIG.probabilities.short_window, horizon),
            }

            # ---------------- modèles ML ----------------
            ml_preds = {}
            for name in models:
                try:
                    p = _fit_predict(name, Xtr, ytr, Xte, seed=seed)
                except Exception as exc:  # un modèle qui casse ne doit pas arrêter le backtest
                    ml_preds[name] = np.full(len(Xte), np.nan)
                    print(f"  [warn] {name} cible {t} fold {fold}: {exc!r}")
                    continue
                ml_preds[name] = p
                if name in ("lightgbm", "xgboost", "random_forest") and fold == 0:
                    try:
                        clf = build_model(name, random_state=seed).fit(Xtr, ytr.astype(int))
                        est = clf[-1] if hasattr(clf, "__getitem__") else clf
                        imp = getattr(est, "feature_importances_", None)
                        if imp is not None:
                            fold_importance.append(
                                pd.DataFrame({"model": name, "target": t, "feature": fcols, "importance": imp})
                            )
                    except Exception:
                        pass

            # ---------------- collecte ----------------
            for name, p in {**base_preds, **ml_preds}.items():
                for k in range(len(p)):
                    rows.append(
                        {
                            "fold": fold,
                            "round_id": int(te_rounds[k]),
                            "model": name,
                            "target": t,
                            "y": float(yte[k]),
                            "p": float(p[k]) if np.isfinite(p[k]) else np.nan,
                        }
                    )
        if progress:
            progress(fold + 1, len(starts), f"fold {fold + 1}/{len(starts)} — {time.time() - t0:.1f}s")

    probs = pd.DataFrame(rows)
    metrics, skill = _aggregate(probs, targets)
    predicted_model = choose_deployed_model(metrics, skill)
    predictions = build_prediction_records(probs, ds, predicted_model) if record_predictions else None
    summary = {
        "duration_seconds": round(time.time() - t0, 1),
        "n_rounds": int(n),
        "initial_train": initial_train,
        "step": step,
        "purge": purge,
        "n_folds": len(starts),
        "targets": targets,
        "models": models,
        "deployed_model": predicted_model,
        "first_test_round": int(rid.iloc[starts[0]]) if starts else None,
        "last_test_round": int(rid.iloc[min(starts[-1] + step, usable_end) - 1]) if starts else None,
    }
    importance = pd.concat(fold_importance, ignore_index=True) if fold_importance else None
    return WalkForwardResult(
        probabilities=probs, metrics=metrics, skill=skill, summary=summary,
        predictions=predictions, importance=importance, engine=engine,
    )


def choose_deployed_model(metrics: pd.DataFrame, skill: pd.DataFrame, fallback: str = "baseline_rolling_long") -> str:
    """Modèle retenu pour la production : celui qui bat la baseline de façon significative.

    Sans avantage significatif (Holm), on retombe sur la baseline empirique —
    c'est la décision honnête : ne pas déployer un modèle qui fait moins bien.
    """
    if skill is None or skill.empty:
        return fallback
    good = skill[(skill["significant_holm"] == True) & (skill["delta_brier"] > 0)]  # noqa: E712
    if good.empty:
        return fallback
    best = good.groupby("model")["delta_brier"].mean().idxmax()
    return str(best)


def build_prediction_records(probs: pd.DataFrame, dataset: pd.DataFrame, model: str) -> pd.DataFrame:
    """Transforme les probabilités du backtest en prédictions horodatées (1 par manche).

    Une prédiction = (round d'ancrage, cible, probabilité, modèle) et sa
    validation automatique sur les 3 manches suivantes.
    """
    sub = probs[(probs["model"] == model) & np.isfinite(probs["p"])].copy()
    if sub.empty:
        return pd.DataFrame()
    ds = dataset.set_index("round_id")
    out = []
    for _, r in sub.iterrows():
        rid = int(r["round_id"])
        if rid not in ds.index:
            continue
        row = ds.loc[rid]
        window_ids = [rid + k for k in range(1, 4)]
        window_vals = [row.get(f"next_{k}") for k in range(1, 4)]
        if any(pd.isna(v) for v in window_vals):
            status, actual = "PENDING", None
        else:
            actual = float(max(window_vals))
            status = "SUCCESS" if actual >= float(r["target"]) else "FAILURE"
        out.append(
            {
                "anchor_round_id": rid,
                "target": float(r["target"]),
                "probability": float(r["p"]),
                "model": model,
                "window_round_ids": window_ids,
                "window_values": [None if pd.isna(v) else float(v) for v in window_vals],
                "actual_max": actual,
                "status": status,
                "fold": int(r["fold"]),
                "ts": row.get("ts"),
            }
        )
    return pd.DataFrame(out)


def _aggregate(probs: pd.DataFrame, targets: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Métriques par (modèle, cible) et tests appariés vs baseline glissante."""
    metric_rows: list[dict] = []
    skill_rows: list[dict] = []

    for t in targets:
        sub_t = probs[probs["target"] == t]
        if sub_t.empty:
            continue
        baseline = "baseline_rolling_long"
        base = sub_t[sub_t["model"] == baseline].set_index("round_id")["p"]
        y_base = sub_t[sub_t["model"] == baseline].set_index("round_id")["y"]
        for model in sorted(sub_t["model"].unique()):
            m = sub_t[sub_t["model"] == model].set_index("round_id")
            p = m["p"].dropna()
            if p.empty:
                continue
            y = m.loc[p.index, "y"]
            p_base = base.reindex(p.index)
            stats = summarize_probabilities(y.to_numpy(), p.to_numpy(), p_base.to_numpy())
            for key, value in stats.items():
                metric_rows.append({"model": model, "target": t, "metric": key, "value": value, "n": len(p)})
            if model != baseline:
                test = paired_skill_test(y.to_numpy(), p.to_numpy(), p_base.to_numpy())
                skill_rows.append(
                    {
                        "model": model,
                        "target": t,
                        "baseline": baseline,
                        **test,
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    skill = pd.DataFrame(skill_rows)
    if not skill.empty:
        # correction de Holm sur toute la famille (modèles × cibles)
        holm = holm_correction(dict(zip(skill["model"] + "@" + skill["target"].astype(str), skill["p_value"])))
        skill["p_adjusted_holm"] = [
            holm[f"{m}@{t}"]["p_adjusted"] for m, t in zip(skill["model"], skill["target"])
        ]
        skill["significant_holm"] = [
            bool(holm[f"{m}@{t}"]["significant_holm"]) for m, t in zip(skill["model"], skill["target"])
        ]
    return metrics, skill
