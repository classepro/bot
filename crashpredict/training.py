"""Pipeline d'entraînement complet : backtest walk-forward + rapport persistant.

`run_training` est l'unique point d'entrée utilisé par la CLI, l'interface et
les scripts. Il :
  1. charge l'historique depuis la base ;
  2. exécute le walk-forward (splits chronologiques, purge de 3 manches) ;
  3. calcule les métriques, les tests de significativité (Holm) et l'apport
     propre de chaque famille de features ;
  4. choisit le modèle à déployer (baseline si aucun gain significatif) ;
  5. persiste le run, les métriques et les prédictions historiques en base ;
  6. sauvegarde les artefacts nécessaires à la prédiction en direct.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import db
from .backtesting.metrics import brier_score, log_loss, roc_auc, summarize_probabilities
from .backtesting.significance import paired_skill_test
from .backtesting.walk_forward import WalkForwardResult, choose_deployed_model, run_walk_forward
from .config import CONFIG, MODEL_DIR
from .features import build_dataset, build_features, feature_columns
from .models import build_model

DEFAULT_MODELS = ["logistic", "random_forest", "lightgbm", "xgboost"]
OPTIONAL_MODELS = ["catboost", "extra_trees", "mlp"]


def load_history() -> tuple[pd.Series, pd.Series, pd.Series]:
    rows = db.get_rounds()
    if not rows:
        raise RuntimeError("Base vide : lancez d'abord `python scripts/import_data.py`.")
    m = pd.Series([r["multiplier"] for r in rows], dtype=float)
    rid = pd.Series([r["round_id"] for r in rows])
    ts = pd.Series([r["ts"] for r in rows])
    return m, rid, ts


def feature_group_ablation(
    multipliers: pd.Series,
    round_ids: pd.Series,
    target: float,
    model_name: str = "logistic",
    initial_train: int | None = None,
    step: int | None = None,
    max_folds: int = 1,
) -> pd.DataFrame:
    """Mesure l'apport marginal de chaque famille de features (test d'honnêteté).

    Si retirer une famille ne dégrade pas le Brier, cette famille n'apporte rien.
    """
    initial_train = initial_train or CONFIG.backtest.initial_train
    step = step or CONFIG.backtest.step
    ds = build_dataset(multipliers, round_ids, targets=[target])
    fcols = feature_columns(ds)
    groups = {
        "toutes": fcols,
        "sans_fréquences": [c for c in fcols if "freq" not in c],
        "sans_moyennes": [c for c in fcols if "mean" not in c and "ratio" not in c],
        "sans_fenêtres_longues": [c for c in fcols if not any(w in c for w in ("_500", "_100"))],
        "sans_séries": [c for c in fcols if "streak" not in c],
        "sans_courant": [c for c in fcols if c not in ("f_cur", "f_log_cur", "f_is_instant_1x", "f_is_low_1_5", "f_delta_log", "f_pct_change")],
    }
    rows = []
    n = len(ds)
    usable_end = n - CONFIG.rules.horizon
    starts = list(range(initial_train, usable_end, step))[:max_folds]
    for gname, cols in groups.items():
        ys, ps = [], []
        for start in starts:
            train_end = max(0, start - CONFIG.rules.horizon)
            train = ds.iloc[max(0, train_end - CONFIG.backtest.max_train_window):train_end]
            test = ds.iloc[start : min(start + step, usable_end)]
            ytr = train[f"y_{target}"].to_numpy()
            yte = test[f"y_{target}"].to_numpy()
            ok_tr, ok_te = np.isfinite(ytr), np.isfinite(yte)
            if ok_tr.sum() < 100 or ok_te.sum() < 10 or not cols:
                continue
            model = build_model(model_name, random_state=CONFIG.training.random_state)
            model.fit(train[cols][ok_tr], ytr[ok_tr].astype(int))
            p = model.predict_proba(test[cols][ok_te])[:, 1]
            ys.append(yte[ok_te]); ps.append(p)
        if ys:
            y = np.concatenate(ys); p = np.concatenate(ps)
            rows.append({"groupe": gname, "n_features": len(cols), "n": len(y),
                         "brier": brier_score(y, p), "auc": roc_auc(y, p)})
    return pd.DataFrame(rows)


def run_training(
    targets: Iterable[float] | None = None,
    models: Iterable[str] | None = None,
    initial_train: int | None = None,
    step: int | None = None,
    max_folds: int | None = None,
    with_ablation: bool = False,
    ablation_target: float = 2.0,
    progress=None,
) -> dict:
    """Entraînement + backtest complet, persisté en base."""
    targets = list(targets or CONFIG.targets.fixed)
    models = list(models or DEFAULT_MODELS)
    db.init_db()
    m, rid, ts = load_history()
    fingerprint = db.data_fingerprint()
    run_id = db.create_run(
        {
            "targets": targets, "models": models, "initial_train": initial_train or CONFIG.backtest.initial_train,
            "step": step or CONFIG.backtest.step, "max_folds": max_folds,
            "horizon": CONFIG.rules.horizon, "purge": CONFIG.backtest.purge,
            "max_train_window": CONFIG.backtest.max_train_window,
        },
        fingerprint,
    )
    t0 = time.time()
    try:
        result = run_walk_forward(
            m, rid, targets=targets, models=models,
            initial_train=initial_train, step=step, max_folds=max_folds,
            progress=progress, record_predictions=True, engine="base",
            max_warmup=int(initial_train or CONFIG.backtest.initial_train) + CONFIG.backtest.max_train_window,
        )
        ablation = None
        if with_ablation:
            ablation = feature_group_ablation(m, rid, ablation_target, initial_train=initial_train, step=step)
        _persist(run_id, result, ablation, m, rid, ts)
        summary = result.to_serializable()
        summary["ablation"] = ablation.to_dict(orient="records") if ablation is not None else None
        summary["run_id"] = run_id
        db.finish_run(run_id, "DONE", summary)
        db.set_app_config("last_training_run", run_id)
        db.set_app_config("deployed_model", result.summary.get("deployed_model"))
        db.set_app_config("last_training_seconds", round(time.time() - t0, 1))
        _generate_reports()
        return summary
    except Exception as exc:
        db.finish_run(run_id, "FAILED", {"error": repr(exc)})
        raise


def _persist(run_id: str, result: WalkForwardResult, ablation: pd.DataFrame | None,
             m: pd.Series, rid: pd.Series, ts: pd.Series) -> None:
    # métriques
    metric_rows = []
    for _, r in result.metrics.iterrows():
        metric_rows.append(
            {"model_name": r["model"], "target": r["target"], "metric": r["metric"],
             "value": r["value"], "n": r.get("n")}
        )
    # skill (significativité)
    for _, r in result.skill.iterrows():
        metric_rows.append(
            {"model_name": r["model"], "target": r["target"],
             "metric": "delta_brier_vs_baseline", "value": r["delta_brier"],
             "ci_low": r["ci_low"], "ci_high": r["ci_high"], "n": r["n"]}
        )
        metric_rows.append(
            {"model_name": r["model"], "target": r["target"],
             "metric": "p_value_holm", "value": r["p_adjusted_holm"], "n": r["n"]}
        )
    if not result.skill.empty:
        for model, grp in result.skill.groupby("model"):
            metric_rows.append(
                {"model_name": model, "target": None, "metric": "delta_brier_moyen",
                 "value": grp["delta_brier"].mean(), "n": int(grp["n"].sum())}
            )
    db.add_metrics(run_id, metric_rows)

    # prédictions historiques (1 par manche, modèle déployé, toutes cibles)
    if result.predictions is not None and not result.predictions.empty:
        preds = result.predictions.copy()
        preds["run_id"] = run_id
        _insert_backtest_predictions(preds)

    # artefacts pour la prédiction en direct
    _save_artifacts(result, m, rid, ts)


def _insert_backtest_predictions(preds: pd.DataFrame) -> None:
    fields = ("id", "created_at", "model_name", "model_version", "strategy", "target", "probability",
              "status", "anchor_round_id", "n_rounds_seen", "window_round_ids", "window_values",
              "actual_max", "validated_at", "explanation", "backtest_run_id", "is_backtest")
    ts_now = db.utcnow()
    with db.connect() as conn:
        existing = conn.execute("SELECT COUNT(*) c FROM predictions WHERE is_backtest=1").fetchone()["c"]
        if existing:
            conn.execute("DELETE FROM predictions WHERE is_backtest=1")
        rows = []
        for i, r in enumerate(preds.itertuples(index=False)):
            rows.append(
                (
                    f"BT{r.run_id[-8:]}{i:06d}", r.ts or ts_now, r.model, "backtest", "historique",
                    float(r.target), float(r.probability), r.status, int(r.anchor_round_id), None,
                    json.dumps(r.window_round_ids), json.dumps(r.window_values),
                    r.actual_max, ts_now if r.status != "PENDING" else None,
                    json.dumps({"source": "walk_forward", "fold": int(r.fold)}),
                    r.run_id, 1,
                )
            )
        conn.executemany(
            f"INSERT OR REPLACE INTO predictions ({', '.join(fields)}) VALUES ({', '.join('?' for _ in fields)})",
            rows,
        )


def _save_artifacts(result: WalkForwardResult, m: pd.Series, rid: pd.Series, ts: pd.Series) -> None:
    """Sauvegarde le jeu de probabilités, les métriques, le skill et les importances."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    result.probabilities.to_parquet(MODEL_DIR / "last_probabilities.parquet") if _has_parquet() else result.probabilities.to_pickle(MODEL_DIR / "last_probabilities.pkl")
    result.metrics.to_pickle(MODEL_DIR / "last_metrics.pkl")
    result.skill.to_pickle(MODEL_DIR / "last_skill.pkl")
    result.predictions.to_pickle(MODEL_DIR / "last_predictions.pkl")
    if result.importance is not None:
        result.importance.to_pickle(MODEL_DIR / "last_importance.pkl")
    with open(MODEL_DIR / "last_summary.json", "w") as fh:
        json.dump(result.to_serializable(), fh, indent=2, default=str)


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def _generate_reports() -> None:
    """Régénère les rapports HTML/Markdown après un entraînement (sans bloquer en cas d'erreur)."""
    try:
        from .reporting import write_reports

        write_reports()
    except Exception as exc:  # pragma: no cover - la génération ne doit jamais casser l'entraînement
        print(f"[warn] rapport non généré: {exc!r}")
