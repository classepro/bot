"""Métriques probabilistes — implémentées explicitement (pas de dépendance cachée)."""
from __future__ import annotations

import numpy as np


def _arr(y, p, clip: bool = True) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    return (y, np.clip(p, 1e-9, 1 - 1e-9)) if clip else (y, p)


def brier_score(y, p) -> float:
    y, p = _arr(y, p, clip=False)
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def log_loss(y, p) -> float:
    y, p = _arr(y, p)
    if not len(y):
        return float("nan")
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def roc_auc(y, p) -> float:
    y, p = _arr(y, p)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # correction des ex-aequo
    sorted_p = p[order]
    i = 0
    while i < len(sorted_p):
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def calibration_table(y, p, bins: int = 10) -> list[dict]:
    y, p = _arr(y, p)
    if not len(y):
        return []
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if mask.sum() == 0:
            continue
        out.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": int(mask.sum()),
                "predicted": float(p[mask].mean()),
                "observed": float(y[mask].mean()),
                "gap": float(y[mask].mean() - p[mask].mean()),
            }
        )
    return out


def accuracy_at_half(y, p) -> float:
    y, p = _arr(y, p)
    return float(np.mean((p >= 0.5) == (y >= 0.5))) if len(y) else float("nan")


def summarize_probabilities(y, p, prob_baseline=None) -> dict:
    """Résumé complet d'un couple (prédiction, vérité), avec gain vs baseline."""
    y, p = _arr(y, p, clip=True)
    out = {
        "n": int(len(y)),
        "brier": brier_score(y, p),
        "log_loss": log_loss(y, p),
        "auc": roc_auc(y, p),
        "accuracy": accuracy_at_half(y, p),
        "observed_rate": float(y.mean()) if len(y) else float("nan"),
        "mean_probability": float(p.mean()) if len(p) else float("nan"),
    }
    if prob_baseline is not None:
        b = np.asarray(prob_baseline, dtype=float)[: len(y)]
        b = np.clip(b[np.isfinite(b)], 1e-9, 1 - 1e-9)
        if len(b) == len(y):
            out["brier_baseline"] = brier_score(y, np.asarray(prob_baseline, dtype=float)[: len(y)])
            out["skill_vs_baseline"] = 1 - out["brier"] / out["brier_baseline"] if out["brier_baseline"] > 0 else float("nan")
    return out


def rolling_brier(y, p, window: int = 1000) -> list[float]:
    y, p = _arr(y, p)
    err = (p - y) ** 2
    if len(err) < window:
        return [float(err.mean())] if len(err) else []
    kernel = np.ones(window) / window
    return list(np.convolve(err, kernel, mode="valid"))
