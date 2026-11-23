"""Tests de significativité : le robot doit prouver qu'il bat une baseline.

Toutes les comparaisons sont appariées (mêmes manches, mêmes cibles) :
  * bootstrap d'IC 95 % sur la différence de Brier ;
  * test de différence moyenne + correction de Holm pour comparaisons multiples ;
  * test binomial sur le taux de réussite contre la probabilité de baseline.

Aucune conclusion n'est tirée d'un simple chiffre ; si l'IC contient 0,
on conclut explicitement à l'absence d'avantage démontré.
"""
from __future__ import annotations

import math

import numpy as np

try:  # scipy est une dépendance douce : fallback normal approx si absent
    from scipy import stats as _st
except Exception:  # pragma: no cover
    _st = None


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
    """IC bootstrap percentile de la moyenne d'un échantillon (apparié)."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_skill_test(y, p_model, p_base, n_boot: int = 2000, seed: int = 42) -> dict:
    """Compare deux jeux de probabilités sur les mêmes observations.

    delta_brier = Brier(baseline) - Brier(modèle)
      > 0  : le modèle est meilleur
      IC 95 % bootstrap sur la moyenne des différences élémentaires
    """
    y = np.asarray(y, dtype=float)
    pm = np.clip(np.asarray(p_model, dtype=float), 1e-9, 1 - 1e-9)
    pb = np.clip(np.asarray(p_base, dtype=float), 1e-9, 1 - 1e-9)
    mask = np.isfinite(y) & np.isfinite(pm) & np.isfinite(pb)
    y, pm, pb = y[mask], pm[mask], pb[mask]
    if len(y) < 30:
        return {"n": int(len(y)), "delta_brier": float("nan"), "p_value": float("nan"), "significant": False,
                "ci_low": float("nan"), "ci_high": float("nan"), "verdict": "échantillon insuffisant"}

    d = (pb - y) ** 2 - (pm - y) ** 2  # gain élémentaire de Brier
    mean, lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    if _st is not None:
        t = _st.ttest_1samp(d, 0.0, alternative="two-sided")
        pval = float(t.pvalue)
    else:  # pragma: no cover
        se = d.std(ddof=1) / np.sqrt(len(d))
        pval = float(2 * (1 - 0.5 * (1 + math.erf(abs(mean / se) / np.sqrt(2)))))
    significant = (lo > 0) or (hi < 0)
    if not significant:
        verdict = "aucun avantage démontré (IC 95 % contient 0)"
    elif mean > 0:
        verdict = "avantage statistiquement significatif du modèle"
    else:
        verdict = "le modèle est significativement PIRE que la baseline"
    return {
        "n": int(len(y)),
        "delta_brier": float(mean),
        "ci_low": lo,
        "ci_high": hi,
        "p_value": pval,
        "significant": bool(significant),
        "verdict": verdict,
    }


def binomial_test_vs_rate(successes: int, n: int, p0: float) -> dict:
    """Le taux de réussite observé est-il différent de la probabilité de baseline ?"""
    if n == 0:
        return {"n": 0, "observed": float("nan"), "p0": p0, "p_value": float("nan"), "significant": False}
    obs = successes / n
    if _st is not None:
        res = _st.binomtest(int(successes), int(n), p0, alternative="two-sided")
        pval = float(res.pvalue)
    else:  # pragma: no cover
        se = np.sqrt(p0 * (1 - p0) / n)
        pval = float(2 * (1 - 0.5 * (1 + math.erf(abs(obs - p0) / se / np.sqrt(2)))))
    return {
        "n": int(n),
        "observed": float(obs),
        "p0": float(p0),
        "p_value": pval,
        "significant": pval < 0.05,
        "verdict": "différence significative" if pval < 0.05 else "pas de différence significative",
    }


def holm_correction(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Correction de Holm-Bonferroni (famille de tests = modèles × cibles)."""
    items = sorted(((k, v) for k, v in p_values.items() if v is not None and np.isfinite(v)), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    for i, (key, p) in enumerate(items):
        adj = min(1.0, p * (m - i))
        out[key] = {"p_value": p, "p_adjusted": adj, "significant_holm": adj < alpha}
    for key in p_values:
        out.setdefault(key, {"p_value": None, "p_adjusted": None, "significant_holm": False})
    return out


def required_sample_size(p0: float, min_effect: float = 0.01, alpha: float = 0.05, power: float = 0.8) -> int:
    """Ordre de grandeur : manches nécessaires pour détecter un gain de Brier `min_effect`."""
    p1 = min(0.999, p0 + min_effect if p0 + min_effect < 1 else p0 - min_effect)
    p_bar = (p0 + p1) / 2
    z_a, z_b = 1.959963985, 0.841621234
    num = (z_a * np.sqrt(2 * p_bar * (1 - p_bar)) + z_b * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    return int(np.ceil(num / max((p1 - p0) ** 2, 1e-12)))


def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """IC de Wilson (score) d'une proportion — exact pour p proche de 0 ou 1.

    Choisi plutôt que l'intervalle normal : les taux des stratégies légacy sont
    souvent > 95 % ou < 5 %, zone où l'approximation normale est fausse.
    Retourne (proportion observée, borne basse, borne haute).
    """
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    z = 1.959963985 if abs(alpha - 0.05) < 1e-9 else float(_st.norm.ppf(1 - alpha / 2)) if _st is not None else 1.959963985
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0))
    return float(p), float(max(0.0, centre - half)), float(min(1.0, centre + half))
