"""Feature engineering causal — aucune fuite de données.

Convention stricte : `build_features` calcule, pour chaque ligne `i`, des
statistiques qui n'utilisent QUE les manches `<= i`. La cible de validation
reste `MAX(m[i+1..i+3])` et n'est jamais utilisée dans les features (fonction
`build_targets`, séparée, et jamais appelée pendant la construction du X).

Contrôles automatiques (tests/test_no_leakage.py) :
  * fenêtres glissantes strictement causales (pas de center=True) ;
  * la cible n'apparaît dans aucune colonne de features ;
  * décaler le futur ne doit pas modifier les features du passé.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CONFIG

# Colonnes utilisées par les modèles (les autres restent descriptives)
FEATURE_PREFIX = "f_"


def _rolling_past(s: pd.Series, window: int, func: str) -> pd.Series:
    """Statistique sur les `window` valeurs PRÉCÉDENTES (valeur courante exclue)."""
    return getattr(s.shift(1).rolling(window, min_periods=max(3, window // 10)), func)()


def build_features(multipliers: pd.Series) -> pd.DataFrame:
    """Construit la matrice de features causales à partir des coefficients seuls.

    `multipliers` doit être ordonné chronologiquement (index = position de manche).
    Chaque ligne i décrit l'information disponible APRÈS la manche i.
    """
    m = multipliers.astype(float).reset_index(drop=True)
    # log1p réduit l'influence des queues extrêmes (5 000x, 127 000x)
    lm = np.log(np.maximum(m, 1.0))
    f = pd.DataFrame(index=m.index)

    f["f_cur"] = m
    f["f_log_cur"] = lm
    f["f_is_instant_1x"] = (m <= 1.0).astype(float)
    f["f_is_low_1_5"] = (m < 1.5).astype(float)

    # moyennes / médianes / dispersion sur fenêtres passées
    for w in (5, 10, 20, 50, 100, 500):
        f[f"f_mean_log_{w}"] = _rolling_past(lm, w, "mean")
        f[f"f_max_{w}"] = _rolling_past(m, w, "max")
    for w in (20, 100, 500):
        f[f"f_median_{w}"] = _rolling_past(m, w, "median")
        f[f"f_std_log_{w}"] = _rolling_past(lm, w, "std")

    # fréquences empiriques passées par catégorie de coefficient
    for th in (1.5, 2.0, 3.0, 5.0, 10.0):
        ind = (m >= th).astype(float).rename(f"ind_{th}")
        for w in (100, 500):
            f[f"f_freq_ge_{th}_{w}"] = ind.shift(1).rolling(w, min_periods=max(10, w // 20)).mean()

    # séries (nombre de manches consécutives sous 1.5x / au-dessus de 2x)
    low = (m < 1.5).astype(int)
    high = (m >= 2.0).astype(int)
    f["f_streak_low"] = _streak(low)
    f["f_streak_high"] = _streak(high)

    # variations temporelles
    f["f_delta_log"] = lm.diff().fillna(0.0)
    f["f_pct_change"] = m.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    f["f_ratio_mean5_50"] = f["f_mean_log_5"] - f["f_mean_log_50"]

    return f


def _streak(binary: pd.Series) -> pd.Series:
    """Longueur de la série consécutive se terminant à i (inclut i)."""
    out = np.zeros(len(binary), dtype=float)
    run = 0
    for i, v in enumerate(binary.values):
        run = run + 1 if v else 0
        out[i] = run
    return pd.Series(out, index=binary.index)


def build_targets(multipliers: pd.Series, targets: list[float] | None = None, horizon: int | None = None) -> pd.DataFrame:
    """Cibles binaires : 1 si MAX(m[i+1..i+horizon]) >= t.

    Les `horizon` dernières lignes sont NaN (pas de futur disponible) : elles
    doivent être exclues, jamais imputées.
    """
    targets = targets or CONFIG.targets.grid
    horizon = horizon or CONFIG.rules.horizon
    m = multipliers.astype(float).reset_index(drop=True)
    out = pd.DataFrame(index=m.index)
    for t in targets:
        out[f"y_{t}"] = np.nan
    n = len(m)
    if n <= horizon:
        return out
    windows = np.vstack([m.values[k : n - horizon + k] for k in range(1, horizon + 1)])
    max_future = windows.max(axis=0)
    for t in targets:
        col = np.full(n, np.nan)
        col[: n - horizon] = (max_future >= t).astype(float)
        out[f"y_{t}"] = col
    return out


def build_future_window(multipliers: pd.Series, horizon: int | None = None) -> pd.DataFrame:
    """Valeurs des `horizon` manches suivantes (pour la validation, pas les features)."""
    horizon = horizon or CONFIG.rules.horizon
    m = multipliers.astype(float).reset_index(drop=True)
    out = pd.DataFrame(index=m.index)
    for k in range(1, horizon + 1):
        out[f"next_{k}"] = m.shift(-k)
    return out


def build_dataset(multipliers: pd.Series, round_ids: pd.Series | None = None,
                  targets: list[float] | None = None, timestamps: pd.Series | None = None) -> pd.DataFrame:
    """Features + cibles + fenêtre future + round_id (+ timestamp informatif)."""
    feats = build_features(multipliers)
    tgt = build_targets(multipliers, targets)
    fut = build_future_window(multipliers)
    df = pd.concat([feats, tgt, fut], axis=1)
    if round_ids is not None:
        df.insert(0, "round_id", pd.Series(round_ids).values)
    df.insert(1 if round_ids is not None else 0, "multiplier", multipliers.values)
    if timestamps is not None:
        df["ts"] = pd.Series(timestamps).values
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
