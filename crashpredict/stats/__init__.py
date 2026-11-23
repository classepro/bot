"""Agrégats pour le tableau de bord (toujours calculés depuis la base)."""
from __future__ import annotations

from .dashboard import (
    dashboard_stats,
    performance_by_model,
    performance_by_period,
    performance_by_target,
    prediction_history,
    streaks,
    last_metrics_run,
    feature_importances,
    calibration_of_last_run,
    probability_samples,
)

__all__ = [
    "dashboard_stats", "performance_by_model", "performance_by_period", "performance_by_target",
    "prediction_history", "streaks", "last_metrics_run",
    "feature_importances", "calibration_of_last_run", "probability_samples",
]
