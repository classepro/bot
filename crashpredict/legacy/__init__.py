"""Rétro-ingénierie et reproduction des générations précédentes du robot.

Contenu :
  * `robotv45` — reproduction fidèle de ROBOT V4.5 (Standard / Avancé / Pro) ;
  * `backtest` — évaluation chronologique stricte des trois modes sur les
                 manches réelles, sous les deux horizons (3 et 5 manches) ;
  * `compare`  — confrontation aux modèles ML et à la loi marginale de référence.

Ces fonctions n'écrivent pas en base et ne consultent jamais le futur : elles
sont pures, testables, et utilisées telles quelles par le backtest, le rapport
et l'interface.
"""
from __future__ import annotations

from .backtest import (
    LEGACY_STRATEGIES,
    BettingEvaluation,
    ModeEvaluation,
    declared_probabilities,
    evaluate_betting,
    evaluate_mode,
    evaluate_policies,
    legacy_targets,
    reference_probabilities,
    validation_thresholds,
)
from .compare import (
    claim_quality,
    labelling_bias_curve,
    load_ml_probabilities,
    ml_vs_reference,
    round_of_success,
)
from .robotv45 import (
    MODES,
    LegacyPrediction,
    aggregate_odds,
    calculate_trend,
    displayed_probability,
    js_to_fixed,
    legacy_target,
    parse_odds_input,
    predict,
    prediction_from_sequence,
)

__all__ = [
    "MODES", "LegacyPrediction", "aggregate_odds", "calculate_trend", "displayed_probability",
    "js_to_fixed", "legacy_target", "parse_odds_input", "predict", "prediction_from_sequence",
    "LEGACY_STRATEGIES", "ModeEvaluation", "BettingEvaluation", "declared_probabilities",
    "evaluate_betting", "evaluate_mode", "evaluate_policies", "legacy_targets",
    "reference_probabilities", "validation_thresholds",
    "claim_quality", "labelling_bias_curve", "load_ml_probabilities", "ml_vs_reference",
    "round_of_success",
]
