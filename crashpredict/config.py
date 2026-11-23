"""Configuration centrale du projet Crash Game AI Predictor.

Toutes les constantes méthodologiques sont ici (et nulle part ailleurs) afin
qu'il soit impossible d'utiliser une cible, une fenêtre ou une purge
différente entre l'entraînement, le backtest, la prédiction et l'interface.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXPORT_DIR = DATA_DIR / "exports"
MODEL_DIR = ROOT / "models_store"
REPORT_DIR = ROOT / "reports"

DB_PATH = Path(os.environ.get("CRASH_DB", DATA_DIR / "crash.db"))


@dataclass(frozen=True)
class Rules:
    """Règle de validation officielle du cahier des charges."""

    horizon: int = 3  # nombre de manches utilisées pour valider une prédiction
    min_multiplier: float = 1.0

    def validate(self, target: float, following: list[float]) -> bool:
        """SUCCÈS si MAX(manches suivantes) >= cible, sur exactement `horizon` manches."""
        window = list(following)[: self.horizon]
        if not window:
            raise ValueError("aucune manche disponible pour valider")
        return max(window) >= target


@dataclass(frozen=True)
class Targets:
    """Cibles proposées par le robot (stratégies prudente -> très ambitieuse)."""

    conservative: float = 1.5
    balanced: float = 2.0
    ambitious: float = 2.5
    very_ambitious: float = 5.0

    @property
    def fixed(self) -> list[float]:
        return [self.conservative, self.balanced, self.ambitious, self.very_ambitious]

    @property
    def grid(self) -> list[float]:
        """Grille de cibles testée en backtest (pour les courbes taux/probabilité)."""
        return [1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]


@dataclass(frozen=True)
class Backtest:
    """Paramètres de la validation chronologique (walk-forward)."""

    horizon: int = 3
    initial_train: int = 25_000      # manches minimales avant la 1re prédiction
    test_block: int = 3_000          # taille d'un bloc de test -> 1 réentraînement
    step: int = 3_000                # granularité de réentraînement (purge = horizon)
    min_history_features: int = 50   # historique minimal pour calculer des features fiables
    max_train_window: int = 40_000   # taille max de la fenêtre d'entraînement (réentraînement glissant)

    @property
    def purge(self) -> int:
        return self.horizon


@dataclass(frozen=True)
class Strategy:
    """Conversion probabilité -> cible, et seuils de décision."""

    desired_success_rate: float = 0.80   # taux de réussite visé pour la cible "optimisée"
    min_probability: float = 0.60        # sous ce seuil, le robot ne recommande pas de cible
    confidence_high: float = 0.85
    confidence_medium: float = 0.70


@dataclass(frozen=True)
class Probabilities:
    """Fenêtres (en manches) utilisées par les baselines empiriques."""

    short_window: int = 500
    long_window: int = 5_000
    sensitivity: float = 0.0  # force de réaction aux séries (0 = loi marginale pure)


@dataclass(frozen=True)
class Training:
    random_state: int = 42
    n_jobs: int = 2                  # le sandbox a 2 cœurs ; -1 sur une machine plus large
    calibration_folds: int = 2       # blocs de calibration chronologique
    quick_mode: bool = os.environ.get("CRASH_QUICK", "0") == "1"


@dataclass(frozen=True)
class Config:
    rules: Rules = field(default_factory=Rules)
    targets: Targets = field(default_factory=Targets)
    backtest: Backtest = field(default_factory=Backtest)
    probabilities: Probabilities = field(default_factory=Probabilities)
    strategy: Strategy = field(default_factory=Strategy)
    training: Training = field(default_factory=Training)
    db_path: Path = DB_PATH
    source_platform: str = "BC.Game"
    source_game: str = "crash"


CONFIG = Config()

__all__ = [
    "CONFIG", "Config", "Rules", "Targets", "Backtest", "Probabilities", "Strategy", "Training",
    "ROOT", "DATA_DIR", "RAW_DIR", "EXPORT_DIR", "MODEL_DIR", "REPORT_DIR", "DB_PATH",
]
