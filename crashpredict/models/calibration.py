"""Calibration chronologique des probabilités de modèle.

Un gradient boosting non calibré produit des probabilités trop tranchées ou trop
timides ; comparer son Brier brut à une fréquence empirique serait déloyal.

`CalibratedClassifierCV` de scikit-learn impose un découpage « en partitions »
(`cross_val_predict`), incompatible avec un train chronologique croissant. On
implémente donc une calibration **explicitement temporelle** :

    bloc 1        bloc 2        bloc 3        bloc 4
    [----fit----][--calib--]
                 [----fit--------][--calib--]
                                [----fit--------][--calib--]
    => paires (probabilité, vérité) toutes postérieures à leur entraînement
    => régression isotone monotone sur ces paires
    => modèle final réentraîné sur TOUT le train, probabilités passées dans l'isotone

Aucune donnée du bloc de test n'est utilisée, à aucun moment.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression


class ChronologicalCalibrator(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=None, n_splits: int = 3):
        self.estimator = estimator
        self.n_splits = n_splits

    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        X = np.asarray(X, dtype=float) if not hasattr(X, "iloc") else X
        y = np.asarray(y, dtype=int)
        n = len(y)
        folds = int(self.n_splits) + 1
        if n < 500 or len(np.unique(y)) < 2:
            self.estimator_ = clone(self.estimator).fit(X, y)
            self.calibrator_ = None
            self.classes_ = np.array([0, 1])
            return self

        bounds = np.linspace(0, n, folds + 1).astype(int)
        calib_X, calib_p = [], []
        for k in range(1, folds):
            train_end = bounds[k]
            block = slice(bounds[k], bounds[k + 1])
            if train_end < 200 or (bounds[k + 1] - bounds[k]) < 20:
                continue
            base = clone(self.estimator)
            try:
                base.fit(X[:train_end], y[:train_end])
                p = base.predict_proba(X[block])[:, 1]
            except Exception:
                continue
            calib_X.append(y[block])
            calib_p.append(p)

        if calib_X:
            yy = np.concatenate(calib_X)
            pp = np.clip(np.concatenate(calib_p), 1e-6, 1 - 1e-6)
            if len(np.unique(yy)) > 1:
                self.calibrator_ = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
                self.calibrator_.fit(pp, yy)
            else:
                self.calibrator_ = None
        else:
            self.calibrator_ = None

        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float) if not hasattr(X, "iloc") else X
        p = np.asarray(self.estimator_.predict_proba(X))[:, 1]
        if getattr(self, "calibrator_", None) is not None:
            p = self.calibrator_.predict(np.clip(p, 1e-6, 1 - 1e-6))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def calibrate(estimator, n_splits: int = 3) -> ChronologicalCalibrator:
    return ChronologicalCalibrator(estimator=estimator, n_splits=n_splits)
