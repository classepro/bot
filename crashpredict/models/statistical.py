"""Modèles statistiques explicites (interprétables) et tests d'apport.

1. `ParetoTail` : ajuste la loi de Pareto à la queue des coefficients (x >= xmin)
   et en déduit F(t). Sert de baseline paramétrique et de test de qualité
   d'ajustement face à la loi empirique.
2. `MarkovChain` : chaîne de Markov d'ordre k sur des états de coefficient
   (catégories) — mesure s'il existe une dépendance sérielle exploitable.
3. `HawkesIntensity` : intensité auto-excitée — teste si les gros multiplicateurs
   augmentent la probabilité des suivants (cluster).

Chaque classe est volontairement simple, sans dépendance externe, et expose
`validation_probability(...)` pour une cible donnée.
"""
from __future__ import annotations

import numpy as np

from ..config import CONFIG

BINS = np.array([1.0, 1.01, 1.5, 2.0, 3.0, 5.0, 10.0, np.inf])


def discretize(m: np.ndarray) -> np.ndarray:
    return (np.digitize(np.asarray(m, dtype=float), BINS[1:-1])).astype(int)


def _choose_order(evidence: list[float], orders: list[int]) -> int:
    """Ordre de Markov le plus simple dans un intervalle de vraisemblance.

    On préfère l'ordre le plus bas ; on ne monte d'ordre que si le gain de
    log-vraisemblance par paramètre dépasse nettement le coût (critère BIC).
    """
    if not evidence:
        return 1
    best_ll = max(evidence)
    for order, ll in zip(orders, evidence):
        if best_ll - ll <= 2.0:  # 2 log-unités : gain non concluant
            return order
    return orders[-1]

class ParetoTail:
    """Ajustement de Pareto sur la queue, mélangé avec l'empirique (test d'ajustement).

    Permet de vérifier si la queue des coefficients suit une loi de puissance,
    et de comparer P(validation) paramétrique vs empirique.
    """

    name = "stat_pareto"

    def __init__(self, xmin: float = 2.0):
        self.xmin = xmin
        self.alpha: float | None = None
        self.p_base: float | None = None

    def fit(self, history: np.ndarray):
        h = np.asarray(history, dtype=float)
        h = h[np.isfinite(h)]
        tail = h[h >= self.xmin]
        if len(tail) >= 100:
            self.alpha = float(len(tail) / np.sum(np.log(tail / self.xmin)))
        else:
            self.alpha = None
        self.p_base = float(np.mean(h >= self.xmin)) if len(h) else None
        return self

    def survival(self, t: float) -> float:
        if self.alpha is None or self.p_base is None or t < self.xmin:
            return float("nan")
        return float(self.p_base * (t / self.xmin) ** (-self.alpha))

    def validation_probability(self, t: float, horizon: int | None = None) -> float:
        horizon = horizon or CONFIG.rules.horizon
        s = self.survival(t)
        if not np.isfinite(s):
            return float("nan")
        return float(1 - (1 - s) ** horizon)

    def describe(self) -> dict:
        return {"name": self.name, "xmin": self.xmin, "alpha": self.alpha, "p_base": self.p_base}


class MarkovChain:
    """Chaîne de Markov sur catégories de coefficients, ordre choisi par log-vraisemblance.

    Robustesse : transitions lissées vers la distribution marginale ; on préfère
    toujours l'ordre le plus bas parmi ceux dont la vraisemblance est équivalente
    (pénalité de complexité), afin de ne pas conclure à un « pattern » dû au bruit.
    """

    name = "stat_markov"
    ORDERS = [1, 2, 3, 5, 10]

    def __init__(self, max_order: int = 10, smoothing: float = 50.0, force_order: int | None = None):
        self.max_order = max_order
        self.smoothing = smoothing
        self.force_order = force_order
        self.order = force_order or 1
        self.transitions: dict[tuple, np.ndarray] = {}
        self.marginal: np.ndarray | None = None
        self.evidence: dict[int, float] = {}

    def _count(self, states: np.ndarray, order: int, n_states: int) -> dict[tuple, np.ndarray]:
        counts: dict[tuple, np.ndarray] = {}
        for k in range(order, len(states)):
            key = tuple(int(x) for x in states[k - order : k])
            counts.setdefault(key, np.zeros(n_states))
            counts[key][states[k]] += 1
        return counts

    @staticmethod
    def _loglik(counts: dict[tuple, np.ndarray], n_states: int) -> float:
        ll = 0.0
        for vec in counts.values():
            total = vec.sum()
            probs = (vec + 1e-3) / (total + 1e-3 * n_states)
            ll += float(np.sum(vec * np.log(np.maximum(probs, 1e-12))))
        return ll

    def fit(self, history: np.ndarray):
        states = discretize(history)
        n_states = len(BINS) - 1
        self.marginal = np.bincount(states, minlength=n_states).astype(float)
        self.marginal /= max(self.marginal.sum(), 1.0)

        if self.force_order:
            counts = self._count(states, self.force_order, n_states)
            self.order = self.force_order
            self.evidence = {self.force_order: self._loglik(counts, n_states)}
            self._store(counts)
            return self

        per_order = []
        for order in [o for o in self.ORDERS if o <= self.max_order]:
            if len(states) <= order + 50:
                continue
            counts = self._count(states, order, n_states)
            per_order.append((order, self._loglik(counts, n_states), counts))
        if not per_order:
            self._store({})
            return self
        best_ll = max(ll for _, ll, _ in per_order)
        # on retient l'ordre le plus SIMPLE dont la vraisemblance est à 1 % du meilleur
        compatible = [t for t in per_order if t[1] >= best_ll * 0.99] or [per_order[0]]
        self.order, _, counts = compatible[0]
        self.evidence = {o: ll for o, ll, _ in per_order}
        self._store(counts)
        return self

    def _store(self, counts: dict[tuple, np.ndarray]) -> None:
        self.transitions = {}
        for key, vec in counts.items():
            total = vec.sum()
            self.transitions[key] = (vec + self.smoothing * self.marginal) / (total + self.smoothing)

    def predict_distribution(self, recent: np.ndarray) -> np.ndarray:
        st = tuple(int(s) for s in discretize(recent)[-self.order :])
        return self.transitions.get(st, self.marginal)

    def validation_probability(self, recent: np.ndarray, target: float, horizon: int | None = None) -> float:
        horizon = horizon or CONFIG.rules.horizon
        dist = self.predict_distribution(recent)
        upper = np.array([1.01, 1.5, 2.0, 3.0, 5.0, 10.0, 1e6])
        p_round = float(np.clip(np.sum(dist * (upper >= target)), 0, 1))
        return float(1 - (1 - p_round) ** horizon)

    def describe(self) -> dict:
        return {"name": self.name, "order": self.order, "states": BINS[:-1].tolist(),
                "log_likelihood_par_ordre": {str(k): round(v, 1) for k, v in self.evidence.items()}}

class _LegacyMarkov:
    """(conservé pour référence) chaîne d'ordre fixe."""

    name = "stat_markov"

    def __init__(self, order: int = 1, smoothing: float = 5.0):
        self.order = order
        self.smoothing = smoothing
        self.trans: dict[tuple, np.ndarray] = {}
        self.state_survival: np.ndarray | None = None

    def fit(self, history: np.ndarray):
        states = discretize(history)
        n_states = len(BINS) - 1
        counts = np.zeros((n_states,) * (self.order + 1))
        idx = np.stack([states[k : len(states) - self.order + k] for k in range(self.order)] +
                       [states[self.order :]], axis=1)
        for row in idx:
            counts[tuple(row)] += 1
        # probabilités conditionnelles lissées
        self.counts = counts
        self.state_survival = np.array([
            np.mean(np.asarray(history)[states == s] >= 2.0) if np.any(states == s) else 0.0
            for s in range(n_states)
        ])
        return self

    def predict_distribution(self, recent: np.ndarray) -> np.ndarray:
        """Distribution du prochain état conditionnellement aux `order` derniers états."""
        n_states = len(BINS) - 1
        st = discretize(recent)[-self.order :]
        sub = self.counts[tuple(st)]
        total = sub.sum()
        if total == 0:
            marg = self.counts.sum(axis=tuple(range(self.order)))
            dist = marg / max(marg.sum(), 1)
        else:
            dist = (sub + self.smoothing) / (total + self.smoothing * n_states)
        return dist

    def validation_probability(self, recent: np.ndarray, target: float, horizon: int | None = None) -> float:
        horizon = horizon or CONFIG.rules.horizon
        dist = self.predict_distribution(recent)
        # probabilité qu'une manche >= cible, par état : approximation via bornes d'états
        upper = np.array([1.01, 1.5, 2.0, 3.0, 5.0, 10.0, 1e6])
        p_round = float(np.sum(dist * (upper >= target)))
        p_round = min(max(p_round, 0.0), 1.0)
        return float(1 - (1 - p_round) ** horizon)

    def describe(self) -> dict:
        return {"name": self.name, "order": self.order, "bins": BINS.tolist()}


class HawkesIntensity:
    """Approximation d'un processus auto-excité sur la fréquence des gros multiplicateurs.

    Si l'intensité estimée ne diffère pas significativement de la fréquence
    de base, on conclut à l'absence de clustering.
    """

    name = "stat_hawkes"

    def __init__(self, threshold: float = 3.0, kernel: str = "exp", decay: float = 20.0):
        self.threshold = threshold
        self.decay = decay
        self.base_rate = None
        self.excitation = None

    def fit(self, history: np.ndarray):
        h = np.asarray(history, dtype=float)
        hits = (h >= self.threshold).astype(float)
        self.base_rate = float(hits.mean())
        # régression approchée : intensité(t) = mu + alpha * exp(-dt/decay)
        lam = np.zeros(len(hits))
        acc = 0.0
        for i, v in enumerate(hits):
            lam[i] = acc
            acc = acc * np.exp(-1.0 / self.decay) + v
        if np.std(lam) > 0:
            corr = float(np.corrcoef(lam[1:], hits[1:])[0, 1]) if len(hits) > 2 else 0.0
        else:
            corr = 0.0
        self.excitation = corr
        return self

    def validation_probability(self, recent: np.ndarray, target: float, horizon: int | None = None) -> float:
        horizon = horizon or CONFIG.rules.horizon
        h = np.asarray(recent, dtype=float)
        hits = (h >= self.threshold).astype(float)
        acc = 0.0
        for v in hits:
            acc = acc * np.exp(-1.0 / self.decay) + v
        p_round = float(np.mean(h >= target))
        if self.excitation and np.isfinite(self.excitation):
            p_round = min(1.0, max(0.0, p_round * (1 + 0.05 * np.tanh(acc / 3.0))))
        return float(1 - (1 - p_round) ** horizon)

    def describe(self) -> dict:
        return {"name": self.name, "threshold": self.threshold, "decay": self.decay,
                "base_rate": self.base_rate, "excitation_corr": self.excitation}


class MarkovSweep:
    """Balayage systématique des ordres de Markov : détecte un signal sériel, ou son absence.

    Pour chaque ordre k, on prédit P(validation ≥ 2x) à partir des k dernières
    catégories, en mode walk-forward (la chaîne est réajustée par blocs).
    Le résultat est un tableau ordre -> Brier / AUC comparé à la loi marginale.
    """

    def __init__(self, orders: list[int] | None = None, target: float = 2.0):
        self.orders = orders or [1, 2, 3, 5, 10, 20]
        self.target = target

    def run(self, history: np.ndarray, block: int = 20_000, burn_in: int = 20_000) -> list[dict]:
        h = np.asarray(history, dtype=float)
        n = len(h)
        if n < burn_in + block:
            return []
        from ..backtesting.metrics import brier_score, roc_auc

        results = []
        for order in self.orders:
            ys, ps = [], []
            for start in range(burn_in, n - CONFIG.rules.horizon, block):
                end = min(start + block, n - CONFIG.rules.horizon)
                chain = MarkovChain(max_order=order).fit(h[:start])
                for i in range(start, end):
                    p = chain.validation_probability(h[max(0, i - order + 1) : i + 1], self.target)
                    fut = h[i + 1 : i + 1 + CONFIG.rules.horizon]
                    if len(fut) < CONFIG.rules.horizon:
                        continue
                    ps.append(p)
                    ys.append(1.0 if np.max(fut) >= self.target else 0.0)
            if ys:
                results.append(
                    {
                        "ordre": order,
                        "n": len(ys),
                        "brier": brier_score(ys, ps),
                        "auc": roc_auc(np.array(ys), np.array(ps)),
                        "taux_observe": float(np.mean(ys)),
                    }
                )
        # référence : probabilité constante (loi marginale estimée sur le passé)
        h_ref = h[:burn_in]
        p_ref = 1 - (1 - float(np.mean(h_ref >= self.target))) ** CONFIG.rules.horizon
        ys_ref = [1.0 if np.max(h[i + 1 : i + 1 + CONFIG.rules.horizon]) >= self.target else 0.0
                  for i in range(burn_in, n - CONFIG.rules.horizon)]
        if ys_ref:
            results.append(
                {
                    "ordre": 0,
                    "n": len(ys_ref),
                    "brier": brier_score(ys_ref, [p_ref] * len(ys_ref)),
                    "auc": 0.5,
                    "taux_observe": float(np.mean(ys_ref)),
                    "note": f"référence : probabilité constante {p_ref:.4f}",
                }
            )
        return results
