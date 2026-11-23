# Rapport de performance — Crash Game AI Predictor

Généré le **2026-11-23T11:04:41+00:00** · run `R1555C028505F4A80`

## 1. Données

- 104 893 manches (5822149 → 5927041)  0 trou  0 doublon
- Chaîne provably-fair : 104 892 transitions vérifiées  **0 échec**
- Médiane 1.98x · moyenne 14.43x · maximum 127151.29x · part de 1.00x 2.03 %

## 2. La loi marginale explique-t-elle les validations ?

| Cible | P(1 manche ≥ cible) | P(validation) théorique | Observé (fenêtres de 3) | Écart | p-value binomiale |
|---|---|---|---|---|---|
| 1.2x | 82.33 % | 99.45 % | 99.44 % | -0.01 pts | 0.770 |
| 1.5x | 65.96 % | 96.06 % | 95.97 % | -0.08 pts | 0.170 |
| 1.8x | 55.03 % | 90.91 % | 90.99 % | +0.08 pts | 0.367 |
| 2x | 49.55 % | 87.16 % | 87.34 % | +0.18 pts | 0.080 |
| 2.5x | 39.68 % | 78.05 % | 78.18 % | +0.12 pts | 0.332 |
| 3x | 33.07 % | 70.02 % | 70.10 % | +0.08 pts | 0.567 |
| 4x | 24.80 % | 57.47 % | 57.47 % | -0.00 pts | 0.983 |
| 5x | 19.79 % | 48.39 % | 48.38 % | -0.01 pts | 0.931 |
| 7.5x | 13.12 % | 34.42 % | 34.40 % | -0.02 pts | 0.897 |
| 10x | 9.93 % | 26.93 % | 26.90 % | -0.03 pts | 0.805 |

Aucun écart significatif ⇒ les manches sont compatibles avec des tirages indépendants.

## 3. Dépendance sérielle

- Autocorrélation de l'indicateur « ≥ 1.5x » (lag 1) : +0.0033
- Autocorrélation de l'indicateur « ≥ 2x » (lag 1) : -0.0042
- Autocorrélation de l'indicateur « ≥ 5x » (lag 1) : +0.0019
- Test des séries (runs test) : 52690 séries observées contre 52446.6 attendues, z = +1.50, p = 0.133

## 4. Recherche de structure sérielle (Markov)

Ordres de Markov testés en walk-forward pour la cible 2.00x :

| Ordre | n | Brier | AUC | Taux observé |
|---|---|---|---|---|
| Markov 1 | 84 890 | 0.118767 | 0.5034 | 87.28 % |
| Markov 2 | 84 890 | 0.118767 | 0.5034 | 87.28 % |
| Markov 3 | 84 890 | 0.118767 | 0.5034 | 87.28 % |
| Markov 5 | 84 890 | 0.118767 | 0.5034 | 87.28 % |
| référence (constante) | 84 890 | 0.111029 | 0.5000 | 87.28 % |

**Aucun signal sériel** : aucun ordre de Markov ne bat la probabilité constante ; lisser davantage les transitions les rend même moins précis. Les coefficients se comportent comme des tirages indépendants.

## 5. Backtest walk-forward

| Cible | Prédictions | Taux observé | Probabilité annoncée | Brier |
|---|---|---|---|---|
| 1.5x | 39 997 | 96.01 % | 96.07 % | 0.038291 |
| 2x | 39 997 | 87.45 % | 87.20 % | 0.109759 |
| 2.5x | 39 997 | 78.39 % | 78.22 % | 0.169451 |
| 5x | 39 997 | 48.82 % | 48.77 % | 0.249946 |

## 6. Conclusion

**AUCUN AVANTAGE STATISTIQUEMENT SIGNIFICATIF** — Aucun modèle ne bat la baseline empirique de façon significative ; 7 configuration(s) font significativement pire. Le robot déploie donc la loi marginale (probabilités calibrées, sans prétention prédictive).
