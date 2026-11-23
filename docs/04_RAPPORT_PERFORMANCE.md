# Rapport de performance

> Ce document est un **instantané rédigé** des résultats mesurés. La version
> toujours à jour, générée depuis la base et accompagnée des tableaux complets,
> se trouve dans [`reports/rapport_performance.md`](../reports/rapport_performance.md),
> [`reports/rapport_performance.html`](../reports/rapport_performance.html) et
> [`reports/rapport_performance.json`](../reports/rapport_performance.json)
> (régénérée par `python scripts/report.py` ou après chaque entraînement).

## 1. Données utilisées

| Élément | Valeur |
|---|---|
| Plateforme | BC.Game — jeu *Crash* (provably fair) |
| Manches | **104 893** consécutives (`round_id` 5 822 149 → 5 927 041) |
| Trous | **0** |
| Doublons | **0** |
| Transitions de chaîne SHA256 vérifiées | **104 892** |
| Échecs de vérification | **0** |
| Médiane / moyenne / maximum | 1.98x / 14.43x / 127 151x |
| Part de manches à 1.00x (crash instantané) | 2.03 % |
| Horodatage | absent de la source historique → estimé ; réel pour la collecte live |

La chaîne `SHA256(hash[i+1]) == hash[i]` prouve que la série est complète, dans
l'ordre et non retouchée. **Aucune donnée n'a été fabriquée.**

## 2. Résultat central : la loi marginale explique tout

Fenêtres de 3 manches, règle `MAX(3) ≥ cible`, test binomial contre
`1 − (1 − P(1 manche ≥ cible))³` :

| Cible | P(1 manche) | P(validation) théorique | Observé | Écart | p-value |
|---|---|---|---|---|---|
| 1.20x | 82.33 % | 99.45 % | 99.44 % | −0.01 pt | 0.770 |
| 1.50x | 65.96 % | 96.06 % | 95.97 % | −0.08 pt | 0.170 |
| 1.80x | 55.03 % | 90.91 % | 90.99 % | +0.08 pt | 0.367 |
| 2.00x | 49.55 % | 87.16 % | 87.34 % | +0.18 pt | 0.080 |
| 2.50x | 39.68 % | 78.05 % | 78.18 % | +0.12 pt | 0.332 |
| 3.00x | 33.07 % | 70.02 % | 70.10 % | +0.08 pt | 0.567 |
| 5.00x | 19.79 % | 48.39 % | 48.38 % | −0.01 pt | 0.931 |
| 10.00x | 9.93 % | 26.93 % | 26.90 % | −0.03 pt | 0.805 |

**Aucun écart significatif** (aucune p-value < 0.05, et aucune ne survit à une
correction de Holm sur 10 cibles). Les manches sont compatibles avec des tirages
indépendants.

## 3. Absence de structure sérielle

| Test | Résultat | Interprétation |
|---|---|---|
| Autocorrélation « ≥ 1.5x » (lag 1) | +0.003 | nul |
| Autocorrélation « ≥ 2x » (lag 1) | −0.004 | nul |
| Autocorrélation « ≥ 5x » (lag 1) | ≈ 0.000 | nul |
| Runs test (médiane) | z ≈ 0, p ≈ 1 | aucune tendance |
| Taux par décile de temps (≥ 2x) | 49.8 % → 48.6 % | stable |
| Chaînes de Markov ordres 1 à 5 (walk-forward) | Brier 0.1188 contre **0.1110** pour la probabilité constante | **les transitions Markov dégradent** la précision |
| Ajustement de Pareto sur la queue | α ≈ 1.01, P(5x) estimé 0.1974 vs empirique 0.1976 | queue conforme à une loi de puissance |
| Intensité auto-excitée (Hawkes) | corrélation ≈ −0.005 | aucun clustering |

**Conclusion** : il n'existe pas de mémoire exploitable dans la suite. Toute
prédiction « par pattern » (séries, écarts, moyennes mobiles, Markov) ajoute du
bruit, pas de l'information.

## 4. Backtest walk-forward des modèles

Protocole : entraînement initial 25 000 manches, réentraînement tous les 3 000,
purge de 3 manches, fenêtre d'entraînement ≤ 60 000, test chronologique strict,
**12 folds** couvrant les manches 5 847 149 → 5 883 148
(36 000 prédictions par cible, 144 000 au total).

Modèles comparés : régression logistique, RandomForest, LightGBM, XGBoost, CatBoost,
plus les baselines empiriques (probabilité fixe, fenêtre glissante 5 000 et 500).

### 4.1 Brier moyen hors échantillon

| Modèle | Brier | Δ Brier vs baseline |
|---|---|---|
| baseline_fixed | 0.143315 | +0.000017 |
| **baseline_rolling_long** (déployée) | **0.143331** | — |
| random_forest | 0.143461 | −0.000130 |
| logistic | 0.143700 | −0.000369 |
| catboost | 0.143784 | −0.000452 |
| xgboost | 0.144168 | −0.000837 |
| lightgbm | 0.144450 | −0.001119 |
| baseline_rolling_short | 0.143900 | −0.000569 |

### 4.2 Significativité (28 tests, correction de Holm)

| Résultat | Nombre de configurations (modèle × cible) |
|---|---|
| Modèle **meilleur** que la baseline, significativement | **0** |
| Modèle **pire** que la baseline, significativement | **17** |
| Différence non significative | 11 |

Aucun modèle ne bat la baseline. LightGBM et XGBoost, les plus flexibles, sont
significativement **pires** : ils ajustent du bruit et « trouvent » des motifs qui
n'existent pas. Les modèles qui s'en sortent le mieux sont ceux qui sont les plus
proches d'une probabilité constante.

### 4.3 Calibration (cible 2.00x, 36 000 prédictions)

| Tranche de probabilité | Annoncée | Observée | Écart |
|---|---|---|---|
| 0.8 – 0.9 | 87.04 % | 87.07 % | +0.03 pt |

| Cible | Taux observé | Probabilité annoncée | Écart | Brier |
|---|---|---|---|---|
| 1.50x | 95.92 % | 96.04 % | −0.12 pt | 0.039 |
| 2.00x | 87.07 % | 87.04 % | +0.03 pt | 0.113 |
| 2.50x | 77.96 % | 77.92 % | +0.04 pt | 0.172 |
| 5.00x | 48.15 % | 48.24 % | −0.10 pt | 0.250 |

Les probabilités annoncées sont donc **honnêtes** : elles correspondent aux
fréquences réellement observées hors échantillon, à 0.1 point près.

**Décision automatique du système** : `deployed_model = baseline_rolling_long`.
Le robot déploie la loi empirique, pas un modèle ML qui fait moins bien.

## 5. Stratégies et espérance

| Cible | Taux de validation (3 manches) | Gain espéré par manche |
|---|---|---|
| 1.50x | 95.97 % | 0.9894 |
| 2.00x | 87.34 % | 0.9910 |
| 2.50x | 78.05 % | 0.9920 |
| 5.00x | 48.38 % | 0.9894 |

Le gain espéré par manche est **inférieur à 1.00** pour toutes les cibles : la
marge maison (≈ 1 %) l'emporte. Le taux de validation d'une cible n'est pas un
rendement ; c'est une fréquence.

## 6. Reproductibilité

```bash
python scripts/import_data.py      # 104 893 manches, chaîne vérifiée
python scripts/train.py            # walk-forward complet (4 cibles x 4 modèles)
python scripts/report.py           # régénère reports/rapport_performance.*
pytest -q                          # 26 tests (règle, anti-fuite, intégrité, stats)
```

Toute conclusion de ce document est recalculable à partir de la base et des
scripts fournis ; aucun chiffre n'est saisi à la main dans le code.

## 7. Verdict

**Aucun avantage statistiquement significatif.** Le système :

1. estime correctement des probabilités de validation (calibrées, vérifiées hors
   échantillon) ;
2. démontre que ces probabilités sont entièrement expliquées par la loi marginale ;
3. refuse de déployer un modèle qui ne fait pas mieux, et l'écrit dans l'interface
   comme dans le rapport.

Le robot est donc un **instrument de mesure honnête**, pas une machine à gagner.
C'est exactement ce que demandait la priorité du cahier des charges : un système
capable de tester scientifiquement ses propres prédictions et de ne jamais masquer
ses échecs.
