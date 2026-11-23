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

Protocole : entraînement initial 60 000 manches, réentraînement tous les 6 000,
purge de 3 manches, fenêtre d'entraînement glissante ≤ 40 000 manches, test
chronologique strict, **7 folds** couvrant les manches **5 882 149 → 5 922 145**
(≈ 40 000 prédictions par cible, **159 988 au total**).

Chaque modèle est **calibré par régression isotone chronologique** à l'intérieur
de son fold d'entraînement : sans cela, un gradient boosting produit des
probabilités non comparables à une fréquence empirique et le test de Brier serait
déloyal.

Modèles comparés : régression logistique, RandomForest, LightGBM, XGBoost,
CatBoost, plus les baselines empiriques (probabilité fixe, fenêtre glissante 5 000
et 500 manches).

### 4.1 Brier moyen hors échantillon

| Modèle | Brier | Δ Brier vs baseline |
|---|---|---|
| baseline_fixed | 0.141836 | +0.000026 |
| **baseline_rolling_long** (déployée) | **0.141862** | — |
| logistic | 0.141867 | −0.000005 |
| lightgbm | 0.141869 | −0.000007 |
| random_forest | 0.141876 | −0.000014 |
| xgboost | 0.141901 | −0.000039 |
| catboost | 0.141993 | −0.000131 |
| baseline_rolling_short | 0.142306 | −0.000444 |

### 4.2 Significativité (28 tests, correction de Holm)

| Résultat | Configurations (modèle × cible) |
|---|---|
| Modèle **meilleur** que la baseline, significativement | **0** |
| Modèle **pire** que la baseline, significativement | 7 |
| Différence non significative | 21 |

Aucun modèle ne bat la baseline. Les seules dégradations significatives sont
marginales (Δ Brier de l'ordre de 10⁻⁴) et proviennent surtout de la baseline à
fenêtre courte (500 manches), qui réagit trop au bruit.

> **Nuance méthodologique honnête** : lors d'un run antérieur *sans* calibration
> isotone, LightGBM et XGBoost apparaissaient **significativement pires** (Δ Brier
> jusqu'à −0.0022). L'écart mesuré était donc en partie un artefact de calibration,
> pas seulement du sur-apprentissage. Après calibration, les GBM redeviennent
> comparables à la baseline sans la battre. Ce détail est conservé ici parce qu'il
> illustre la différence entre « résultat » et « artefact de protocole ».

### 4.3 Calibration et exactitude (modèle déployé)

| Cible | Prédictions | Taux observé | Probabilité annoncée | Écart | Brier |
|---|---|---|---|---|---|
| 1.50x | 39 997 | 96.01 % | 96.07 % | −0.06 pt | 0.038 |
| 2.00x | 39 997 | 87.45 % | 87.20 % | +0.25 pt | 0.110 |
| 2.50x | 39 997 | 78.39 % | 78.22 % | +0.17 pt | 0.169 |
| 5.00x | 39 997 | 48.82 % | 48.77 % | +0.05 pt | 0.250 |

Sur 160 000 prédictions hors échantillon, l'écart entre probabilité annoncée et
fréquence observée est **inférieur à 0.25 point de pourcentage**. Les probabilités
du robot sont donc des fréquences vérifiées, pas des promesses.

**Décision automatique du système** : `deployed_model = baseline_rolling_long`.
Le robot déploie la loi empirique, pas un modèle ML qui ne fait pas mieux.

## 5. Stratégies et espérance

| Cible | Taux de validation (3 manches) | Gain espéré par manche |
|---|---|---|
| 1.50x | 96.01 % | 0.9894 |
| 2.00x | 87.45 % | 0.9910 |
| 2.50x | 78.39 % | 0.9920 |
| 5.00x | 48.82 % | 0.9894 |

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
