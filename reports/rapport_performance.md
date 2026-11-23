# Rapport de performance — Crash Game AI Predictor

Généré le **2026-11-23T12:27:47+00:00** · run `R1555C028505F4A80`

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

## 6. ROBOT V4.5 (génération précédente) — reproduction expérimentale

Protocole : fenêtre de saisie **10 cotes passées**, ordre *recent*, première prédiction possible à la manche `5822159`, cibles produites uniquement à partir du passé (cible = agrégat de m[a-window:a] uniquement ; validation = m[a:a+horizon]).

### Validation sur les 3 manches suivantes

| Stratégie | Prédictions | Succès | Échecs | Taux | IC 95 % | Loi marginale | Écart | p (Holm) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| Standard (moyenne simple) | 104 881 | 74 602 | 30 279 | **71.13 %** | [70.86 ; 71.40] | 71.06 % | +0.07 pts | 1.000 | aucun avantage démontré (IC 95 % contient 0) |
| Avancé (pondération 1 + 0,1·i) | 104 881 | 71 040 | 33 841 | **67.73 %** | [67.45 ; 68.02] | 67.56 % | +0.17 pts | 1.000 | aucun avantage démontré (IC 95 % contient 0) |
| Professionnel (sans tendance) = Standard | 104 881 | 57 617 | 47 264 | **54.94 %** | [54.63 ; 55.24] | 54.79 % | +0.14 pts | 1.000 | aucun avantage démontré (IC 95 % contient 0) |
| Professionnel + analyse tendancielle | 104 881 | 40 617 | 64 264 | **38.73 %** | [38.43 ; 39.02] | 38.63 % | +0.08 pts | 1.000 | aucun avantage démontré (IC 95 % contient 0) |
| Réglage conseillé par le guide | 104 881 | 74 602 | 30 279 | **71.13 %** | [70.86 ; 71.40] | 71.06 % | +0.07 pts | 1.000 | aucun avantage démontré (IC 95 % contient 0) |
| Guide + report au 3e round | 104 881 | 74 602 | 30 279 | **71.13 %** | [70.86 ; 71.40] | 71.06 % | +0.07 pts | 1.000 | aucun avantage démontré (IC 95 % contient 0) |
| Guide + martingale | 104 881 | 74 602 | 30 279 | **71.13 %** | [70.86 ; 71.40] | 71.06 % | +0.07 pts | 1.000 | aucun avantage démontré (IC 95 % contient 0) |

### Validation sur les 5 manches suivantes

| Stratégie | Prédictions | Succès | Échecs | Taux | IC 95 % | Loi marginale | Écart | p (Holm) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| Standard (moyenne simple) | 104 879 | 89 119 | 15 760 | **84.97 %** | [84.76 ; 85.19] | 84.81 % | +0.16 pts | 1.000 | aucun avantage démontré (IC 95 % contient 0) |
| Avancé (pondération 1 + 0,1·i) | 104 879 | 86 322 | 18 557 | **82.31 %** | [82.07 ; 82.54] | 81.99 % | +0.31 pts | 0.047 | meilleur que la loi marginale (significatif) |
| Professionnel (sans tendance) = Standard | 104 879 | 72 931 | 31 948 | **69.54 %** | [69.26 ; 69.82] | 69.32 % | +0.21 pts | 1.000 | aucun avantage démontré (IC 95 % contient 0) |
| Professionnel + analyse tendancielle | 104 879 | 55 174 | 49 705 | **52.61 %** | [52.31 ; 52.91] | 52.55 % | +0.04 pts | 0.747 | aucun avantage démontré (IC 95 % contient 0) |
| Réglage conseillé par le guide | 104 879 | 89 119 | 15 760 | **84.97 %** | [84.76 ; 85.19] | 84.81 % | +0.16 pts | 1.000 | aucun avantage démontré (IC 95 % contient 0) |
| Guide + report au 3e round | 104 879 | 89 119 | 15 760 | **84.97 %** | [84.76 ; 85.19] | 84.81 % | +0.16 pts | 1.000 | aucun avantage démontré (IC 95 % contient 0) |
| Guide + martingale | 104 879 | 89 119 | 15 760 | **84.97 %** | [84.76 ; 85.19] | 84.81 % | +0.16 pts | 0.928 | aucun avantage démontré (IC 95 % contient 0) |

### Ce que la reproduction démontre

- Aucune stratégie n'atteint les 85–95 % annoncés **sous la règle des 3 manches** : la meilleure plafonne à 71.13 %.
- Sous l'horizon 5 de la promesse commerciale, la meilleure atteint 84.97 % — mais la loi marginale annonçait déjà ce niveau, puisqu'une cible de 3.38x est atteinte dans 5 manches avec une probabilité de 84.81 %.
- Écart significatif après correction de Holm : advanced. En ne gardant qu'une manche sur `horizon` (fenêtres non chevauchantes, IC exacts), il reste : aucun — l'avantage ne survit donc pas au contrôle du chevauchement.
- Confrontation aux modèles ML (dernier walk-forward   39997 manches) : 0 configuration(s) meilleure(s) que la loi marginale, 8 pire(s) de façon significative, 24 sans différence — les deux générations de robots se heurtent au même mur.

### L'illusion des 85–95 %

Le robot ne se valide pas lui-même : l'utilisateur clique « ✓ Validé » ou « ✕ Échoué », et une prédiction non clôturée reste `pending`, donc **hors du dénominateur**. Voici le taux affiché selon la part d'échecs réellement marqués comme tels :

| Part des échecs honnêtement marqués | Taux affiché (horizon 5) | Taux affiché (horizon 3) |
|---|---|---|
| 100 % | 84.97 % | 71.13 % |
| 70 % | 88.98 % | 77.87 % |
| 50 % | 91.88 % | 83.13 % |
| 30 % | 94.96 % | 89.15 % |
| 20 % | 96.58 % | 92.49 % |
| 10 % | 98.26 % | 96.10 % |
| 5 % | 99.12 % | 98.01 % |

Autrement dit : **85–95 % est exactement ce qu'on obtient en ne clôturant que les succès**, sans falsifier quoi que ce soit. S'y ajoute le pourcentage affiché par le robot (`1/o₀`, la première côte saisie), qui n'est pas la probabilité de la cible.

| Horizon | Taux observé | % moyen affiché par le robot | Brier du % affiché | Brier de la loi marginale |
|---|---|---|---|---|
| 3 manches | 71.13 % | 83.19 % | 0.21963 | 0.17485 |
| 5 manches | 84.97 % | 83.19 % | 0.13112 | 0.10632 |

### La politique de mise recommandée par le guide

| Stratégie | Politique | Essais | Atteints | Taux | Profit (bankroll initiale = 1) | Espérance théorique | Décroissance max | Ruines | Témoin sans information |
|---|---|---|---|---|---|---|---|---|---|
| standard | mise à chaque manche | 20 976 | 17 790 | 84.81 % | **-4.562** | -9.320 | 4.533 | 13 | -4.031 |
| standard | report au 3e round | 20 976 | 14 863 | 70.86 % | **-13.002** | -7.324 | 4.066 | 14 | -4.471 |
| standard | martingale (5 essais) | 20 976 | 17 790 | 84.81 % | **-56.452** | -34.938 | 17.201 | 126 | -112.108 |
| advanced | mise à chaque manche | 20 976 | 17 228 | 82.13 % | **-0.218** | -9.897 | 4.484 | 10 | -0.891 |
| advanced | report au 3e round | 20 976 | 14 116 | 67.30 % | **-13.635** | -7.629 | 4.722 | 14 | +2.881 |
| advanced | martingale (5 essais) | 20 976 | 17 228 | 82.13 % | **-57.849** | -41.350 | 28.016 | 120 | -7.243 |
| pro | mise à chaque manche | 20 976 | 14 534 | 69.29 % | **-10.544** | -13.586 | 10.587 | 17 | -4.063 |
| pro | report au 3e round | 20 976 | 11 424 | 54.46 % | **-26.701** | -9.583 | 8.141 | 27 | +2.715 |
| pro | martingale (5 essais) | 20 976 | 14 534 | 69.29 % | **-340.027** | -67.986 | 73.035 | 375 | +126.928 |
| pro_guide | mise à chaque manche | 20 976 | 17 790 | 84.81 % | **-4.562** | -9.320 | 4.533 | 13 | -4.031 |
| pro_guide | report au 3e round | 20 976 | 14 863 | 70.86 % | **-13.002** | -7.324 | 4.066 | 14 | -4.471 |
| pro_guide | martingale (5 essais) | 20 976 | 17 790 | 84.81 % | **-56.452** | -34.938 | 17.201 | 126 | -112.108 |

Le profit est exprimé en multiples de la bankroll initiale ; une ruine signifie être passé sous 1 % de cette bankroll. Le **témoin** rejoue les mêmes cibles, mêmes mises et mêmes dates en tirant l'atteinte avec la probabilité de la loi marginale : tout écart entre les deux colonnes mesure la chance, pas la compétence.

### Conclusion sur ROBOT V4.5

Les seuls avantages significatifs disparaissent quand les fenêtres de validation ne se chevauchent plus : signature du hasard, pas d'un edge.


## 7. Conclusion

**AUCUN AVANTAGE STATISTIQUEMENT SIGNIFICATIF** — Aucun modèle ne bat la baseline empirique de façon significative ; 7 configuration(s) font significativement pire. Le robot déploie donc la loi marginale (probabilités calibrées, sans prétention prédictive).
