# Méthodologie statistique

Ce document décrit **exactement** ce que le système calcule, pourquoi, et comment il
évite de se mentir à lui-même.

## 1. Le problème, formulé correctement

On observe une suite de coefficients `m[1], m[2], …, m[n]` et une règle de validation :

```
validation(i, t) = 1  ⇔  MAX(m[i+1], m[i+2], m[i+3]) ≥ t
```

La question utile n'est pas « quel sera `m[i+1]` ? » (ce qui n'a de sens que si
la suite n'est pas i.i.d.), mais :

> **La probabilité de validation est-elle prévisible à partir du passé, au-delà de
> ce que prédit la loi marginale ?**

Formellement, on compare deux estimateurs de `P(validation(i,t) | F_i)` :

- **baseline** : `1 − (1 − F̂(t))³`, avec `F̂` estimée sur une fenêtre passée ;
- **modèle** : `ĝ(features causales de i)`, calibré.

Sous l'hypothèse d'indépendance, la baseline est optimale et aucun modèle ne peut
la battre durablement. Le système est donc construit pour **détecter** un écart
s'il existe, et pour **l'annoncer s'il n'existe pas**.

## 2. Règle de validation : une seule implémentation

`crashpredict/validation/three_rounds.py` définit `apply_rule(target, values)` :

| Situation | Statut |
|---|---|
| 3 manches disponibles, `MAX ≥ cible` | `SUCCESS` |
| 3 manches disponibles, `MAX < cible` | `FAILURE` |
| moins de 3 manches | `PENDING` |

Le backtest, la validation en direct et l'interface appellent cette même fonction.
Les tests `test_validation_rule.py` couvrent les exemples du cahier des charges,
le cas d'égalité exacte et le fait qu'une 4ᵉ manche ne peut pas sauver une prédiction.

## 3. Anti-fuite temporelle

### 3.1 Features causales

Pour chaque ligne `i`, toutes les variables sont calculées uniquement sur `m[≤ i]` :
moyennes/médianes/écarts-types sur fenêtres **passées** (`shift(1).rolling(...)`),
fréquences empiriques passées, longueur de séries, variations, etc. Aucun
`center=True`, aucun `.shift(-k)` dans les features.

Test automatique (`tests/test_no_leakage.py::test_features_are_causal`) :
on remplace brutalement le futur (`m[300:] = 50`) et on vérifie que les features
des indices `< 300` sont **identiques au bit près**.

### 3.2 Purge

La cible de la ligne `i` dépend des manches `i+1, i+2, i+3`. Pour un test démarrant
à la manche `s`, on n'entraîne que sur les lignes `i ≤ s − 4` :

```python
train_end = training_slice_end(start=s, purge=3)   # = s - 3
```

ainsi la cible de la dernière ligne d'entraînement est entièrement connue avant `s`.

### 3.3 Pas de séparation aléatoire

Aucun `train_test_split(shuffle=True)` dans le code : uniquement des blocs
chronologiques. Le walk-forward réentraîne chaque modèle tous les `step` manches,
sur une fenêtre glissante bornée (`max_train_window`), avec les mêmes hyperparamètres
(sélectionnés avant, jamais sur le test).

## 4. Métriques

| Métrique | Formule | Lecture |
|---|---|---|
| Brier | `mean((p − y)²)` | erreur quadratique, plus bas = mieux, unité = probabilité² |
| Log-loss | `−mean(y log p + (1−y) log(1−p))` | pénalise les probabilités confiantes et fausses |
| AUC | rang des positifs | capacité de discrimination (0.5 = hasard) |
| Calibration | fréquence observée par tranche de probabilité | les probabilités annoncées sont-elles honnêtes ? |
| Skill | `1 − Brier_modèle / Brier_baseline` | gain relatif |

Les taux de réussite sont toujours accompagnés du **dénominateur** et de la
**probabilité annoncée** au même moment : un taux de 87 % sans le 87 % attendu ne
veut rien dire.

## 5. Tests de significativité

1. **Différence appariée de Brier.** Pour chaque manche testée, `d_i = (p_base − y)² − (p_model − y)²`.
   `Δ = mean(d_i) > 0` ⇒ le modèle est meilleur.
2. **IC 95 % bootstrap** (2 000 rééchantillonnages) sur `Δ`. Si l'IC contient 0 :
   « aucun avantage démontré » (formulation utilisée dans l'interface).
3. **t-test apparié** sur `d` pour la p-value brute.
4. **Correction de Holm** sur toute la famille (modèles × cibles) : comparer 5 modèles
   sur 4 cibles, c'est 20 tests ; sans correction, un faux positif est quasi certain.
5. **Taux de réussite** : test binomial contre la probabilité de baseline, et
   test de dérive (500 dernières manches contre 5 000).

## 6. Décision de déploiement

`choose_deployed_model` retient un modèle **seulement si** `Δ Brier > 0` **et**
`p_holm < 0.05` sur au moins une configuration. Sinon, le robot déploie la
**baseline empirique**, et l'interface l'écrit noir sur blanc :
« aucun avantage statistiquement significatif ».

C'est une décision de conception, pas un aveu d'échec : un système qui déploie un
modèle perdant parce qu'il « fait sérieux » est un système malhonnête.

## 7. Stratégies de cible

Cibles fixes : 1.50x / 2.00x / 2.50x / 5.00x.

Cible optimisée pour un taux visé `p` (par défaut 80 %) :

```
F(t) = 1 − (1 − p)^(1/3)     puis t = le plus grand palier 0.05x avec P(validation) ≥ p
```

`F` est estimée sur tout l'historique disponible (aucune extrapolation), et
`n` (le nombre de manches utilisées) est toujours affiché.

## 8. Limites assumées

- **Pas de variable exogène** (aucune information autre que l'historique des coefficients).
  Si le serveur mélange un état caché, il n'est pas observable ici.
- **Horodatage estimé** pour l'historique importé ; les analyses temporelles
  fines (par heure de la journée) ne sont fiables qu'après collecte live.
- **Échantillon d'une seule plateforme** (BC.Game). Les conclusions ne sont pas
  transférables automatiquement à une autre implémentation.
- **Fenêtres chevauchantes** : en mode « 1 prédiction par manche », les fenêtres de
  validation se recouvrent ; les taux restent non biaisés mais les échantillons ne
  sont pas indépendants. Les tests de significativité utilisent des différences
  appariées (robustes à cette dépendance), et le rapport présente aussi la version
  à fenêtres **disjointes**.
- **Marge maison** : si le RNG est sain, aucune stratégie n'a d'espérance positive.
  Le système mesure ; il ne promet pas.
