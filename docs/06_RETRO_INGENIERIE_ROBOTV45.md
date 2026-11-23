# 06 — Rétro-ingénierie de ROBOT V4.5 et reproduction expérimentale

> Document de méthode. Tous les chiffres cités sont produits par
> `scripts/legacy_backtest.py` et stockés dans `reports/robotv45.json`.
> Aucun n'est saisi à la main ; aucun n'est repris de la communication du robot.

---

## 1. Ce qu'est réellement ROBOT V4.5

`legacy/ROBOTV4.5.html` est une **page HTML unique de 1 565 lignes**. Elle contient
trois choses, qu'il faut absolument distinguer :

| Bloc | Lignes | Nature |
|---|---|---|
| page commerciale, codes d'accès, Firebase, compte à rebours | 1–718, 1258–1465 | logistique de vente (1 500 / 10 500 / 35 000 F CFA) |
| **algorithme de prédiction** | **742–826** | ≈ 60 lignes de JavaScript |
| interface, historique, statistiques, graphiques | 827–1257 | affichage et saisie manuelle des résultats |

L'algorithme complet tient en une fonction `predictOdds()` et une fonction
`calculateTrend()`. Il n'y a **ni modèle, ni apprentissage, ni base de données, ni
mémoire** : le robot ne conserve rien d'une session à l'autre et ne lit jamais
l'historique d'un site. Son unique entrée est la série de cotes que l'utilisateur
colle dans un champ de texte.

Conséquence directe : le robot ne peut pas « analyser l'algorithme du jeu » comme
l'annonce son guide (§ « Analyse Intelligence »). Le panneau `analyzeHistoricalData()`
(L1100-1253) ne calcule que des fréquences descriptives sur les prédictions déjà
saisies, à partir de `localStorage`.

---

## 2. Reconstruction exacte de l'algorithme

### 2.1 Entrée (L742-752)

```js
let oddsArray = oddsInput.split(',')
    .map(odd => parseFloat(odd.trim()))
    .filter(odd => !isNaN(odd) && odd > 1);
if (oddsArray.length < 5) { /* refus : « au moins 5 côtes valides » */ }
```

* séparateur `,` ; espaces ignorés ;
* les valeurs non numériques et **toutes les cotes ≤ 1 sont supprimées**
  (le guide conseille d'écrire `1.01` pour un crash à `1.00`) ;
* **minimum 5 cotes**, sinon refus.

### 2.2 Option « exclure les valeurs extrêmes » (L755-759)

```js
oddsArray.sort((a, b) => a - b);
oddsArray = oddsArray.slice(1, -1);
```

Tri croissant, puis suppression du minimum **et** du maximum. C'est le seul
endroit du code qui trie : sans cette option, **l'ordre de saisie est conservé**.

### 2.3 Les trois modes (L761-775)

| Mode | Code | Formule |
|---|---|---|
| `standard` (défaut) | `default:` | `T = moyenne(o)` |
| `advanced` | `case 'advanced':` | `wᵢ = 1 + 0,1·i`<br>`T = Σ(oᵢ·wᵢ) / Σ(wᵢ)` |
| `pro` | `case 'pro':` | `tendance = trendAnalysis ? calculateTrend(o) : 1`<br>`T = moyenne(o) × tendance` |

et (L821-826) :

```js
function calculateTrend(oddsArray) {
    const firstHalf = oddsArray.slice(0, Math.floor(oddsArray.length / 2));
    const secondHalf = oddsArray.slice(Math.floor(oddsArray.length / 2));
    return 1 + ((mean(secondHalf) - mean(firstHalf)) / mean(firstHalf));
}
```

**Trois constats décisifs, tous vérifiés dans le code :**

1. **Le mode « Professionnel », le plus recommandé par le guide, est
   arithmétiquement identique au mode Standard.** Le défaut de
   `localStorage['predictionSettings']` est `trendAnalysis: false`, et le guide
   recommande explicitement de **ne pas** cocher l'analyse tendancielle. Dans ce
   cas `tendance = 1` et `T = moyenne(o)`. Trois modes, deux formules.
2. **Le mode « Avancé » dépend de l'ordre de saisie**, notion qui n'a aucune
   contrepartie temporelle dans le code : rien n'indique si la première côte
   tapée est la plus récente ou la plus ancienne. C'est un paramètre libre, donc
   un degré de liberté pour l'utilisateur — et pour quiconque veut embellir un
   résultat a posteriori. Nous le testons en sensibilité (§ 4.3).
3. **Le mode « Professionnel » avec tendance est procyclique** : il augmente la
   cible quand les dernières cotes montent (`tendance > 1`), c'est-à-dire quand
   la série vient d'être généreuse. Il parie donc sur la continuation d'une
   tendance dans un jeu dont les manches sont indépendantes (§ 3 du rapport).

### 2.4 Ce que le robot affiche (L777-787, L810)

```js
const probabilities = oddsArray.map(odd => (1 / odd * 100).toFixed(2));
averageOdds: averageOdds.toFixed(2)         // « Côte Probable »
<p>Probabilité: ${predictionResult.probabilities[0]}%</p>
```

* la **cible** est `averageOdds`, arrondie à 2 décimales par `toFixed(2)` ;
* le **pourcentage affiché** est `probabilities[0]` = `1/o₀`, où `o₀` est la
  **première côte de la série utilisée** (ou la plus petite si le tri a eu lieu) ;
* ce pourcentage n'est **pas** `1/cible`, ni `P(MAX ≥ cible)`. C'est un artefact
  d'interface : il décrit la probabilité implicite d'**une seule** des cotes
  saisies, souvent la plus petite, donc la plus flatteuse.

Exemple mesuré : sur les 104 893 manches, le réglage conseillé par le guide
affiche en moyenne **83,19 %** alors que la cible produite (≈ 3,38x) est
réellement atteinte en 3 manches dans **71,13 %** des cas (Brier du pourcentage
affiché : 0,21963, contre 0,17485 pour la loi marginale).

### 2.5 La validation est humaine (L910-965)

```js
onclick="updatePredictionStatus('${prediction.id}', 'success')"   // « ✓ Validé »
onclick="updatePredictionStatus('${prediction.id}', 'failed')"    // « ✕ Échoué »
// puis un <select> propose « Round 1 … Round 5 »
```

`updateStatistics()` (L970-975) calcule ensuite :

```js
total    = predictionHistory.length
completed= predictionHistory.filter(p => p.status !== 'pending').length
success  = predictionHistory.filter(p => p.status === 'success').length
failed   = predictionHistory.filter(p => p.status === 'failed').length
```

et le taux affiché est `success / (success + failed)`. **Une prédiction que
l'utilisateur ne clôture pas reste `pending` et sort du dénominateur.** Aucune
comparaison automatique entre la cible et les manches suivantes n'existe dans le
code : c'est l'utilisateur qui juge, et qui choisit même *à quelle manche* la
prédiction est déclarée gagnante.

---

## 3. Protocole de reproduction (aucune fuite)

`crashpredict/legacy/` reproduit l'algorithme à l'identique, y compris
`toFixed(2)` (voir § 6). L'évaluation automatise ce que le robot laissait à
l'utilisateur :

```
pour chaque manche d'ancrage a :
    entrée   = m[a−w … a−1]                 (w cotes PASSÉES, aucune future)
    T        = formule du mode (arrondie à 2 décimales)
    règle 3  : succès si MAX(m[a], m[a+1], m[a+2]) ≥ T
    règle 5  : succès si MAX(m[a] … m[a+4])       ≥ T
```

* **Aucune manche future n'entre dans le calcul de `T`** : test dédié
  (`test_targets_ignore_future_rounds`) qui modifie le futur et vérifie que les
  cibles antérieures ne bougent pas d'un iota.
* Les `horizon` dernières manches de la série forment la fenêtre de validation,
  pas l'échantillon : elles ne sont jamais comptées comme prédictions.
* **Référence** : loi marginale causale, `P(MAX_h ≥ T) = 1 − (1 − F(T))^h`, où
  `F` est la fonction de survie empirique des 5 000 manches qui précèdent
  l'ancrage, réévaluée sur la cible **courante**. C'est la même règle que le
  moteur en direct : un robot sans information doit obtenir exactement ce taux.
* **Statistiques** : IC de Wilson (valides près de 0 et 1), écart apparié
  `succès − p_référence` avec IC bootstrap (1 000 rééchantillonnages), test
  binomial, puis **correction de Holm** sur toute la famille de tests.
* **Contrôle du chevauchement** : par défaut toutes les manches sont utilisées
  (estimateur plus précis, IC légèrement optimistes car les fenêtres de
  validation se recouvrent). Une seconde passe ne garde qu'une manche sur
  `horizon` : les fenêtres deviennent disjointes et les IC exacts.

---

## 4. Résultats

### 4.1 Les trois modes sous la règle des 3 manches (104 881 prédictions)

| Stratégie | Prédictions | Succès | Échecs | Taux | IC 95 % | Loi marginale | Écart | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| Standard | 104 881 | 74 602 | 30 279 | 71,13 % | [70,86 ; 71,40] | 71,06 % | +0,07 pt | 1,000 |
| Avancé | 104 881 | 71 040 | 33 841 | 67,73 % | [67,45 ; 68,02] | 67,56 % | +0,17 pt | 1,000 |
| Professionnel | 104 881 | 57 617 | 47 264 | 54,94 % | [54,63 ; 55,24] | 54,79 % | +0,14 pt | 1,000 |
| Professionnel + tendance | 104 881 | 40 617 | 64 264 | 38,73 % | [38,43 ; 39,02] | 38,63 % | +0,08 pt | 1,000 |

**Aucun mode ne dépasse le taux que la loi marginale annonçait déjà.** Les écarts
(+0,07 à +0,17 point) sont dans le bruit : aucun n'est significatif après
correction de Holm.

### 4.2 Sous l'horizon 5 de la promesse commerciale

| Stratégie | Prédictions | Succès | Échecs | Taux | IC 95 % | Loi marginale | Écart | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| Standard | 104 879 | 89 119 | 15 760 | 84,97 % | [84,76 ; 85,19] | 84,81 % | +0,16 pt | 1,000 |
| Avancé | 104 879 | 86 322 | 18 557 | 82,31 % | [82,07 ; 82,54] | 81,99 % | +0,31 pt | **0,047** |
| Professionnel | 104 879 | 72 931 | 31 948 | 69,54 % | [69,26 ; 69,82] | 69,32 % | +0,21 pt | 1,000 |
| Professionnel + tendance | 104 879 | 55 174 | 49 705 | 52,61 % | [52,31 ; 52,91] | 52,55 % | +0,04 pt | 0,747 |

Le mode « Avancé » franchit de justesse le seuil (p = 0,047) : c'est le seul
écart significatif de tout le dispositif. Il ne survit pas aux contrôles ci-dessous.

### 4.3 Pourquoi ce +0,31 point n'est pas un avantage

1. **Contrôle du chevauchement.** En ne gardant qu'une manche sur 5 (fenêtres de
   validation disjointes, IC exacts) : Avancé passe à **82,13 %** contre 81,99 %
   annoncés, écart **+0,17 pt, p = 0,49**. L'avantage disparaît.
2. **Sensibilité à la fenêtre de saisie** (5, 10, 20 cotes ; 3 ordres ; 2 horizons
   ; 4 modes = 72 évaluations complètes, corrigées par Holm). Les écarts
   significatifs apparaissent sur des cellules isolées et **changent de signe** :
   marge 5 cotes +0,05, marge 10 cotes +0,31, marge 20 cotes +0,18 pour Avancé ;
   pour Standard : +0,06 / +0,16 / +0,28. Un avantage réel serait stable.
3. **Sous-périodes.** Sur les tiers de la série, Avancé donne +0,05 / +0,25 /
   +0,62 pt. La significativité vient du dernier tiers, alors qu'aucune
   modification du jeu n'est documentée sur cette plage — signature d'une
   fluctuation, pas d'une structure.
4. **Le témoin sans information.** En rejouant les mêmes cibles avec la seule loi
   marginale, on obtient 82,29 % là où Avancé obtient 82,13 %.

**Conclusion** : le taux de réussite de ROBOT V4.5 est **entièrement expliqué par
le niveau de cible qu'il choisit**. Plus la cible proposée est basse, plus le
taux est élevé — sans aucune compétence prédictive. Le mode Standard réussit à
85 % sur 5 manches simplement parce que sa cible médiane (≈ 2,7x) est atteinte
dans 5 manches environ 85 % du temps.

### 4.4 D'où viennent les 85–95 % du guide

Le guide annonce « prédictions entre 85 % et 100 % : très haute probabilité de
réalisation ». Ce chiffre se reconstruit **sans aucune falsification**, par deux
mécanismes qui se cumulent :

**a) Le biais de survie de la validation manuelle.** Le dénominateur ne contient
que les prédictions clôturées. Si l'utilisateur ne marque « échoué » qu'une
partie des échecs réels :

| Part des échecs honnêtement marqués | Taux affiché (5 manches) | Taux affiché (3 manches) |
|---|---|---|
| 100 % | 84,97 % | 71,13 % |
| 70 % | 88,98 % | 77,87 % |
| 50 % | 91,88 % | 83,13 % |
| 30 % | 94,96 % | 89,15 % |
| 20 % | 96,58 % | 92,49 % |
| 10 % | 98,26 % | 96,10 % |

Il suffit de laisser en attente la moitié des échecs pour afficher 91,9 %.
Aucune donnée n'est inventée : seulement un dénominateur incomplet.

**b) Le pourcentage affiché n'est pas la probabilité de la cible.** `1/o₀` est
souvent bien supérieur à `P(MAX ≥ cible)` (83,19 % affichés contre 71,13 % réels
sur 3 manches). Lu comme une « précision », il fabrique l'impression d'un robot
qui ne se trompe jamais.

**c) Le choix de l'horizon.** 85 % est atteint en 5 manches, mais 71 % en 3.
Présenter le chiffre de 5 manches sans le comparer au taux de base fait passer un
résultat banal pour une performance.

### 4.5 La stratégie de mise du guide est perdante

Le guide recommande : miser à chaque manche, commencer au 3e round après deux
échecs, et « pari auto avec multiplication de la mise en cas de perte ». Simulation
manche par manche (bankroll initiale = 1, mise de base 2 %) :

| Stratégie | Politique | Essais | Taux d'atteinte | Profit | Espérance théorique | Décroissance max | Ruines | Témoin sans information |
|---|---|---|---|---|---|---|---|---|
| Standard | mise à chaque manche | 20 976 | 84,81 % | **−4,562** | −9,320 | 4,533 | 13 | −4,031 |
| Standard | report au 3e round | 20 976 | 70,86 % | **−13,002** | −7,324 | 4,066 | 14 | −4,471 |
| Standard | martingale (5 essais) | 20 976 | 84,81 % | **−56,452** | −34,938 | 17,201 | **126** | −112,108 |
| Avancé | mise à chaque manche | 20 976 | 82,13 % | **−0,218** | −9,897 | 4,484 | 10 | −0,891 |
| Avancé | report au 3e round | 20 976 | 67,30 % | **−13,635** | −7,629 | 4,722 | 14 | +2,881 |
| Avancé | martingale (5 essais) | 20 976 | 82,13 % | **−57,849** | −41,350 | 28,016 | **120** | −7,243 |
| Professionnel | mise à chaque manche | 20 976 | 69,29 % | **−10,544** | −13,586 | 10,587 | 17 | −4,063 |
| Professionnel | martingale (5 essais) | 20 976 | 69,29 % | **−340,027** | −67,986 | 73,035 | **375** | +126,928 |

Toutes les politiques sont **perdantes**, et l'espérance théorique négative
(pourtant calculée avec les seules fréquences du passé) le confirme : c'est la
marge maison. Trois enseignements concrets :

* augmenter le taux de réussite ne protège de rien — le mode Standard réussit à
  85 % et perd quand même de l'argent, parce que 15 % d'échecs coûtent plus que
  85 % de gains à 2,7x ne rapportent ;
* le « report au 3e round » **dégrade** le taux (84,8 % → 70,9 %) : les atteintes
  des manches 1 et 2 sont les plus fréquentes (36,8 % des prédictions et 21,2 %
  sont déjà atteintes là) ;
* la martingale multiplie le nombre de ruines par ~10 (13 → 126) sans améliorer
  le taux d'un iota.

Un **témoin sans information** (mêmes cibles, mêmes mises, mêmes dates, atteinte
tirée selon la loi marginale) produit des profits du même ordre, souvent
meilleurs : le résultat réel du robot n'est pas distinguable de la chance.

### 4.6 Comparaison avec la génération actuelle

Comparaison appariée sur les 39 997 manches couvertes par le dernier
walk-forward (5 887 042 → 5 927 038, soit 38,1 % de la série ; hors de cette
plage, il n'existe aucune probabilité ML à comparer) :

| Configuration | Verdict après Holm |
|---|---|
| modèles ML meilleurs que la loi marginale | **0** |
| modèles ML significativement pires | **8** |
| sans différence démontrée | **24** |

**Les deux générations échouent au même endroit.** ROBOT V4.5 obtient ses 85 %
en baissant la cible ; les modèles actuels obtiennent leurs probabilités
calibrées en estimant la loi marginale et ne trouvent, comme le robot, aucune
structure sérielle exploitable. Aucune des deux ne bat la loi marginale.

---

## 5. Faut-il garder ROBOT V4.5 ?

Le cahier des charges prévoyait de conserver une stratégie legacy si elle battait
les modèles. **Ce n'est pas le cas, et il faut le dire clairement** : aucune des
sept stratégies évaluées ne démontre d'avantage, et l'unique écart significatif
disparaît dès qu'on contrôle le chevauchement des fenêtres.

Ce qui est conservé, en revanche, est plus utile que le robot lui-même :

* le **module de reproduction** (`crashpredict/legacy/robotv45.py`), utilisable
  comme calculateur de référence et comme outil pédagogique : n'importe qui peut
  vérifier ce que le robot conseille et à quel point son « % » est décoratif ;
* les **deux horizons de validation** dans le système actuel : 3 manches pour la
  règle officielle, 5 manches pour la comparaison honnête avec la promesse
  commerciale de la génération précédente ;
* la **décomposition du biais de survie**, qui explique pourquoi les deux
  générations peuvent afficher des taux flatteurs sans jamais les démontrer.

La page **🤖 ROBOT V4.5** de l'interface expose tout cela, y compris un
calculateur interactif du robot d'origine.

---

## 6. Fidélité de la reproduction (comment elle est prouvée)

La reproduction n'est pas une lecture approximative du code : elle est comparée
**au JavaScript d'origine exécuté par Node**. `legacy/tools/oracle.js` découpe le
fichier HTML d'origine, en extrait la fonction `calculateTrend` et le bloc
L742-786 **sans les réécrire**, et les évalue. `tests/test_legacy_robotv45.py`
lance cet oracle sur des saisies aléatoires (5, 6, 8 et 12 cotes) pour les
3 modes × 2 options × 2 options, et compare cible et pourcentage affiché au bit
près : **12 tests de fidélité**, tous verts.

Un détail a été détecté par ce test et non par lecture : `toFixed(2)` de
JavaScript n'arrondit pas comme `round()` de Python. `toFixed` arrondit la valeur
binaire exacte du double, les égalités étant tranchées vers le haut
(`(2.675).toFixed(2) === "2.67"`), tandis que `round()` arrondit au pair sur la
représentation décimale courte. `js_to_fixed()` reproduit donc `Decimal(float)`
avec `ROUND_HALF_UP`. Sans cela, environ 1 cible sur 1 000 différait, ce qui
suffit à décaler un comptage de succès.

---

## 7. Reproduire les résultats

```bash
# évaluation complète (~9 min sur 2 cœurs) -> reports/robotv45.json + table metrics
PYTHONPATH=. .venv/bin/python scripts/legacy_backtest.py

# régénérer le rapport scientifique (HTML/MD/JSON) qui inclut la section ROBOT V4.5
PYTHONPATH=. .venv/bin/python scripts/report.py

# tests, dont les 12 tests de fidélité contre le JavaScript d'origine
PYTHONPATH=. .venv/bin/python -m pytest tests -q
```

Options utiles : `--window 5|10|20`, `--order recent|chrono|asc`,
`--no-ml` (sauter la comparaison ML), `--no-sensitivity` (sauter les 72
évaluations de sensibilité, ~7 min), `--no-persist`.

---

## 8. Réserves méthodologiques (à ne pas retirer)

1. **Ordre de saisie non déterminé par le code d'origine.** Le mode « Avancé »
   pondère selon la position dans le champ de texte ; le robot ne précise jamais
   si la première côte est la plus récente. Trois interprétations sont évaluées
   (`recent`, `chrono`, `asc`) ; aucune ne change la conclusion.
2. **Chevauchement des fenêtres.** Par défaut, chaque manche ouvre une fenêtre de
   validation : les observations ne sont pas indépendantes, les IC sont
   légèrement optimistes. La passe non chevauchante est fournie pour cette raison,
   et c'est elle qui tranche.
3. **Recouvrement partiel avec le ML.** La comparaison avec les modèles porte sur
   38,1 % de la série (plage du walk-forward), pas sur 104 893 manches.
4. **Aucune manche ne provient du robot.** Les 104 893 manches sont les manches
   BC.Game vérifiées (chaîne SHA256, 0 trou, 0 doublon) ; les prédictions du
   robot sont recalculées, jamais importées.
5. **La simulation de mise suppose un cash-out exact à la cible.** C'est la
   lecture stricte du guide ; une exécution réelle subirait latence et
   glissement, ce qui dégraderait encore le résultat.
