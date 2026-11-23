# Crash Game AI Predictor — Analyse des sources de données & Plan d'architecture

Date : 2026-11-23 · Statut : **phase 1 terminée (analyse + preuve de faisabilité)** — développement du système complet à valider.

---

## 1. Ce qui a déjà été fait (et vérifié, pas supposé)

### 1.1 Sources de données testées

| Source | Accessibilité | Verdict |
|---|---|---|
| **BC.Game — Crash** (provably fair) | Page de vérification publique : `https://bcgame.com/crash` → table `#game_verify_table`, 10 000 manches max par requête, pagination par curseur « row number » | ✅ **Retenue** |
| Stake (`api.stake.com`, `/_api/graphql`) | Bloqué par Cloudflare (cookies `cf_clearance` rotatifs toutes ~5 j), nécessite compte + token | ❌ fragile / non conforme aux CGU sans compte |
| bustabit / moneypot | Ancien dump (repo supprimé depuis), API non accessible depuis cet environnement | ❌ indisponible |
| trackersino, casino.org, crashgamesplay… | Agrégateurs commerciaux, données non vérifiables / non brutes | ❌ non retenus |
| Sources web « Signal / Aviator Predictor » (APK, bots Telegram) | Aucune donnée brute, aucune preuve, code obfuscado | ❌ écartés (voir §5) |

**Important** : j'ai accès réseau uniquement à GitHub / PyPI / npm depuis cet environnement. `bcgame.com` n'est **pas** joignable d'ici.
Les données ont donc été récupérées depuis un **dump public horodaté de la table de vérification BC.Game**
(repo GitHub `AnywhereWithyou/BC-Game-Crash-History`, dossier `BC History/`, fichiers `.xlsx` exportés par un
script Playwright qui clique sur `#game_verify_submit`), puis **re-vérifiées cryptographiquement par mes soins**.

### 1.2 Preuve de provenance et d'intégrité (refaite localement, pas recopiée)

- **104 893 manches** consécutives, `round_id` de **5 822 149 à 5 927 041**, **aucun trou** (0 manche manquante).
- Chaque ligne contient `(round_id, hash, multiplier)`.
- **Chaîne provably-fair vérifiée : `SHA256(hash[i+1]) == hash[i]` sur 104 892 transitions → 0 échec.**
  Cela prouve que la série est complète, dans l'ordre, non retouchée, et issue du RNG officiel BC.Game.
- Distribution conforme au modèle « Stake-style » : crash instantané `1.00x` = **2,03 %**, `P(x ≥ 1,01) = 97,97 %`
  (marge maison ≈ 1 %), queue lourde jusqu'à 127 151x. Aucune valeur fabriquée.
- Fichier normalisé : `data/raw/bcgame/bcgame_crash_5822149_5927041.csv` (8,3 Mo, 104 893 lignes).
  SHA-256 : `90dea8a066f730500c5ac98cb2fe8a0a61b3b1c7effb0b9816104cfae0d9d736`

**Limite connue** : ce dump ne contient **pas de timestamp**. Pour l'exigence « horodatage », 2 sources de temps sont prévues :
1. Rétro-remplissage par cadence moyenne (BC.Game crash ≈ 1 manche / ~10–15 s) → **estimation, marquée comme telle** (`timestamp_source='estimated'`) ;
2. **Collecteur live** (module `collection/`) qui horodate chaque nouvelle manche (`timestamp_source='live'`) — c'est la source de vérité temporelle.
Aucun timestamp ne sera inventé silencieusement : la colonne `timestamp_source` est explicite dans la base.

### 1.3 Test de réalité statistique (fait avant d'écrire une ligne de modèle)

Fenêtres de 3 manches (n = 104 890), règle de validation `MAX(3 manches) ≥ cible` :

| Cible | P(1 manche ≥ cible) | P(validation) **observée** | P si indépendance | Écart |
|---|---|---|---|---|
| 1,50x | 0,6596 | **0,9597** | 0,9606 | −0,0008 |
| 2,00x | 0,4955 | **0,8734** | 0,8716 | +0,0018 |
| 2,50x | 0,3968 | **0,7818** | 0,7805 | +0,0013 |
| 3,00x | 0,3307 | **0,7010** | 0,7002 | +0,0008 |
| 5,00x | 0,1979 | **0,4838** | 0,4839 | −0,0001 |
| 10,0x | 0,0993 | **0,2690** | 0,2693 | −0,0003 |

Autocorrélation de l'indicateur « ≥ 2x » : lag 1 = −0,004 ; lag 2 = −0,009 ; lag 5 = −0,001 ; taux par décile de temps
= 0,498 → 0,486 (pas de dérive). **Conclusion honnête : les manches sont indiscernables d'un tirage indépendant.**
Le taux de réussite d'une cible est donc **entièrement expliqué** par la loi marginale :
`P(MAX3 ≥ t) = 1 − (1 − F(t))³`, où `F(t)` s'estime directement sur l'historique.

→ **Conséquence de conception majeure** : « prédire le coefficient de la prochaine manche » est sans objet.
Le système doit donc (a) **calibrer les probabilités par la loi empirique**, (b) **tester si les features (séries, écarts,
moyennes mobiles…) apportent un gain mesurable** sur cette baseline, et (c) **publier le résultat du test**, même
s'il est négatif. C'est exactement ce que demande la section 10 du cahier des charges ; le système est conçu pour
détecter un avantage s'il existe et pour l'annoncer s'il n'existe pas.

---

## 2. Ce que le système mesurera (définition formelle, identique partout)

Pour un instant `i` (après la manche `i`, avant la manche `i+1`) et une cible `t` :

```
validation(i, t) = ( MAX(m[i+1], m[i+2], m[i+3]) >= t )   # OUI / NON, jamais ambigu
probabilité       = P(validation = OUI | information disponible en i)
baseline          = 1 - (1 - F_emp(t))^3        # F_emp estimée sur fenêtre glissante passée
gain              = Brier(baseline) - Brier(modèle)   (+ test de DeLong bootstrap / IC 95 %)
```

Trois régimes de probabilité comparés sur données de test jamais vues :
`P_fixe` (tout l'historique d'entraînement), `P_glissante` (fenêtre 1 000/5 000), `P_modèle` (ML).
Métriques : **Brier score**, **log-loss**, **AUC**, **calibration (reliability curve)**, **skill vs baseline**,
et pour la stratégie : **taux de réussite, cote de seuil atteinte, valeur seuil `t*` où le taux chute**.

---

## 3. Architecture cible (modulaire, pas de fichier unique)

```
bot/
├── data/                         # base locale (SQLite) + exports + raw vérifiés
│   ├── crash.db                  # rounds, predictions, model_runs, metrics, config, ingest_log
│   ├── raw/                      # dumps bruts + manifest.json (URL, sha256, date)
│   └── exports/
├── crashpredict/                 # package principal
│   ├── config.py                 # chemins, seuils, cibles, hyperparamètres (YAML/env)
│   ├── db.py                     # schéma SQLite, migrations, requêtes (aucune logique ML)
│   ├── collection/
│   │   ├── base.py               # interface Collector (fetch → validate → dedupe → store)
│   │   ├── bcgame_verify.py      # collector BC.Game (Playwright, pagination par curseur)
│   │   ├── historical_import.py  # import du dump XLSX + vérif chaîne SHA256
│   │   └── live.py               # boucle temps réel + horodatage + reprise sur incident
│   ├── cleaning/                 # validation, dédoublonnage, ordre chronologique, trous
│   ├── features/                 # features causales uniquement (shift(1) systématique)
│   ├── models/                   # registry + un fichier par famille
│   │   ├── baselines.py          # P_fixe, P_glissante, fréquence catégorielle
│   │   ├── gbm.py                # LightGBM / XGBoost / CatBoost (classification calibrée)
│   │   ├── forest.py             # RandomForest / ExtraTrees
│   │   ├── linear.py             # LogReg + splines, Ridge
│   │   ├── nn.py                 # MLP (torch/sklearn) si volume & gain justifiés
│   │   └── statistical.py        # loi de Pareto estimée, Hawkes/Markov (test d'apport)
│   ├── prediction/               # choix de cible, probabilité, explication (SHAP/importances)
│   ├── validation/               # application de la règle 3 manches (module unique, partagé)
│   ├── backtesting/
│   │   ├── walk_forward.py       # splits chronologiques, réentraînement glissant
│   │   ├── metrics.py            # Brier, logloss, AUC, calibration, skill
│   │   └── significance.py       # bootstrap, tests appariés, correction multi-tests
│   ├── stats/                    # agrégats du dashboard (par cible/modèle/période, séries)
│   └── ui/                       # interface Streamlit
│       ├── app.py                # navigation
│       └── pages/                # 1_Dashboard, 2_Prediction, 3_Historique,
│                                 # 4_Entrainement, 5_Backtesting, 6_Donnees, 7_Rapport
├── scripts/                      # CLI : collect, import, train, backtest, predict, report
├── tests/                        # pytest : anti-fuite temporelle, règle de validation, dédup
├── docs/                         # ce document, installation, méthodologie, performance
└── reports/                      # rapport HTML/PDF généré (chiffres réels uniquement)
```

### 3.1 Interface : Streamlit (recommandé) vs PySide6

Streamlit est retenu **par défaut** : dashboard pro, graphiques Plotly, zéro installation native, lancement
`streamlit run crashpredict/ui/app.py`, et surtout **aperçu live possible dans cet environnement** (nécessaire pour
que tu valides le rendu). PySide6 exige un serveur X / desktop et ne se prévisualise pas ici.
Les pages couvrent les 5 modules demandés (Dashboard, Prédiction, Historique, Entraînement, Données) + Backtesting
+ Rapport scientifique. L'UI **ne contient aucune logique métier** : elle appelle les modules Python (réutilisables en CLI).

### 3.2 Anti-fuite de données (contrôles automatiques, testés)

- Toute feature en `i` n'utilise que `m[≤ i]` — construction par `shift(1)` + `rolling` en amont, jamais `center=True`.
- Interdiction de `train_test_split(shuffle=True)` ; seuls des **splits chronologiques / walk-forward**.
- Cible = `MAX(m[i+1..i+3])` ⇒ **purge de 3 manches** entre fin de train et début de validation.
- Tests pytest dédiés : un test échoue si une feature corrèle parfaitement avec le futur, si un index de validation
  est antérieur à son train, ou si la purge est absente.

### 3.3 Collecte live et horodatage

- `collector live` : boucle configurable (intervalle 60 s), reprise par curseur (`max(round_id)` en base),
  retry/backoff, journalisation dans `ingest_log`, **déduplication par `round_id` UNIQUE**.
- Chaque insert porte `source`, `collected_at`, `timestamp_source ∈ {live, estimated}`.
- Le collecteur ne contourne aucun mécanisme de protection : il lit la page publique de vérification à cadence lente
  et respecte les conditions d'utilisation ; aucune connexion de compte, aucun contournement Cloudflare.

### 3.4 Modèles comparés (même protocole, mêmes splits)

Baselines (`P_fixe`, `P_glissante`, fréquence) · Régression logistique + splines · RandomForest / ExtraTrees ·
LightGBM · XGBoost · CatBoost · MLP (si ≥ 20 k échantillons utiles) · modèles statistiques (Pareto, Markov).
Chaque modèle est jugé sur **Brier/log-loss/AUC/calibration** et sur le **gain vs baseline** avec IC 95 % bootstrap,
plus une **correction de Bonferroni/Holm** car comparer 8 modèles sur 6 cibles multiplie les faux positifs.
Hyperparamètres par recherche aléatoire bornée **à l'intérieur** de chaque fold (jamais sur le test).

### 3.5 Stratégies de cible

Cibles fixes `1,50x / 2,00x / 2,50x / 5,00x` + **cible optimisée** : pour un taux de réussite visé (ex. 80 %), la cible
est le quantile empirique `t = F⁻¹(1 − (1 − 0,80)^(1/3))` **estimé uniquement sur le passé**, avec cote et
probabilité de validation. Table « cible | probabilité estimée | historique vérifié (succès/total) » exactement comme
demandé, et **toujours sourcée** (jamais de probabilité affichée sans dénominateur et fenêtre).

---

## 4. Livrables et calendrier de réalisation

| Étape | Contenu | Livrable |
|---|---|---|
| 1 ✅ | Analyse sources, import vérifié, test de réalité | ce document + `data/raw` + manifest |
| 2 | Base SQLite + modules collecte/import/nettoyage + tests | `crashpredict/{db,collection,cleaning}` |
| 3 | Features causales + baselines + backtesting walk-forward | `features/`, `backtesting/` + premier rapport |
| 4 | Modèles ML comparés + calibration + tests de significativité | `models/`, rapport de performance |
| 5 | Module prédiction + validation 3 manches + historique | `prediction/`, `validation/` |
| 6 | Interface Streamlit (dashboard, prédiction, historique, entraînement, données, backtest) | `ui/` + aperçu live |
| 7 | Documentation (installation, usage, méthodologie) + rapport final honnête | `docs/`, `reports/` |

## 5. Ce que le système ne fera pas (transparence)

- Il **ne prédit pas** le coefficient exact de la prochaine manche, et ne prétendra jamais le faire.
- Il **ne promet aucun gain** : sur ces données, la meilleure probabilité atteignable est la loi marginale ;
  la marge maison (~1 % par manche) rend toute stratégie à espérance positive non démontrable si le RNG est sain.
- Il **ne fabrique aucune donnée** : chaque ligne de la base est traçable (round_id + hash + vérification de chaîne).
- Il **n'utilise pas** les « prédicteurs Aviator/Crash » commerciaux, qui reposent sur des captures d'écran ou des
  patterns inexistants.
- Il **affichera explicitement** « aucun avantage statistiquement significatif » si c'est le résultat des tests.

## 6. Validation demandée avant développement complet

1. **Source de données** : BC.Game crash (104 893 manches vérifiées) — OK, ou souhaites-tu une autre plateforme ?
2. **Interface** : Streamlit (aperçu live ici) — OK, ou impératif PySide6 desktop ?
3. **Cibles par défaut** : 1,50 / 2,00 / 2,50 / 5,00x + cible optimisée pour un taux visé — OK ?
4. **Périmètre** : construire les 7 étapes ci-dessus en une fois (système complet) — OK ?
