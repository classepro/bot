# Crash Game AI Predictor

Système d'analyse et de **validation statistique** de prédictions sur jeu de crash
(type BC.Game / Stake Crash). Le projet ne prétend pas prédire la manche suivante :
il produit un **coefficient cible**, estime sa **probabilité de validation sur les 3 manches
suivantes**, et **mesure honnêtement** sa performance hors échantillon, y compris quand
cette performance n'a aucune valeur.

---

## 1. Ce que fait le système

| Module | Rôle |
|---|---|
| `crashpredict/collection` | Import d'un dump vérifié (XLSX) et collecteur live de la page publique de vérification |
| `crashpredict/db.py` | Base SQLite : manches, prédictions, modèles, runs de backtest, métriques, journal de collecte |
| `crashpredict/features` | Variables **causales** uniquement (aucune fuite temporelle, testé automatiquement) |
| `crashpredict/models` | Baselines empiriques + régression logistique, forêts, LightGBM, XGBoost, CatBoost, MLP, modèles statistiques |
| `crashpredict/backtesting` | Walk-forward avec purge de 3 manches, Brier / log-loss / AUC / calibration, bootstrap et correction de Holm |
| `crashpredict/prediction` | Choix de la cible (fixe ou taux visé), probabilité calibrée, explication chiffrée |
| `crashpredict/validation` | Application **unique** de la règle `MAX(3 manches) ≥ cible` |
| `crashpredict/stats` | Agrégats du dashboard et graphiques Plotly |
| `crashpredict/ui` | Interface Streamlit (web) et PySide6 (desktop), sans logique métier |
| `scripts/` | CLI : import, collecte, entraînement, backtest, prédiction |

---

## 2. Installation

```bash
python -m venv .venv
source .venv/bin/activate            # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Optionnel :

```bash
playwright install chromium          # uniquement pour le collecteur temps réel
```

## 3. Démarrage rapide

```bash
# 1) base + import des données déjà présentes dans data/raw (104 893 manches vérifiées)
python scripts/import_data.py

# 2) entraînement + backtest walk-forward (métriques, tests de significativité)
python scripts/train.py --quick

# 3) prédiction en ligne de commande
python scripts/predict.py --target 2.5
python scripts/predict.py --rate 0.8

# 4) interface web
streamlit run crashpredict/ui/app.py
#    ou : ./start.sh   (bind 0.0.0.0, ports d'aperçu)

# 5) interface desktop (optionnelle)
pip install PySide6 && python -m crashpredict.ui.desktop.app
```

---

## 4. Données

- **Source** : table de vérification provably-fair de **BC.Game Crash** (page publique).
- **Volume** : 104 893 manches consécutives (`round_id` 5 822 149 → 5 927 041), **aucun trou**.
- **Intégrité** : `SHA256(hash[i+1]) == hash[i]` vérifié sur **104 892 transitions → 0 échec**.
- **Horodatage** : absent du dump d'origine ⇒ estimé (`ts_source='estimated'`) ; le collecteur
  live écrit des horodatages réels (`ts_source='live'`).
- Aucune donnée n'est fabriquée, imputée ou lissée : une manche manquante reste manquante.

Détails, preuves et limites : [`docs/01_ANALYSE_DONNEES_ET_PLAN.md`](docs/01_ANALYSE_DONNEES_ET_PLAN.md)
et [`data/raw/bcgame/manifest.json`](data/raw/bcgame/manifest.json).

---

## 5. Règle de validation (identique partout)

```
SUCCÈS  si  MAX(coefficient manche+1, manche+2, manche+3) >= coefficient cible
ÉCHEC   sinon
EN ATTENTE tant que les 3 manches ne sont pas connues
```

Implémentation unique : `crashpredict/validation/three_rounds.py` (utilisée par le backtest,
la validation en direct et l'interface).

---

## 6. Configuration

Tous les paramètres méthodologiques sont centralisés dans `crashpredict/config.py` :

| Paramètre | Valeur |
|---|---|
| horizon de validation | 3 manches |
| purge entraînement/test | 3 manches |
| entraînement initial | 25 000 manches |
| pas de réentraînement | 3 000 manches |
| fenêtre d'entraînement max | 60 000 manches |
| cibles standard | 1.50x / 2.00x / 2.50x / 5.00x |
| taux visé (cible optimisée) | 80 % |

Variable d'environnement `CRASH_DB` pour changer le chemin de la base.

---

## 7. Tests

```bash
pytest -q                                   # 26 tests : règle, anti-fuite, intégrité, stats
python tests/ui/check_pages.py              # les 7 pages Streamlit s'exécutent sans erreur
```

Les tests anti-fuite vérifient notamment que **modifier le futur ne modifie pas les features du passé**.

---

## 8. Documentation

- [`docs/01_ANALYSE_DONNEES_ET_PLAN.md`](docs/01_ANALYSE_DONNEES_ET_PLAN.md) — sources, preuves, plan
- [`docs/02_INSTALLATION_ET_UTILISATION.md`](docs/02_INSTALLATION_ET_UTILISATION.md) — guide pas à pas
- [`docs/03_METHODOLOGIE.md`](docs/03_METHODOLOGIE.md) — protocole statistique détaillé
- [`docs/04_RAPPORT_PERFORMANCE.md`](docs/04_RAPPORT_PERFORMANCE.md) — résultats mesurés
- `reports/rapport_performance.html` — rapport généré depuis la base

---

## 9. Avertissement

Un jeu de crash a une espérance mathématique négative (marge maison ≈ 1 % par manche sur cette
implémentation). Ce logiciel produit des probabilités **descriptives** et un historique auditable ;
il ne permet pas de gagner à long terme et ne doit pas être présenté comme tel. Jeu réservé aux
personnes majeures ; jouez de manière responsable.
