# Installation et utilisation

## 1. Prérequis

- Python **3.10+** (testé sur 3.11)
- 2 Go de RAM libres (le backtest complet en consomme ~1 Go)
- Aucune carte graphique requise (sauf MLP, uniquement CPU)

## 2. Installation

```bash
git clone <repo> && cd bot        # ou utiliser le dossier existant
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

> Les modèles LightGBM / XGBoost sont utilisés par défaut. CatBoost et le MLP sont
> détectés automatiquement : s'ils ne sont pas installés, ils n'apparaissent pas dans
> la liste des modèles (aucune erreur, aucun faux résultat).

Collecteur temps réel (facultatif) :

```bash
pip install playwright
playwright install chromium
```

## 3. Préparer les données

Le dépôt contient déjà un dump vérifié dans `data/raw/bcgame/`. Il suffit de l'importer :

```bash
python scripts/import_data.py
```

Sortie attendue :

```
bcgame_crash_5822149_5927041.csv: lues=104893 rejetées=0 insérées=104893 déjà_en_base=0
                                  chaîne_testée=104892 échecs=0
Base: {'n': 104893, 'first': 5822149, 'last': 5927041, 'gaps': 0, 'chain_ok': 104892,
       'chain_bad': 0, 'duplicates': 0}
```

L'import est **idempotent** : on peut le relancer sans créer de doublons.

### Importer d'autres fichiers

- **XLSX** de la page de vérification (`Row No | Hash Content | Bang Content`) :

```bash
python scripts/import_data.py --xlsx data/raw/uploads/
```

- **CSV** normalisé (`round_id, hash, multiplier`) :

```bash
python scripts/import_data.py --csv mes_fichiers/*.csv
```

### Collecter en temps réel (horodatage réel)

```bash
python scripts/collect_live.py --once --amount 200        # une passe
python scripts/collect_live.py --interval 120 --amount 500 # boucle, 1 passe / 2 min
```

Le collecteur lit la **page publique de vérification**, reprend au dernier `round_id`
connu, dédoublonne et vérifie la chaîne SHA256. Aucun compte, aucun contournement de
protection.

## 4. Entraîner et backtester

```bash
python scripts/train.py --quick        # 3 folds, rapide (démonstration)
python scripts/train.py                # complet (4 cibles × 4 modèles, tous les folds)
python scripts/train.py --targets 2 2.5 --models lightgbm catboost --step 5000 --max-folds 8
```

Le script :

1. exécute le **walk-forward** (entraînement sur le passé, purge de 3 manches, test sur le futur) ;
2. enregistre les métriques, les tests de significativité (bootstrap + Holm) ;
3. choisit le modèle à déployer — **la baseline empirique s'il n'y a pas de gain significatif** ;
4. écrit les prédictions historiques dans la base ;
5. génère `reports/rapport_performance.{html,md,json}`.

À la fin, `deployed_model` indique la méthode qui sera utilisée en production.

## 5. Produire une prédiction

### Ligne de commande

```bash
python scripts/predict.py --target 2.5      # cible fixe
python scripts/predict.py --rate 0.8        # cible déduite pour viser 80 % de réussite
python scripts/predict.py --strategies      # table des 4 stratégies + cible optimisée
python scripts/predict.py --validate-only   # valider les prédictions en attente
```

Exemple de sortie :

```
ID               : P9A31C2E55D41B77
Cible            : 2.50x
Probabilité      : 78.2 % (moyen)
Méthode          : loi marginale empirique (n=5000 manches)
Ancrage          : manche 5927041
Manches validées : [5927042, 5927043, 5927044]
Statut           : EN ATTENTE
Calcul           : P(1 manche ≥ cible) = 39.68 % sur 5000 manches → P(3 manches) = 78.05 %
```

La prédiction passe automatiquement à **SUCCÈS** ou **ÉCHEC** dès que les trois
manches suivant l'ancrage existent dans la base (`close_pending_predictions`, appelé
par l'interface et par `predict.py`).

### Interface web (Streamlit)

```bash
streamlit run crashpredict/ui/app.py
# ou
./start.sh
```

7 pages : Dashboard, Prédiction, Historique, Entraînement, Backtesting, Données, Rapport.

### Interface desktop (PySide6)

```bash
pip install PySide6
python -m crashpredict.ui.desktop.app
```

Mêmes fonctions que la version web : dashboard, prédiction, historique, entraînement
(thread séparé), données, rapport.

## 6. Performances de calcul

Mesures sur 2 cœurs (conteneur) avec calibration isotone chronologique :

| Périmètre | Durée |
|---|---|
| 1 fold × 4 cibles × 4 modèles (train 40 000 manches) | ~1 min |
| 8 folds × 4 cibles × 5 modèles | ~25 min |
| Entraînement complet (tous les folds, 5 modèles, 4 cibles) | ~1 h |

Chaque fold réentraîne **et recalibre** tous les modèles (choix méthodologique :
un modèle non calibré n'est pas comparable à une probabilité empirique). Pour un
essai rapide : `--max-folds 2 --step 10000`.

## 7. Tests et contrôles

```bash
pytest -q                        # 35 tests
python tests/ui/check_pages.py   # exécution réelle des 7 pages Streamlit
python scripts/report.py --print # rapport en Markdown dans le terminal
```

## 8. Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| `Base vide` | import non lancé | `python scripts/import_data.py` |
| `historique trop court pour un walk-forward` | moins de 25 000 manches | baisser `--initial-train` |
| `Collecteur indisponible` | pas de réseau vers le site, ou Playwright absent | utiliser l'import historique `--xlsx` |
| Entraînement long | 4 cibles × 5 modèles × folds | `--max-folds 3 --step 10000` |
| `ModuleNotFoundError: lightgbm` | dépendance absente | `pip install lightgbm` (ou retirer le modèle) |
| Les chiffres semblent « trop beaux » | doute légitime | lire `docs/04_RAPPORT_PERFORMANCE.md` : la section verdict tranche |

## 9. Où sont les fichiers ?

```
data/crash.db            base SQLite (tout l'historique et les prédictions)
data/raw/                dumps bruts + manifest de provenance
models_store/            artefacts du dernier backtest (probabilités, métriques, importances)
reports/                 rapport HTML / Markdown / JSON
```
