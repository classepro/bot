# Structure du code

```
bot/
├── crashpredict/
│   ├── config.py              Tous les paramètres méthodologiques (cibles, purge, fenêtres)
│   ├── db.py                  SQLite : schéma, upserts idempotents, requêtes. Aucune logique ML.
│   ├── training.py            Orchestration : walk-forward -> métriques -> persistance -> rapport
│   ├── reporting.py           Rapport HTML / Markdown / JSON généré depuis la base
│   ├── collection/
│   │   ├── cleaner.py         Rejet, dédoublonnage, ordre, vérification SHA256, horodatage estimé
│   │   ├── historical_import.py  Import des exports XLSX de la table de vérification
│   │   └── live.py            Collecteur temps réel (page publique, reprise par curseur)
│   ├── features/__init__.py   Features causales (shift/rolling), cibles MAX(3), fenêtre future
│   ├── models/
│   │   ├── baselines.py       P fixe, P glissante (5 000 / 500), loi catégorielle
│   │   ├── linear.py          Régression logistique standardisée
│   │   ├── forest.py          RandomForest, ExtraTrees
│   │   ├── gbm.py             LightGBM, XGBoost, CatBoost
│   │   ├── nn.py              MLP (si volume suffisant)
│   │   ├── statistical.py     Pareto, chaîne de Markov (ordre choisi), balayage, Hawkes
│   │   └── registry.py        Registre des fabriques de modèles
│   ├── backtesting/
│   │   ├── walk_forward.py    Splits chronologiques, purge, baselines causales, agrégation
│   │   ├── metrics.py         Brier, log-loss, AUC, calibration
│   │   ├── significance.py    Bootstrap, test apparié, Holm, taille d'échantillon
│   │   └── examples.py        Textes explicatifs (protocole, balayage Markov)
│   ├── prediction/
│   │   ├── engine.py          Moteur : probabilité, explication, prédiction persistée, stratégies
│   │   └── targets.py         Grille de cibles, cible optimisée pour un taux visé
│   ├── validation/three_rounds.py   Règle unique MAX(3) >= cible + clôture automatique
│   ├── stats/                 Agrégats du dashboard + graphiques Plotly
│   └── ui/
│       ├── app.py             Entrée Streamlit (navigation, sidebar, état des données)
│       ├── components.py      Cartes KPI, badges, tableaux
│       ├── theme.py           CSS Streamlit + feuille de style PySide6
│       ├── pages/             7 pages Streamlit
│       └── desktop/           Équivalent PySide6 (6 pages, thread d'entraînement)
├── scripts/                   import_data, collect_live, train, backtest, predict, report
├── tests/                     27 tests + test des 7 pages Streamlit (tests/ui/check_pages.py)
├── data/                      base SQLite, dumps bruts vérifiés, manifeste de provenance
├── models_store/              artefacts du dernier backtest (probabilités, métriques, importances)
├── reports/                   rapport HTML / Markdown / JSON
└── docs/                      analyse, installation, méthodologie, rapport, structure
```

## Règles d'architecture

1. **Une seule règle de validation** : `validation/three_rounds.py`, importée par tous les modules.
2. **Aucun accès réseau** hors `collection/live.py` ; aucun module métier ne lit le web.
3. **Aucune logique métier dans l'UI** : les pages appellent `prediction`, `stats`, `training`.
4. **Tout paramètre méthodologique** dans `config.py` (jamais dupliqué dans une page).
5. **Toute écriture en base** passe par `db.py` (upsert idempotent, transactions).
6. **Aucune donnée inventée** : ni en base, ni dans les graphiques, ni dans les rapports.
