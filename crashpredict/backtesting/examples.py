"""Textes méthodologiques affichés dans l'interface (protocole du backtest)."""
from __future__ import annotations

from ..config import CONFIG


def protocol_explanation() -> str:
    return f"""
**Règle de validation (unique, appliquée partout)**

Une prédiction est un **SUCCÈS** si `MAX(coefficient manche+1, manche+2, manche+3) >= coefficient cible`,
sinon un **ÉCHEC**. Elle reste **EN ATTENTE** tant que les trois manches ne sont pas connues.

**Séparation temporelle**

| Élément | Valeur |
|---|---|
| Horizon de validation | {CONFIG.rules.horizon} manches |
| Purge entre entraînement et test | {CONFIG.rules.horizon} manches |
| Manches d'entraînement initial | {CONFIG.backtest.initial_train:,} |
| Pas de réentraînement | {CONFIG.backtest.step:,} manches |
| Fenêtre d'entraînement maximale | {CONFIG.backtest.max_train_window:,} manches |
| Probabilité baseline | loi empirique sur les {CONFIG.probabilities.long_window:,} manches précédentes |
| Découpage aléatoire | **jamais utilisé** |

La ligne `i` du jeu de données contient la cible `MAX(m[i+1..i+3])`, qui n'est connaissable qu'à la manche `i+3`.
Pour un test démarrant à la manche `s`, on n'entraîne donc que sur les lignes `i <= s - 4` : la cible de la
dernière ligne d'entraînement est entièrement connue avant la première prédiction du test.

**Métriques**

- **Brier score** : erreur quadratique de la probabilité (plus bas = mieux).
- **Log-loss**, **AUC** : discrimination.
- **Skill** : `1 - Brier(modèle) / Brier(baseline)`.
- **Δ Brier apparié** : différence élément par élément sur les mêmes manches, avec IC 95 % bootstrap et
  correction de Holm sur la famille de tests (modèles × cibles).

**Critère de déploiement**

Un modèle n'est déployé que s'il bat la baseline de façon **significative après correction de Holm**.
Sinon le robot déploie la loi empirique — et le dit explicitement, y compris dans le rapport.
""".replace(",", " ")


def markov_sweep_text(rows: list[dict] | None) -> str:
    """Rend lisible le balayage des ordres de Markov (recherche de signal sériel)."""
    if not rows:
        return "_Balayage non exécuté._"
    ref = next((r for r in rows if r["ordre"] == 0), None)
    lines = ["| Ordre | n | Brier | AUC | Taux observé |", "|---|---|---|---|---|"]
    for r in rows:
        label = "référence (constante)" if r["ordre"] == 0 else f"Markov {r['ordre']}"
        lines.append(
            f"| {label} | {r['n']:,} | {r['brier']:.6f} | {r['auc']:.4f} | "
            f"{(r['taux_observe']*100 if r['taux_observe'] is not None else float('nan')):.2f} % |".replace(",", " ")
        )
    if ref:
        best = min((r for r in rows if r["ordre"] > 0), key=lambda r: r["brier"], default=None)
        if best and best["brier"] < ref["brier"] - 1e-4:
            lines.append("\n**Signal sériel détecté** : un ordre de Markov bat la probabilité constante.")
        else:
            lines.append(
                "\n**Aucun signal sériel** : aucun ordre de Markov ne bat la probabilité constante ; "
                "lisser davantage les transitions les rend même moins précis. "
                "Les coefficients se comportent comme des tirages indépendants."
            )
    return "\n".join(lines)
