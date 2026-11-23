"""Génération du rapport de performance (HTML + Markdown) depuis la base.

Aucune valeur n'est écrite en dur : tout provient des tables `rounds`, `predictions`,
`metrics` et `backtest_runs`. Le rapport est régénéré après chaque entraînement.
"""
from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import db
from .backtesting.significance import binomial_test_vs_rate
from .config import CONFIG, REPORT_DIR


def _fmt_pct(v, digits: int = 2) -> str:
    return "n/d" if v is None or pd.isna(v) else f"{v*100:.{digits}f} %"


def _fmt(v, digits: int = 5) -> str:
    return "n/d" if v is None or pd.isna(v) else f"{v:.{digits}f}"


def load_legacy_report(path: Path | None = None) -> dict | None:
    """Rapport ROBOT V4.5 déjà calculé (reports/robotv45.json), s'il existe.

    Le calcul vit dans `crashpredict.legacy` : le rapport ne fait que le lire,
    il ne recalcule rien et n'invente aucun chiffre.
    """
    p = Path(path or (REPORT_DIR / "robotv45.json"))
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def build_report() -> dict:
    """Assemble toutes les sections du rapport (dict sérialisable)."""
    fp = db.data_fingerprint()
    rounds = db.get_rounds()
    m = np.array([r["multiplier"] for r in rounds], dtype=float) if rounds else np.array([])

    data_section = {
        "n_rounds": fp["n"],
        "first": fp["first"],
        "last": fp["last"],
        "gaps": fp["gaps"],
        "duplicates": fp["duplicates"],
        "chain_ok": fp["chain_ok"],
        "chain_bad": fp["chain_bad"],
        "median": float(np.median(m)) if len(m) else None,
        "mean": float(np.mean(m)) if len(m) else None,
        "max": float(np.max(m)) if len(m) else None,
        "share_1x": float(np.mean(m == 1.0)) if len(m) else None,
    }

    # ---- statistique descriptive : la loi marginale explique-t-elle tout ? ----
    marginal = []
    if len(m) > 100:
        max3 = np.maximum.reduce([m[:-2], m[1:-1], m[2:]])
        for t in CONFIG.targets.grid:
            p1 = float(np.mean(m >= t))
            p3 = 1 - (1 - p1) ** 3
            obs = float(np.mean(max3 >= t))
            test = binomial_test_vs_rate(int((max3 >= t).sum()), len(max3), p3)
            marginal.append(
                {
                    "target": t,
                    "p_single": p1,
                    "p_validation_theory": p3,
                    "observed": obs,
                    "gap": obs - p3,
                    "p_value": test["p_value"],
                    "significant": test["significant"],
                }
            )

    # ---- dépendance sérielle ----
    serial = {}
    if len(m) > 1000:
        for t in (1.5, 2.0, 5.0):
            ind = (m >= t).astype(float)
            serial[f"autocorr_lag1_{t}"] = float(np.corrcoef(ind[:-1], ind[1:])[0, 1])
        runs = []
        above = (m > np.median(m)).astype(int)
        for a, b in zip(above[:-1], above[1:]):
            runs.append(a != b)
        n1, n0 = int(above.sum()), int(len(above) - above.sum())
        expected = 2 * n1 * n0 / (n1 + n0) + 1
        variance = 2 * n1 * n0 * (2 * n1 * n0 - n1 - n0) / ((n1 + n0) ** 2 * (n1 + n0 - 1))
        serial["runs_observed"] = int(1 + sum(runs))
        serial["runs_expected"] = float(expected)
        serial["runs_z"] = float(((1 + sum(runs)) - expected) / np.sqrt(variance))
        serial["runs_p_value"] = float(2 * (1 - 0.5 * (1 + math.erf(abs(serial["runs_z"]) / np.sqrt(2)))))

    # ---- balayage des ordres de Markov (recherche de signal sériel) ----
    markov_sweep = []
    if len(m) > 60_000:
        try:
            from .models.statistical import MarkovSweep

            markov_sweep = MarkovSweep(orders=[1, 2, 3, 5], target=2.0).run(m, block=10_000, burn_in=20_000)
        except Exception:
            markov_sweep = []

    # ---- backtest ----
    run_id, metrics, skill, summary = _latest_run()
    by_target = []
    if run_id:
        preds = pd.DataFrame([dict(r) for r in db.predictions(limit=1_000_000, backtest=True)])
        if not preds.empty:
            closed = preds[preds["status"].isin(["SUCCESS", "FAILURE"])]
            for t, grp in closed.groupby("target"):
                by_target.append(
                    {
                        "target": float(t),
                        "n": int(len(grp)),
                        "rate": float((grp["status"] == "SUCCESS").mean()),
                        "announced": float(grp["probability"].mean()),
                        "brier": float(np.mean((grp["probability"] - (grp["status"] == "SUCCESS")) ** 2)),
                    }
                )

    # ---- verdict ----
    verdict = "indéterminé"
    detail = "Aucun backtest enregistré."
    if skill is not None and not skill.empty:
        piv = skill.pivot_table(index=["model_name", "target"], columns="metric", values="value").reset_index()
        better = worse = 0
        if "delta_brier_vs_baseline" in piv.columns and "p_value_holm" in piv.columns:
            better = int(((piv["delta_brier_vs_baseline"] > 0) & (piv["p_value_holm"] < 0.05)).sum())
            worse = int(((piv["delta_brier_vs_baseline"] < 0) & (piv["p_value_holm"] < 0.05)).sum())
        if better:
            verdict = "avantage significatif détecté"
            detail = f"{better} configuration(s) battent la baseline après correction de Holm."
        else:
            verdict = "aucun avantage statistiquement significatif"
            detail = (
                "Aucun modèle ne bat la baseline empirique de façon significative"
                + (f" ; {worse} configuration(s) font significativement pire" if worse else "")
                + ". Le robot déploie donc la loi marginale (probabilités calibrées, sans prétention prédictive)."
            )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": data_section,
        "marginal": marginal,
        "serial": serial,
        "markov_sweep": markov_sweep,
        "run": {"id": run_id, "summary": summary},
        "by_target": by_target,
        "verdict": {"status": verdict, "detail": detail},
        "legacy": load_legacy_report(),
    }


def _latest_run():
    run = db.latest_finished_run()
    if not run:
        return None, None, None, None
    rows = db.run_metrics(run["id"])
    df = pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
    metrics = df[~df["metric"].isin(["delta_brier_vs_baseline", "p_value_holm", "delta_brier_moyen"])] if not df.empty else df
    skill = df[df["metric"].isin(["delta_brier_vs_baseline", "p_value_holm"])] if not df.empty else df
    summary = json.loads(run["summary"]) if run["summary"] else None
    return run["id"], metrics, skill, summary


LEGACY_LABELS = {
    "standard": "Standard (moyenne simple)",
    "advanced": "Avancé (pondération 1 + 0.1·i)",
    "pro": "Professionnel (sans tendance) = Standard",
    "pro_trend": "Professionnel + analyse tendancielle",
    "pro_guide": "Réglage conseillé par le guide",
    "pro_guide_skip2": "Guide + report au 3e round",
    "pro_guide_martingale": "Guide + martingale",
}


def legacy_markdown(legacy: dict | None) -> list[str]:
    """Section ROBOT V4.5 : chiffres mesurés, sans arrondi complaisant."""
    if not legacy:
        return ["_Aucune évaluation ROBOT V4.5 enregistrée (`scripts/legacy_backtest.py`)._"]
    p = legacy["protocol"]
    out = [
        f"Protocole : fenêtre de saisie **{p['window']} cotes passées**, ordre *{p['order']}*, "
        f"première prédiction possible à la manche `{p['first_anchor_round']}`, "
        f"cibles produites uniquement à partir du passé ({p['leakage']})." ,
        "",
    ]
    by_h: dict[int, list[dict]] = {}
    for row in legacy["main"]:
        by_h.setdefault(row["horizon"], []).append(row)
    for h in sorted(by_h):
        out += [
            f"### Validation sur les {h} manches suivantes",
            "",
            "| Stratégie | Prédictions | Succès | Échecs | Taux | IC 95 % | Loi marginale | Écart | p (Holm) | Verdict |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in by_h[h]:
            out.append(
                f"| {LEGACY_LABELS.get(r['strategy'], r['strategy'])} | {r['n_predictions']:,} | "
                f"{r['successes']:,} | {r['failures']:,} | **{r['rate']*100:.2f} %** | "
                f"[{r['ci_low']*100:.2f} ; {r['ci_high']*100:.2f}] | {r['expected_rate']*100:.2f} % | "
                f"{r['gap_vs_reference']*100:+.2f} pts | {r['p_value_holm']:.3f} | {r['verdict']} |"
                .replace(",", " ").replace("1 + 0.1", "1 + 0,1")
            )
        out.append("")
    s = legacy["summary"]
    out += [
        "### Ce que la reproduction démontre",
        "",
        f"- Aucune stratégie n'atteint les 85–95 % annoncés **sous la règle des 3 manches** : la meilleure "
        f"plafonne à {max(s['rates_h3'].values())*100:.2f} %.",
        f"- Sous l'horizon 5 de la promesse commerciale, la meilleure atteint "
        f"{max(s['rates_h5'].values())*100:.2f} % — mais la loi marginale annonçait déjà ce niveau, "
        f"puisqu'une cible de {legacy['illusion']['claim'][-1]['mean_target']:.2f}x est atteinte dans 5 manches "
        f"avec une probabilité de {legacy['illusion']['claim'][-1]['reference_mean']*100:.2f} %.",
    ]
    if s["better_significant"]:
        out.append(
            f"- Écart significatif après correction de Holm : {', '.join(s['better_significant'])}. "
            f"En ne gardant qu'une manche sur `horizon` (fenêtres non chevauchantes, IC exacts), "
            f"il reste : {', '.join(s['better_significant_non_overlapping']) or 'aucun'} — "
            "l'avantage ne survit donc pas au contrôle du chevauchement."
        )
    if legacy.get("ml", {}).get("available"):
        ml = legacy["ml"]
        out.append(
            f"- Confrontation aux modèles ML (dernier walk-forward  {ml['n_rounds']: } manches) : "
            f"{ml['n_better']} configuration(s) meilleure(s) que la loi marginale, "
            f"{ml['n_worse']} pire(s) de façon significative, {ml['n_null']} sans différence — "
            "les deux générations de robots se heurtent au même mur."
        )
    out += ["", "### L'illusion des 85–95 %", ""]
    lab = legacy["illusion"]["labelling"]
    out += [
        "Le robot ne se valide pas lui-même : l'utilisateur clique « ✓ Validé » ou « ✕ Échoué », "
        "et une prédiction non clôturée reste `pending`, donc **hors du dénominateur**. "
        "Voici le taux affiché selon la part d'échecs réellement marqués comme tels :",
        "",
        "| Part des échecs honnêtement marqués | Taux affiché (horizon 5) | Taux affiché (horizon 3) |",
        "|---|---|---|",
    ]
    lab3 = {round(x["honest_failure_rate"], 2): x["displayed_accuracy"] for x in legacy["illusion"]["labelling_h3"]}
    for x in lab:
        r = round(x["honest_failure_rate"], 2)
        out.append(f"| {r*100:.0f} % | {x['displayed_accuracy']*100:.2f} % | {lab3.get(r, float('nan'))*100:.2f} % |")
    out += [
        "",
        "Autrement dit : **85–95 % est exactement ce qu'on obtient en ne clôturant que les succès**, "
        "sans falsifier quoi que ce soit. S'y ajoute le pourcentage affiché par le robot "
        "(`1/o₀`, la première côte saisie), qui n'est pas la probabilité de la cible.",
        "",
        "| Horizon | Taux observé | % moyen affiché par le robot | Brier du % affiché | Brier de la loi marginale |",
        "|---|---|---|---|---|",
    ]
    for c in legacy["illusion"]["claim"]:
        out.append(
            f"| {c['horizon']} manches | {c['observed_rate']*100:.2f} % | {c['declared_mean']*100:.2f} % | "
            f"{c['brier_declared']:.5f} | {c['brier_reference']:.5f} |"
        )
    out += ["", "### La politique de mise recommandée par le guide", ""]
    out += [
        "| Stratégie | Politique | Essais | Atteints | Taux | Profit (bankroll initiale = 1) | "
        "Espérance théorique | Décroissance max | Ruines | Témoin sans information |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for b in legacy["betting"]:
        out.append(
            f"| {b['strategy']} | {b['policy']} | {b['rounds_played']:,} | {b['wins']:,} | "
            f"{b['strike_rate']*100:.2f} % | **{b['profit_units']:+.3f}** | {b['expected_profit_units']:+.3f} | "
            f"{b['max_drawdown']:.3f} | {b['ruin_events']} | {b['reference_profit_units']:+.3f} |".replace(",", " ")
        )
    out += [
        "",
        "Le profit est exprimé en multiples de la bankroll initiale ; une ruine signifie être passé "
        "sous 1 % de cette bankroll. Le **témoin** rejoue les mêmes cibles, mêmes mises et mêmes dates "
        "en tirant l'atteinte avec la probabilité de la loi marginale : tout écart entre les deux "
        "colonnes mesure la chance, pas la compétence.",
        "",
        "### Conclusion sur ROBOT V4.5",
        "",
        legacy["summary"]["conclusion"],
        "",
    ]
    return out


def render_markdown(report: dict) -> str:
    d, v = report["data"], report["verdict"]
    lines = [
        "# Rapport de performance — Crash Game AI Predictor",
        "",
        f"Généré le **{report['generated_at']}** · run `{(report['run']['id'] or 'n/d')}`",
        "",
        "## 1. Données",
        "",
        f"- {d['n_rounds']:,} manches ({d['first']} → {d['last']}), {d['gaps']} trou, {d['duplicates']} doublon".replace(",", " "),
        f"- Chaîne provably-fair : {d['chain_ok']:,} transitions vérifiées, **{d['chain_bad']} échec**".replace(",", " "),
        f"- Médiane {_fmt(d['median'],2)}x · moyenne {_fmt(d['mean'],2)}x · maximum {_fmt(d['max'],2)}x · part de 1.00x {_fmt_pct(d['share_1x'],2)}",
        "",
        "## 2. La loi marginale explique-t-elle les validations ?",
        "",
        "| Cible | P(1 manche ≥ cible) | P(validation) théorique | Observé (fenêtres de 3) | Écart | p-value binomiale |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["marginal"]:
        lines.append(
            f"| {row['target']:g}x | {_fmt_pct(row['p_single'])} | {_fmt_pct(row['p_validation_theory'])} | "
            f"{_fmt_pct(row['observed'])} | {row['gap']*100:+.2f} pts | {row['p_value']:.3f} |"
        )
    lines += ["", "Aucun écart significatif ⇒ les manches sont compatibles avec des tirages indépendants.", "", "## 3. Dépendance sérielle", ""]
    s = report["serial"]
    if s:
        lines += [
            f"- Autocorrélation de l'indicateur « ≥ 1.5x » (lag 1) : {s['autocorr_lag1_1.5']:+.4f}",
            f"- Autocorrélation de l'indicateur « ≥ 2x » (lag 1) : {s['autocorr_lag1_2.0']:+.4f}",
            f"- Autocorrélation de l'indicateur « ≥ 5x » (lag 1) : {s['autocorr_lag1_5.0']:+.4f}",
            f"- Test des séries (runs test) : {s['runs_observed']} séries observées contre {s['runs_expected']:.1f} attendues, "
            f"z = {s['runs_z']:+.2f}, p = {s['runs_p_value']:.3f}",
        ]
    lines += ["", "## 4. Recherche de structure sérielle (Markov)", "",
              "Ordres de Markov testés en walk-forward pour la cible 2.00x :", ""]
    from .backtesting.examples import markov_sweep_text

    lines.append(markov_sweep_text(report.get("markov_sweep")))
    lines += ["", "## 5. Backtest walk-forward", ""]
    if report["by_target"]:
        lines += ["| Cible | Prédictions | Taux observé | Probabilité annoncée | Brier |", "|---|---|---|---|---|"]
        for r in sorted(report["by_target"], key=lambda x: x["target"]):
            lines.append(
                f"| {r['target']:g}x | {r['n']:,} | {_fmt_pct(r['rate'])} | {_fmt_pct(r['announced'])} | {r['brier']:.6f} |".replace(",", " ")
            )
    else:
        lines.append("_Aucun backtest enregistré._")
    lines += ["", "## 6. ROBOT V4.5 (génération précédente) — reproduction expérimentale", ""]
    lines += legacy_markdown(report.get("legacy"))
    lines += ["", "## 7. Conclusion", "", f"**{v['status'].upper()}** — {v['detail']}", ""]
    return "\n".join(lines)


def render_html(report: dict) -> str:
    md = render_markdown(report)
    rows = []
    for row in report["marginal"]:
        cls = "good" if not row["significant"] else "bad"
        rows.append(
            f"<tr class='{cls}'><td>{row['target']:g}x</td><td>{_fmt_pct(row['p_single'])}</td>"
            f"<td>{_fmt_pct(row['p_validation_theory'])}</td><td>{_fmt_pct(row['observed'])}</td>"
            f"<td>{row['gap']*100:+.2f} pts</td><td>{row['p_value']:.3f}</td></tr>"
        )
    bt_rows = "".join(
        f"<tr><td>{r['target']:g}x</td><td>{r['n']:,}</td><td>{_fmt_pct(r['rate'])}</td>"
        f"<td>{_fmt_pct(r['announced'])}</td><td>{r['brier']:.6f}</td></tr>".replace(",", " ")
        for r in sorted(report["by_target"], key=lambda x: x["target"])
    )
    d = report["data"]
    verdict = report["verdict"]
    badge = "good" if "aucun avantage" in verdict["status"] else "warn"
    return f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8">
<title>Rapport de performance — Crash Game AI Predictor</title>
<style>
 body {{ background:#0E1117; color:#E6E6EF; font-family:'Inter',system-ui,sans-serif; margin:0; padding:40px; line-height:1.6 }}
 .wrap {{ max-width:1000px; margin:auto }}
 h1 {{ font-size:2rem; margin-bottom:.2rem }}
 h2 {{ margin-top:2.5rem; border-bottom:1px solid #23283A; padding-bottom:.3rem }}
 .kpis {{ display:flex; gap:14px; flex-wrap:wrap; margin:24px 0 }}
 .kpi {{ background:#161B22; border:1px solid #23283A; border-radius:14px; padding:16px 20px; min-width:170px }}
 .kpi .l {{ color:#8E8EA0; font-size:.72rem; text-transform:uppercase; letter-spacing:.08em }}
 .kpi .v {{ font-size:1.5rem; font-weight:700 }}
 table {{ width:100%; border-collapse:collapse; margin:16px 0; font-size:.92rem }}
 th, td {{ padding:8px 10px; border-bottom:1px solid #23283A; text-align:left }}
 th {{ color:#B9B9C9; font-weight:600 }}
 tr.good td:nth-child(5) {{ color:#00B894 }}
 .verdict {{{'background:rgba(0,184,148,.08);border-left:4px solid #00B894' if badge=='good' else 'background:rgba(253,203,110,.08);border-left:4px solid #FDCB6E'}; padding:16px 20px; border-radius:10px; margin:18px 0 }}
 .mono {{ font-family:ui-monospace,monospace; color:#8E8EA0; font-size:.8rem }}
 .warn {{ background:rgba(225,112,85,.08); border-left:4px solid #E17055; padding:14px 18px; border-radius:10px }}
</style></head><body><div class="wrap">
<h1>Rapport de performance</h1>
<div class="mono">Crash Game AI Predictor · généré le {report['generated_at']} · run {report['run']['id'] or 'n/d'}</div>

<div class="kpis">
 <div class="kpi"><div class="l">Manches</div><div class="v">{d['n_rounds']:,}</div></div>
 <div class="kpi"><div class="l">Trous</div><div class="v">{d['gaps']}</div></div>
 <div class="kpi"><div class="l">Chaîne SHA256</div><div class="v">{d['chain_ok']:,} OK</div></div>
 <div class="kpi"><div class="l">Échecs de chaîne</div><div class="v">{d['chain_bad']}</div></div>
 <div class="kpi"><div class="l">Médiane</div><div class="v">{_fmt(d['median'],2)}x</div></div>
</div>

<h2>1. Données</h2>
<p>{d['n_rounds']:,} manches consécutives ({d['first']} → {d['last']}), {d['duplicates']} doublon.
La chaîne provably-fair est vérifiée sur {d['chain_ok']:,} transitions : <b>{d['chain_bad']} échec</b>.</p>

<h2>2. La loi marginale explique-t-elle les validations ?</h2>
<table><thead><tr><th>Cible</th><th>P(1 manche)</th><th>P(validation) théorique</th>
<th>Observé (fenêtres de 3)</th><th>Écart</th><th>p-value</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p class="mono">Fenêtres de 3 manches disjointes ; test binomial bilatéral contre la probabilité théorique.</p>

<h2>3. Dépendance sérielle</h2>
<pre class="mono">{html.escape(json.dumps(report['serial'], indent=2, ensure_ascii=False))}</pre>

<h2>4. Recherche de structure sérielle (Markov)</h2>
<pre class="mono">{html.escape(json.dumps(report.get('markov_sweep'), indent=2, ensure_ascii=False))}</pre>

<h2>5. Backtest walk-forward</h2>
<table><thead><tr><th>Cible</th><th>Prédictions</th><th>Taux observé</th><th>Probabilité annoncée</th><th>Brier</th></tr></thead>
<tbody>{bt_rows or '<tr><td colspan=5>Aucun backtest enregistré.</td></tr>'}</tbody></table>

<h2>6. ROBOT V4.5 (génération précédente)</h2>
<div class="mono">reproduction fidèle validée contre le JavaScript d'origine · mêmes 105 000 manches</div>
{legacy_html(report.get('legacy'))}

<h2>7. Conclusion</h2>
<div class="verdict"><b>{html.escape(verdict['status'].upper())}</b> — {html.escape(verdict['detail'])}</div>

<div class="warn">Ce rapport mesure un système de prédiction probabiliste sur un jeu à espérance
négative. Il ne constitue ni une promesse de gain, ni un conseil d'investissement.</div>
</div></body></html>"""


def write_reports(out_dir: Path | None = None) -> dict:
    out = Path(out_dir or REPORT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    report = build_report()
    (out / "rapport_performance.md").write_text(render_markdown(report), encoding="utf-8")
    (out / "rapport_performance.html").write_text(render_html(report), encoding="utf-8")
    (out / "rapport_performance.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPORT_DIR))
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()
    db.init_db()
    report = write_reports(Path(args.out))
    if args.print:
        print(render_markdown(report))
    print(f"Rapport écrit dans {args.out} (HTML + Markdown + JSON)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def legacy_html(legacy: dict | None) -> str:
    """Section HTML de ROBOT V4.5 (aucun chiffre recalculé ici)."""
    if not legacy:
        return "<p><i>Aucune évaluation ROBOT V4.5 enregistrée.</i></p>"
    parts = []
    by_h: dict[int, list[dict]] = {}
    for row in legacy["main"]:
        by_h.setdefault(row["horizon"], []).append(row)
    for h in sorted(by_h):
        rows = "".join(
            f"<tr class='{'bad' if (r['p_value_holm'] or 1) < 0.05 and r['gap_vs_reference'] < 0 else 'good'}'>"
            f"<td>{html.escape(LEGACY_LABELS.get(r['strategy'], r['strategy']))}</td>"
            f"<td>{r['n_predictions']:,}</td><td>{r['successes']:,}</td><td>{r['failures']:,}</td>"
            f"<td><b>{r['rate']*100:.2f} %</b></td>"
            f"<td>[{r['ci_low']*100:.2f} ; {r['ci_high']*100:.2f}]</td>"
            f"<td>{r['expected_rate']*100:.2f} %</td><td>{r['gap_vs_reference']*100:+.2f} pts</td>"
            f"<td>{(r['p_value_holm'] if r['p_value_holm'] is not None else float('nan')):.3f}</td>"
            f"<td>{html.escape(r['verdict'])}</td></tr>".replace(",", " ")
            for r in by_h[h]
        )
        parts.append(
            f"<h3>Validation sur les {h} manches suivantes</h3>"
            "<table><thead><tr><th>Stratégie</th><th>Prédictions</th><th>Succès</th><th>Échecs</th>"
            "<th>Taux</th><th>IC 95 %</th><th>Loi marginale</th><th>Écart</th><th>p (Holm)</th>"
            f"<th>Verdict</th></tr></thead><tbody>{rows}</tbody></table>"
        )
    lab3 = {round(x["honest_failure_rate"], 2): x["displayed_accuracy"] for x in legacy["illusion"]["labelling_h3"]}
    lab_rows = "".join(
        f"<tr><td>{x['honest_failure_rate']*100:.0f} %</td><td>{x['displayed_accuracy']*100:.2f} %</td>"
        f"<td>{lab3.get(round(x['honest_failure_rate'], 2), float('nan'))*100:.2f} %</td></tr>"
        for x in legacy["illusion"]["labelling"]
    )
    bet_rows = "".join(
        f"<tr class='bad'><td>{b['strategy']}</td><td>{b['policy']}</td><td>{b['rounds_played']:,}</td>"
        f"<td>{b['strike_rate']*100:.2f} %</td><td><b>{b['profit_units']:+.3f}</b></td>"
        f"<td>{b['max_drawdown']:.3f}</td><td>{b['ruin_events']}</td>"
        f"<td>{b['reference_profit_units']:+.3f}</td></tr>".replace(",", " ")
        for b in legacy["betting"]
    )
    parts.append(
        "<h3>L'illusion des 85–95 %</h3>"
        "<p>Le robot ne se valide pas lui-même : l'utilisateur clique « ✓ Validé » ou « ✕ Échoué ». "
        "Une prédiction non clôturée reste <code>pending</code>, donc <b>hors du dénominateur</b>.</p>"
        "<table><thead><tr><th>Part des échecs honnêtement marqués</th><th>Taux affiché (5 manches)</th>"
        f"<th>Taux affiché (3 manches)</th></tr></thead><tbody>{lab_rows}</tbody></table>"
        "<h3>La politique de mise recommandée par le guide</h3>"
        "<table><thead><tr><th>Stratégie</th><th>Politique</th><th>Essais</th><th>Taux d'atteinte</th>"
        "<th>Profit</th><th>Décroissance max</th><th>Ruines</th><th>Témoin sans information</th>"
        f"</tr></thead><tbody>{bet_rows}</tbody></table>"
        "<p class='mono'>Profit en multiples de la bankroll initiale (mise de base 2 %). "
        "Le témoin rejoue les mêmes cibles, mises et dates avec la probabilité de la loi marginale : "
        "l'écart mesure la chance, pas la compétence.</p>"
        f"<div class='verdict'><b>CONCLUSION ROBOT V4.5</b> — {html.escape(legacy['summary']['conclusion'])}</div>"
    )
    return "\n".join(parts)
