"""Données : état de la collecte, source, vérifications, import, collecteur live."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ... import db
from ...config import RAW_DIR
from ...stats import charts
from ..components import dataframe, kpi_row, warn


def render() -> None:
    st.markdown(
        "<div class='cp-hero'><h1>Données & collecte</h1>"
        "<p>Chaque manche est traçable : identifiant, hash, vérification de la chaîne provably-fair, "
        "horodatage et source.</p></div>",
        unsafe_allow_html=True,
    )
    db.init_db()
    fp = db.data_fingerprint()

    kpi_row(
        [
            {"label": "Manches en base", "value": f"{fp['n']:,}".replace(",", " ")},
            {"label": "Plage", "value": f"{fp['first']} → {fp['last']}" if fp["n"] else "—",
             "sub": f"trous : {fp['gaps']}"},
            {"label": "Chaîne SHA256 vérifiée", "value": f"{fp['chain_ok']:,}".replace(",", " "),
             "sub": f"échecs : {fp['chain_bad']}", "accent": "#00B894" if not fp["chain_bad"] else "#E17055"},
            {"label": "Doublons", "value": str(fp["duplicates"]), "accent": "#00B894" if not fp["duplicates"] else "#E17055"},
        ]
    )

    st.markdown("#### Source")
    manifest_path = RAW_DIR / "bcgame" / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(
                f"""
- **Plateforme** : {manifest.get('platform')} — jeu *{manifest.get('game')}*
- **Origine** : {manifest.get('source')}
- **Dump amont** : `{manifest['upstream']['repository']}` (commit `{manifest['upstream']['commit'][:12]}`)
- **Intégrité** : {manifest['integrity']['transitions_tested']:,} transitions de hachage testées,
  **{manifest['integrity']['transitions_failed']} échec**
- **Horodatage** : {'disponible' if manifest['timestamp']['available_in_source'] else 'absent de la source'} —
  {manifest['timestamp']['handling']}
""".replace(",", " ")
            )
        with c2:
            st.markdown(
                """
- La chaîne provably-fair `SHA256(hash[i+1]) == hash[i]` prouve que la série est
  complète, ordonnée et non retouchée.
- Aucune donnée n'est fabriquée ni imputée : une manche manquante reste manquante
  et apparaît dans le compteur « trous ».
- Le collecteur live horodate chaque nouvelle manche (`ts_source='live'`), ce qui
  remplace progressivement les horodatages estimés.
"""
            )

    st.markdown("#### Dernières manches")
    rows = db.get_rounds(limit=50, ascending=False)
    if rows:
        df = pd.DataFrame(
            [{"manche": r["round_id"], "coefficient": round(r["multiplier"], 2),
              "horodatage": r["ts"], "origine du temps": r["ts_source"],
              "source": r["source"], "chaîne": {1: "✅", 0: "❌", None: "—"}[r["chain_ok"]]} for r in rows]
        )
        dataframe(df, height=320)
    else:
        warn("Base vide.", "bad")

    st.markdown("#### Distribution et fonction de survie")
    all_rows = db.get_rounds()
    if all_rows:
        mult = [r["multiplier"] for r in all_rows]
        st.plotly_chart(charts.distribution(mult), width="stretch")

    st.markdown("#### Import de données")
    with st.expander("Importer un dump XLSX de la table de vérification", expanded=False):
        st.markdown(
            "Déposez un ou plusieurs fichiers `.xlsx` exportés depuis la page de vérification "
            "(`Row No | Hash Content | Bang Content`). Les doublons sont ignorés et la chaîne est vérifiée."
        )
        files = st.file_uploader("Fichiers XLSX", type=["xlsx"], accept_multiple_files=True)
        if files and st.button("Importer", type="primary"):
            from ...collection import import_xlsx_dump

            tmpdir = RAW_DIR / "uploads"
            tmpdir.mkdir(parents=True, exist_ok=True)
            paths = []
            for f in files:
                p = tmpdir / f.name
                p.write_bytes(f.getbuffer())
                paths.append(p)
            with st.spinner("Lecture, vérification de la chaîne et insertion…"):
                report = import_xlsx_dump(paths, store=True, verbose=False)
            st.success(
                f"{report['accepted']} manches acceptées, {report['inserted']} nouvelles, "
                f"{report['skipped']} déjà présentes, {report['rejected']} rejetées, "
                f"{report['chain_pairs_tested']} transitions testées ({report['chain_failures']} échec)."
            )
            if report["chain_failures"]:
                warn(
                    f"<b>{report['chain_failures']} transitions de hachage invalides</b> : "
                    "ces lignes sont marquées, ne les utilisez pas pour conclure.",
                    "bad",
                )

    with st.expander("Collecteur temps réel (BC.Game, page publique de vérification)", expanded=False):
        st.markdown(
            """
Le collecteur lit la **page publique de vérification** (aucun compte, aucun contournement de protection),
reprend au curseur `max(round_id)` et horodate chaque manche.

```bash
pip install playwright && playwright install chromium
python scripts/collect_live.py --interval 120 --amount 500
```
Il nécessite un accès réseau au site (non disponible dans l'environnement de développement actuel).
"""
        )
        if st.button("🧪 Tester l'accès au collecteur"):
            from ...collection import collect_once

            try:
                batch = collect_once(amount=20)
            except Exception as exc:
                warn(f"Collecteur indisponible ici : <code>{exc}</code>. L'import historique reste pleinement fonctionnel.", "warn")
            else:
                st.success(f"{len(batch.records)} manches récupérées, {batch.chain_failures} échec de chaîne.")

    st.markdown("#### Journal de collecte")
    log = db.ingest_history(limit=30)
    if log:
        dataframe(
            pd.DataFrame(
                [{"date": r["ts"], "source": r["source"], "action": r["action"],
                  "insérées": r["rows_inserted"], "ignorées": r["rows_skipped"],
                  "statut": r["status"], "message": (r["message"] or "")[:120]} for r in log]
            )
        )
    else:
        st.info("Aucune opération de collecte journalisée.")
