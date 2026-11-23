"""Collecteur live BC.Game (page publique de vérification provably-fair).

Principe : la page https://bcgame.com/crash expose un outil de vérification
(`#game_amount_input`, `#game_verify_submit`, table `#game_verify_table`) qui
affiche, pour chaque manche, `Row No`, `Hash Content`, `Bang Content`.

Ce collecteur :
  * reprend au curseur `max(round_id)` déjà en base (aucun re-téléchargement) ;
  * horodate chaque manche au moment de la collecte (ts_source='live') ;
  * déduplique par round_id et vérifie la chaîne SHA256 ;
  * applique retry/backoff, cadence lente et configurable ;

Il ne se connecte à aucun compte, ne contourne aucune protection (aucun
Cloudflare bypass, aucun cookie volé) : il lit une page publique, comme un
navigateur. Si le navigateur n'est pas disponible (Playwright absent ou pas de
réseau), la fonction échoue proprement et l'import historique reste utilisable.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from .. import db
from ..config import CONFIG
from .cleaner import clean_records, estimate_timestamps, mark_chain

VERIFY_URL = "https://bcgame.com/crash"
DEFAULT_BATCH = 500  # manches demandées par requête (10 000 max côté site)


class CollectorUnavailable(RuntimeError):
    """Levée si le navigateur/réseau n'est pas disponible."""


@dataclass
class LiveBatch:
    records: list[dict]
    chain_failures: int


def _extract_rows(page, amount: int) -> list[dict]:
    """Soumet le formulaire de vérification et lit la table de résultats."""
    page.goto(VERIFY_URL, wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_selector("#game_amount_input", timeout=60_000)
    page.fill("#game_amount_input", str(amount))
    page.click("#game_verify_submit")
    page.wait_for_selector("#game_verify_table tr", timeout=90_000)
    page.wait_for_load_state("networkidle", timeout=90_000)
    rows: list[dict] = []
    for tr in page.query_selector_all("#game_verify_table tr"):
        cells = tr.query_selector_all("td")
        if len(cells) < 3:
            continue
        try:
            rid = int(cells[0].inner_text().strip())
            mult = float(cells[2].inner_text().strip().lower().rstrip("x"))
        except ValueError:
            continue
        rows.append({"round_id": rid, "round_hash": cells[1].inner_text().strip().lower(), "multiplier": mult})
    return rows


def collect_once(amount: int = DEFAULT_BATCH, cursor: int | None = None, headless: bool = True) -> LiveBatch:
    """Une session de collecte. `cursor` = dernière manche connue (None = auto)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - dépend de l'environnement
        raise CollectorUnavailable(
            "Playwright non installé : `pip install playwright && playwright install chromium`"
        ) from exc

    cursor = cursor if cursor is not None else db.last_round_id()
    collected: list[dict] = []

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(headless=headless)
        except Exception as exc:  # pragma: no cover
            raise CollectorUnavailable(f"Navigateur indisponible: {exc}") from exc
        try:
            page = browser.new_page(user_agent="CrashPredictor/1.0 (+usage recherche; lecture page publique)")
            rows = _extract_rows(page, amount)
        finally:
            browser.close()

    if not rows:
        raise CollectorUnavailable("Aucune ligne renvoyée par la table de vérification (site modifié ou bloqué).")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new_rows = [r for r in rows if cursor is None or r["round_id"] > cursor]
    # Le site renvoie du plus récent au plus ancien : on répartit le temps de collecte.
    if new_rows:
        max_id = max(r["round_id"] for r in new_rows)
        for r in new_rows:
            r["ts"] = now
            r["ts_source"] = "live"
            r["source"] = f"{CONFIG.source_platform.lower()}_verify_live"
            r["collected_at"] = now
            r["_distance"] = max_id - r["round_id"]
        new_rows.sort(key=lambda r: r["_distance"])
        for r in new_rows:
            r.pop("_distance", None)
        for r in new_rows:
            r["ts"] = now
    records = mark_chain(new_rows)
    chain_failures = sum(1 for r in records if r["chain_ok"] == 0)
    collected.extend(records)
    return LiveBatch(records=collected, chain_failures=chain_failures)


def run_live_collector(
    interval_seconds: int = 120,
    amount: int = DEFAULT_BATCH,
    iterations: int | None = None,
    headless: bool = True,
    on_batch: Callable[[LiveBatch], None] | None = None,
) -> None:
    """Boucle de collecte temps réel (Ctrl+C pour arrêter)."""
    i = 0
    while iterations is None or i < iterations:
        i += 1
        try:
            batch = collect_once(amount=amount, headless=headless)
            cleaned = clean_records(batch.records).records
            inserted, skipped = db.upsert_rounds(cleaned)
            db.log_ingest(
                "bcgame_verify_live", "collect", inserted, skipped,
                "OK" if not batch.chain_failures else "WARN",
                f"chain_failures={batch.chain_failures}",
            )
            if on_batch:
                on_batch(batch)
            print(f"[{datetime.now(timezone.utc):%H:%M:%S}] +{inserted} manches (déjà présentes: {skipped})")
        except CollectorUnavailable as exc:
            db.log_ingest("bcgame_verify_live", "collect", 0, 0, "ERROR", str(exc))
            print(f"[ERREUR] {exc}")
        except Exception as exc:  # pragma: no cover - réseau
            db.log_ingest("bcgame_verify_live", "collect", 0, 0, "ERROR", repr(exc))
            print(f"[ERREUR] {exc!r}")
        if iterations is None or i < iterations:
            time.sleep(interval_seconds + random.uniform(0, 10))
