"""Couche de stockage : schéma SQLite, migrations légères, requêtes.

Aucune logique métier ici (pas de ML, pas de feature) : uniquement la
persistance (manches, prédictions, modèles, backtests, métriques, journal).
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from . import config as _config
from .config import CONFIG

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS rounds (
    round_id         INTEGER PRIMARY KEY,
    multiplier       REAL    NOT NULL,
    round_hash       TEXT,
    ts               TEXT,                        -- ISO8601 UTC
    ts_source        TEXT    NOT NULL DEFAULT 'unknown',  -- live | estimated | unknown
    source           TEXT    NOT NULL,
    collected_at     TEXT    NOT NULL,
    chain_ok         INTEGER,                     -- 1/0/NULL (non testable)
    CHECK (multiplier >= 1.0)
);
CREATE INDEX IF NOT EXISTS idx_rounds_ts ON rounds(ts);

CREATE TABLE IF NOT EXISTS predictions (
    id               TEXT PRIMARY KEY,
    created_at       TEXT NOT NULL,
    model_name       TEXT NOT NULL,
    model_version    TEXT,
    strategy         TEXT,                        -- prudente | equilibree | ambitieuse | tres_ambitieuse | optimisee
    target           REAL NOT NULL,
    probability      REAL,                        -- probabilité estimée de validation
    status           TEXT NOT NULL DEFAULT 'PENDING',   -- PENDING | SUCCESS | FAILURE
    anchor_round_id  INTEGER,                     -- dernière manche connue au moment de la prédiction
    n_rounds_seen    INTEGER,                     -- nombre de manches dans l'historique utilisé
    window_round_ids TEXT,                        -- JSON [id1, id2, id3] des manches de validation
    window_values    TEXT,                        -- JSON [m1, m2, m3]
    actual_max       REAL,
    validated_at     TEXT,
    explanation      TEXT,                        -- JSON importances / justification
    backtest_run_id  TEXT,
    is_backtest      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_pred_status ON predictions(status);
CREATE INDEX IF NOT EXISTS idx_pred_run ON predictions(backtest_run_id);

CREATE TABLE IF NOT EXISTS models (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    kind             TEXT NOT NULL,
    target           REAL,
    params           TEXT,
    metrics          TEXT,
    artifact_path    TEXT,
    train_start      INTEGER,
    train_end        INTEGER,
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    id               TEXT PRIMARY KEY,
    created_at       TEXT NOT NULL,
    finished_at      TEXT,
    status           TEXT NOT NULL,
    config           TEXT,
    summary          TEXT,
    data_fingerprint TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL,
    model_name       TEXT NOT NULL,
    target           REAL,
    metric           TEXT NOT NULL,
    value            REAL,
    ci_low           REAL,
    ci_high          REAL,
    n                INTEGER,
    segment          TEXT DEFAULT 'all'
);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);

CREATE TABLE IF NOT EXISTS ingest_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ts               TEXT NOT NULL,
    source           TEXT NOT NULL,
    action           TEXT NOT NULL,
    rows_inserted    INTEGER DEFAULT 0,
    rows_skipped     INTEGER DEFAULT 0,
    status           TEXT NOT NULL,
    message          TEXT
);

CREATE TABLE IF NOT EXISTS app_config (
    key              TEXT PRIMARY KEY,
    value            TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str = "") -> str:
    raw = uuid.uuid4().hex[:16].upper()
    return f"{prefix}{raw}" if prefix else raw


def _default_db_path() -> Path:
    """Chemin résolu à l'appel (permet d'isoler les tests sans variable globale)."""
    return Path(_config.DB_PATH)


@contextmanager
def connect(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path) if db_path else _default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | str | None = None) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.execute(
            "INSERT OR IGNORE INTO app_config(key, value, updated_at) VALUES (?,?,?)",
            ("schema_version", str(SCHEMA_VERSION), utcnow()),
        )


# --------------------------------------------------------------------------- #
# Manches
# --------------------------------------------------------------------------- #
def upsert_rounds(rows: Iterable[dict[str, Any]], db_path: Path | str | None = None) -> tuple[int, int]:
    """Insertion idempotente. Returns (insérés, ignorés car déjà présents)."""
    inserted = skipped = 0
    with connect(db_path) as conn:
        for r in rows:
            cur = conn.execute(
                """INSERT OR IGNORE INTO rounds
                   (round_id, multiplier, round_hash, ts, ts_source, source, collected_at, chain_ok)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    int(r["round_id"]),
                    float(r["multiplier"]),
                    r.get("round_hash"),
                    r.get("ts"),
                    r.get("ts_source", "unknown"),
                    r.get("source", "unknown"),
                    r.get("collected_at") or utcnow(),
                    r.get("chain_ok"),
                ),
            )
            if cur.rowcount:
                inserted += 1
            else:
                skipped += 1
    return inserted, skipped


def log_ingest(source: str, action: str, inserted: int, skipped: int, status: str, message: str = "") -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO ingest_log(ts, source, action, rows_inserted, rows_skipped, status, message) VALUES (?,?,?,?,?,?,?)",
            (utcnow(), source, action, inserted, skipped, status, message),
        )


def get_rounds(limit: int | None = None, ascending: bool = True, db_path: Path | str | None = None) -> list[sqlite3.Row]:
    order = "ASC" if ascending else "DESC"
    sql = f"SELECT * FROM rounds ORDER BY round_id {order}"
    params: Sequence[Any] = ()
    if limit:
        sql += " LIMIT ?"
        params = (limit,)
    with connect(db_path) as conn:
        return conn.execute(sql, params).fetchall()


def get_rounds_after(round_id: int, limit: int = 1000, db_path: Path | str | None = None) -> list[sqlite3.Row]:
    with connect(db_path) as conn:
        return conn.execute(
            "SELECT * FROM rounds WHERE round_id > ? ORDER BY round_id ASC LIMIT ?", (round_id, limit)
        ).fetchall()


def last_round_id(db_path: Path | str | None = None) -> int | None:
    with connect(db_path) as conn:
        row = conn.execute("SELECT MAX(round_id) AS m FROM rounds").fetchone()
        return row["m"] if row and row["m"] is not None else None


def first_round_id(db_path: Path | str | None = None) -> int | None:
    with connect(db_path) as conn:
        row = conn.execute("SELECT MIN(round_id) AS m FROM rounds").fetchone()
        return row["m"] if row and row["m"] is not None else None


def count_rounds(db_path: Path | str | None = None) -> int:
    with connect(db_path) as conn:
        return int(conn.execute("SELECT COUNT(*) AS c FROM rounds").fetchone()["c"])


def data_fingerprint(db_path: Path | str | None = None) -> dict[str, Any]:
    with connect(db_path) as conn:
        row = conn.execute(
            """SELECT COUNT(*) n, MIN(round_id) lo, MAX(round_id) hi,
                      SUM(COALESCE(chain_ok,0)) chain_ok, SUM(CASE WHEN chain_ok=0 THEN 1 ELSE 0 END) chain_bad
               FROM rounds"""
        ).fetchone()
        dup = conn.execute("SELECT COUNT(*) c FROM (SELECT round_id FROM rounds GROUP BY round_id HAVING COUNT(*)>1)").fetchone()
        return {
            "n": row["n"],
            "first": row["lo"],
            "last": row["hi"],
            "gaps": (row["hi"] - row["lo"] + 1 - row["n"]) if row["n"] else None,
            "chain_ok": row["chain_ok"],
            "chain_bad": row["chain_bad"],
            "duplicates": dup["c"],
        }


# --------------------------------------------------------------------------- #
# Prédictions
# --------------------------------------------------------------------------- #
PREDICTION_FIELDS = (
    "id", "created_at", "model_name", "model_version", "strategy", "target", "probability",
    "status", "anchor_round_id", "n_rounds_seen", "window_round_ids", "window_values",
    "actual_max", "validated_at", "explanation", "backtest_run_id", "is_backtest",
)


def insert_prediction(pred: dict[str, Any], db_path: Path | str | None = None) -> str:
    pred = dict(pred)
    pred.setdefault("id", new_id("P"))
    pred.setdefault("created_at", utcnow())
    pred.setdefault("status", "PENDING")
    pred.setdefault("is_backtest", 0)
    for key in ("window_round_ids", "window_values", "explanation"):
        if isinstance(pred.get(key), (list, dict)):
            pred[key] = json.dumps(pred[key])
    cols = ", ".join(PREDICTION_FIELDS)
    marks = ", ".join("?" for _ in PREDICTION_FIELDS)
    with connect(db_path) as conn:
        conn.execute(
            f"INSERT INTO predictions ({cols}) VALUES ({marks})",
            tuple(pred.get(f) for f in PREDICTION_FIELDS),
        )
    return pred["id"]


def update_prediction(pred_id: str, **fields: Any) -> None:
    for key in ("window_round_ids", "window_values", "explanation"):
        if isinstance(fields.get(key), (list, dict)):
            fields[key] = json.dumps(fields[key])
    sets = ", ".join(f"{k}=?" for k in fields)
    with connect() as conn:
        conn.execute(f"UPDATE predictions SET {sets} WHERE id=?", (*fields.values(), pred_id))


def pending_predictions(db_path: Path | str | None = None) -> list[sqlite3.Row]:
    with connect(db_path) as conn:
        return conn.execute(
            "SELECT * FROM predictions WHERE status='PENDING' AND is_backtest=0 ORDER BY created_at ASC"
        ).fetchall()


def predictions(limit: int = 200, backtest: bool = False, db_path: Path | str | None = None) -> list[sqlite3.Row]:
    with connect(db_path) as conn:
        return conn.execute(
            "SELECT * FROM predictions WHERE is_backtest=? ORDER BY created_at DESC, id DESC LIMIT ?",
            (1 if backtest else 0, limit),
        ).fetchall()


def get_prediction(pred_id: str) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM predictions WHERE id=?", (pred_id,)).fetchone()


# --------------------------------------------------------------------------- #
# Modèles / backtests / métriques / config
# --------------------------------------------------------------------------- #
def save_model_record(record: dict[str, Any]) -> str:
    rec = dict(record)
    rec.setdefault("id", new_id("M"))
    rec.setdefault("created_at", utcnow())
    for key in ("params", "metrics"):
        if isinstance(rec.get(key), (list, dict)):
            rec[key] = json.dumps(rec[key])
    cols = ("id", "name", "kind", "target", "params", "metrics", "artifact_path", "train_start", "train_end", "created_at")
    with connect() as conn:
        conn.execute(
            f"INSERT INTO models ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
            tuple(rec.get(c) for c in cols),
        )
    return rec["id"]


def list_models() -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute("SELECT * FROM models ORDER BY created_at DESC").fetchall()


def create_run(config: dict[str, Any], fingerprint: dict[str, Any]) -> str:
    run_id = new_id("R")
    with connect() as conn:
        conn.execute(
            "INSERT INTO backtest_runs(id, created_at, status, config, data_fingerprint) VALUES (?,?,?,?,?)",
            (run_id, utcnow(), "RUNNING", json.dumps(config), json.dumps(fingerprint)),
        )
    return run_id


def finish_run(run_id: str, status: str, summary: dict[str, Any]) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE backtest_runs SET status=?, summary=?, finished_at=? WHERE id=?",
            (status, json.dumps(summary), utcnow(), run_id),
        )


def add_metrics(run_id: str, rows: Iterable[dict[str, Any]]) -> None:
    with connect() as conn:
        conn.executemany(
            """INSERT INTO metrics(run_id, model_name, target, metric, value, ci_low, ci_high, n, segment)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            [
                (run_id, r["model_name"], r.get("target"), r["metric"], r.get("value"),
                 r.get("ci_low"), r.get("ci_high"), r.get("n"), r.get("segment", "all"))
                for r in rows
            ],
        )


def list_runs(limit: int = 20) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute("SELECT * FROM backtest_runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()


def latest_finished_run() -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM backtest_runs WHERE status='DONE' ORDER BY finished_at DESC LIMIT 1"
        ).fetchone()


def run_metrics(run_id: str) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute("SELECT * FROM metrics WHERE run_id=? ORDER BY model_name, target, metric", (run_id,)).fetchall()


def set_app_config(key: str, value: Any) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO app_config(key, value, updated_at) VALUES (?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, json.dumps(value), utcnow()),
        )


def get_app_config(key: str, default: Any = None) -> Any:
    with connect() as conn:
        row = conn.execute("SELECT value FROM app_config WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def ingest_history(limit: int = 50) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute("SELECT * FROM ingest_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
