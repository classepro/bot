"""Intégrité des données : chaîne SHA256, dédoublonnage, rejets."""
from __future__ import annotations

import hashlib

from crashpredict.collection.cleaner import clean_records, mark_chain, verify_chain


def _chain(n: int) -> list[dict]:
    """Chaîne provably-fair réelle : SHA256(hash[i+1]) == hash[i]."""
    hashes = [hashlib.sha256(b"seed").hexdigest()]
    for _ in range(n - 1):
        hashes.append(hashlib.sha256(hashes[-1].encode()).hexdigest())
    hashes = hashes[::-1]  # hash[dernier] = seed, hash[i] = sha256(hash[i+1])
    return [{"round_id": i + 1, "round_hash": hashes[i], "multiplier": 1.5 + i * 0.1} for i in range(n)]


def test_chain_detects_order_and_hash():
    recs = mark_chain(_chain(20))
    tested, bad = verify_chain(recs)
    assert tested == 19 and bad == []


def test_chain_detects_tampering():
    recs = _chain(20)
    recs[10]["round_hash"] = "0" * 64
    marked = mark_chain(recs)
    tested, bad = verify_chain(marked)
    assert tested == 19
    assert len(bad) == 2  # la ligne modifiée et la suivante


def test_duplicates_and_invalid_values_rejected():
    recs = [
        {"round_id": 1, "multiplier": 2.0},
        {"round_id": 1, "multiplier": 2.0},
        {"round_id": 2, "multiplier": 0.5},
        {"round_id": 3, "multiplier": "abc"},
        {"round_id": 4, "multiplier": 1.01},
    ]
    out = clean_records(recs)
    assert [r["round_id"] for r in out.records] == [1, 4]
    assert out.n_rejected == 3


def test_records_sorted_chronologically():
    recs = [{"round_id": 5, "multiplier": 2.0}, {"round_id": 2, "multiplier": 1.2}]
    out = clean_records(recs)
    assert [r["round_id"] for r in out.records] == [2, 5]
