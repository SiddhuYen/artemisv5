"""Persistent relation store — grounded relationships kept across jobs.

Every relationship this service grounds, asserted or co-listed, is written here
with its full evidence: the verbatim span, the offsets it was found at, the URL,
and when that page was fetched. A later job researching the same person replays
them straight into its graph without spending a search, a fetch, or a model call.

Two properties matter:

* **Evidence travels with the relation.** A replayed hop cites the same span and
  the same URL as the day it was captured, and `retrieved_at` is the original
  fetch time, not the replay time. Nothing is presented as fresher than it is.
* **Identity does not travel.** Relations are stored under a coarse name key and
  replayed through the ordinary merge ladder in the new job's graph. A cached
  "Jane Smith" relation never silently joins a different Jane Smith — the
  receiving job re-decides that on its own evidence.

SQLite rather than a file per name: writes are atomic (the previous
read-modify-write on a JSON file could lose a relation when two coroutines
touched the same name), dedup is a UNIQUE constraint instead of a linear scan,
and the store can actually be queried — "what do we know about X", "what
connects X and Y" — without loading every shard.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from artemis.identity.normalize import name_key
from artemis.models import Extraction, ResolutionBasis, iso_z, utcnow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS relations (
    id                 INTEGER PRIMARY KEY,
    subject_key        TEXT NOT NULL,
    object_key         TEXT NOT NULL,
    subject_name       TEXT NOT NULL,
    object_name        TEXT NOT NULL,
    span_text          TEXT NOT NULL,
    span_start         INTEGER NOT NULL,
    span_end           INTEGER NOT NULL,
    context_before     TEXT NOT NULL DEFAULT '',
    resolved_statement TEXT NOT NULL DEFAULT '',
    resolution_basis   TEXT NOT NULL,
    subject_attributes TEXT NOT NULL DEFAULT '[]',
    object_attributes  TEXT NOT NULL DEFAULT '[]',
    source_url         TEXT NOT NULL,
    source_title       TEXT,
    retrieved_at       TEXT NOT NULL,
    cached_at          TEXT NOT NULL,
    UNIQUE (subject_name, object_name, source_url, span_start, span_end)
);
CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations (subject_key);
CREATE INDEX IF NOT EXISTS idx_relations_object  ON relations (object_key);
CREATE INDEX IF NOT EXISTS idx_relations_cached  ON relations (cached_at);
"""


@dataclass(frozen=True)
class CachedRelation:
    subject_name: str
    object_name: str
    span_text: str
    span_start: int
    span_end: int
    context_before: str
    resolved_statement: str
    resolution_basis: str
    subject_attributes: tuple[str, ...]
    object_attributes: tuple[str, ...]
    source_url: str
    source_title: Optional[str]
    retrieved_at: str
    cached_at: str

    def to_extraction(self) -> Optional[Extraction]:
        try:
            return Extraction(
                subject_name=self.subject_name,
                object_name=self.object_name,
                span_text=self.span_text,
                span_start=self.span_start,
                span_end=self.span_end,
                context_before=self.context_before,
                resolved_statement=self.resolved_statement,
                resolution_basis=ResolutionBasis(self.resolution_basis),
                subject_attributes=list(self.subject_attributes),
                object_attributes=list(self.object_attributes),
            )
        except (ValueError, KeyError):
            return None


def _row_to_relation(row: sqlite3.Row) -> CachedRelation:
    return CachedRelation(
        subject_name=row["subject_name"],
        object_name=row["object_name"],
        span_text=row["span_text"],
        span_start=row["span_start"],
        span_end=row["span_end"],
        context_before=row["context_before"],
        resolved_statement=row["resolved_statement"],
        resolution_basis=row["resolution_basis"],
        subject_attributes=tuple(json.loads(row["subject_attributes"])),
        object_attributes=tuple(json.loads(row["object_attributes"])),
        source_url=row["source_url"],
        source_title=row["source_title"],
        retrieved_at=row["retrieved_at"],
        cached_at=row["cached_at"],
    )


class RelationCache:
    def __init__(self, root: Path, *, enabled: bool = True, ttl_s: int = 30 * 86400) -> None:
        self.root = Path(root)
        self.path = self.root / "relations.db"
        self.enabled = enabled
        self.ttl_s = ttl_s
        self.written = 0
        self.replayed = 0
        self._ready = False

    # -- connection ---------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        self.root.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # WAL lets readers and a writer coexist, which matters because the
        # crawl replays from this store while still writing to it.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init(self) -> None:
        if self._ready:
            return
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        self._migrate_json()
        self._ready = True

    def _migrate_json(self) -> None:
        """One-time import of the previous file-per-name layout."""
        legacy = self.root / "relations"
        if not legacy.is_dir():
            return
        rows: list[dict] = []
        for path in legacy.rglob("*.json"):
            try:
                rows.extend(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        if rows:
            with self._connect() as conn:
                for row in rows:
                    self._insert(conn, row)
        try:
            legacy.rename(self.root / "relations.migrated")
        except OSError:
            pass

    # -- writing ------------------------------------------------------------
    @staticmethod
    def _insert(conn: sqlite3.Connection, row: dict) -> bool:
        try:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO relations (
                    subject_key, object_key, subject_name, object_name,
                    span_text, span_start, span_end, context_before,
                    resolved_statement, resolution_basis,
                    subject_attributes, object_attributes,
                    source_url, source_title, retrieved_at, cached_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    name_key(row["subject_name"]),
                    name_key(row["object_name"]),
                    row["subject_name"],
                    row["object_name"],
                    row["span_text"],
                    row["span_start"],
                    row["span_end"],
                    row.get("context_before", ""),
                    row.get("resolved_statement", ""),
                    row["resolution_basis"],
                    json.dumps(list(row.get("subject_attributes", []))),
                    json.dumps(list(row.get("object_attributes", []))),
                    row["source_url"],
                    row.get("source_title"),
                    row["retrieved_at"],
                    row.get("cached_at") or iso_z(utcnow()),
                ),
            )
            return cur.rowcount > 0
        except (sqlite3.Error, KeyError):
            return False

    def _add_sync(self, row: dict) -> bool:
        self._init()
        with self._connect() as conn:
            return self._insert(conn, row)

    async def add(
        self,
        extraction: Extraction,
        *,
        source_url: str,
        source_title: Optional[str],
        retrieved_at: datetime,
    ) -> None:
        """Record one grounded relation. Dedup is the UNIQUE constraint."""
        if not self.enabled:
            return
        row = {
            "subject_name": extraction.subject_name,
            "object_name": extraction.object_name,
            "span_text": extraction.span_text,
            "span_start": extraction.span_start,
            "span_end": extraction.span_end,
            "context_before": extraction.context_before,
            "resolved_statement": extraction.resolved_statement,
            "resolution_basis": extraction.resolution_basis.value,
            "subject_attributes": list(extraction.subject_attributes),
            "object_attributes": list(extraction.object_attributes),
            "source_url": source_url,
            "source_title": source_title,
            "retrieved_at": iso_z(retrieved_at),
            "cached_at": iso_z(utcnow()),
        }
        if await asyncio.to_thread(self._add_sync, row):
            self.written += 1

    # -- reading ------------------------------------------------------------
    def _lookup_sync(self, name: str) -> list[CachedRelation]:
        self._init()
        key = name_key(name)
        if not key:
            return []
        cutoff = iso_z(datetime.fromtimestamp(utcnow().timestamp() - self.ttl_s, tz=utcnow().tzinfo))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM relations
                 WHERE (subject_key = ? OR object_key = ?) AND cached_at >= ?
                """,
                (key, key, cutoff),
            ).fetchall()
        return [_row_to_relation(r) for r in rows]

    async def lookup(self, name: str) -> list[CachedRelation]:
        """Everything previously grounded about someone with this name key."""
        if not self.enabled:
            return []
        return await asyncio.to_thread(self._lookup_sync, name)

    # -- inspection ---------------------------------------------------------
    def _stats_sync(self) -> dict[str, int]:
        self._init()
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
            people = conn.execute(
                "SELECT COUNT(*) FROM (SELECT subject_key FROM relations "
                "UNION SELECT object_key FROM relations)"
            ).fetchone()[0]
            colisted = conn.execute(
                "SELECT COUNT(*) FROM relations WHERE resolution_basis = 'co_listing'"
            ).fetchone()[0]
        return {"relations": total, "people": people, "co_listings": colisted}

    async def stats(self) -> dict[str, int]:
        if not self.enabled:
            return {"relations": 0, "people": 0, "co_listings": 0}
        return await asyncio.to_thread(self._stats_sync)
