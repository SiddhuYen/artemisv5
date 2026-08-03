"""Persistent relation cache — grounded relationships kept across jobs.

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
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from artemis.identity.normalize import name_key
from artemis.models import Extraction, ResolutionBasis, iso_z, utcnow


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

    @property
    def fingerprint(self) -> tuple:
        return (
            self.subject_name.casefold(),
            self.object_name.casefold(),
            self.source_url,
            self.span_start,
            self.span_end,
        )

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


class RelationCache:
    def __init__(self, root: Path, *, enabled: bool = True, ttl_s: int = 30 * 86400) -> None:
        self.root = Path(root) / "relations"
        self.enabled = enabled
        self.ttl_s = ttl_s
        self.written = 0
        self.replayed = 0

    def _path(self, name: str) -> Path:
        key = name_key(name) or "unknown"
        digest = "".join(c if c.isalnum() else "_" for c in key)[:64]
        return self.root / digest[:2] / f"{digest}.json"

    # -- reading ------------------------------------------------------------
    def _read(self, path: Path) -> list[dict]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

    async def lookup(self, name: str) -> list[CachedRelation]:
        """Everything previously grounded about someone with this name key."""
        if not self.enabled:
            return []
        rows = await asyncio.to_thread(self._read, self._path(name))
        out: list[CachedRelation] = []
        cutoff = utcnow().timestamp() - self.ttl_s
        for row in rows:
            try:
                relation = CachedRelation(**{**row,
                                             "subject_attributes": tuple(row["subject_attributes"]),
                                             "object_attributes": tuple(row["object_attributes"])})
            except (TypeError, KeyError):
                continue
            try:
                if datetime.fromisoformat(relation.cached_at.replace("Z", "+00:00")).timestamp() < cutoff:
                    continue
            except ValueError:
                continue
            out.append(relation)
        return out

    # -- writing ------------------------------------------------------------
    def _append(self, path: Path, row: dict, fingerprint: tuple) -> bool:
        rows = self._read(path)
        for existing in rows:
            if (
                existing.get("subject_name", "").casefold(),
                existing.get("object_name", "").casefold(),
                existing.get("source_url"),
                existing.get("span_start"),
                existing.get("span_end"),
            ) == fingerprint:
                return False
        rows.append(row)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            return False
        return True

    async def add(
        self,
        extraction: Extraction,
        *,
        source_url: str,
        source_title: Optional[str],
        retrieved_at: datetime,
    ) -> None:
        """Record one grounded relation under both parties' name keys."""
        if not self.enabled:
            return
        relation = CachedRelation(
            subject_name=extraction.subject_name,
            object_name=extraction.object_name,
            span_text=extraction.span_text,
            span_start=extraction.span_start,
            span_end=extraction.span_end,
            context_before=extraction.context_before,
            resolved_statement=extraction.resolved_statement,
            resolution_basis=extraction.resolution_basis.value,
            subject_attributes=tuple(extraction.subject_attributes),
            object_attributes=tuple(extraction.object_attributes),
            source_url=source_url,
            source_title=source_title,
            retrieved_at=iso_z(retrieved_at),
            cached_at=iso_z(utcnow()),
        )
        row = asdict(relation)
        row["subject_attributes"] = list(relation.subject_attributes)
        row["object_attributes"] = list(relation.object_attributes)

        wrote = False
        for name in {name_key(extraction.subject_name), name_key(extraction.object_name)}:
            if not name:
                continue
            path = self._path(
                extraction.subject_name
                if name == name_key(extraction.subject_name)
                else extraction.object_name
            )
            wrote |= await asyncio.to_thread(self._append, path, row, relation.fingerprint)
        if wrote:
            self.written += 1
