"""The operator's own network: a LinkedIn CSV roster, kept beside the caches.

This layer deliberately does **not** feed the graph. A row in a LinkedIn export
is not a page asserting a relationship, and the whole service rests on the rule
that an edge cannot exist without a verbatim span from a fetched page. Turning
"Priya Raman,Acme,Founder" into a synthesized sentence and extracting from it —
what ArtemisV2 did — would put an ungrounded edge in the same graph as grounded
ones, which is the one thing this build refuses to do.

So the roster earns its keep two honest ways instead:

* it is where ``person_a`` comes from — you pick a real person out of your own
  contacts rather than typing a name and hoping the crawler seeds it;
* at return time, route pivots are matched against it by name, so a route that
  happens to run through somebody you already know is *labelled* as such. That
  is a fact about your CSV, reported next to the route, never a hop — and when
  the name is ambiguous against the roster it says so rather than picking one.

The operator is who the CSV belongs to. Without a CSV there is nobody to be the
owner of, so the operator is never demanded; ``POST /network/upload`` is the one
call that requires it.
"""

from __future__ import annotations

import asyncio
import csv
import io
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from artemis.identity.normalize import (
    could_be_same_name,
    name_key,
    normalize_name,
    surname,
)
from artemis.models import iso_z, utcnow

#: Rows are keyed on the *full* normalised name, not ``name_key``. ``name_key``
#: is given-initial + surname by design — it decides who gets considered for
#: merging — and using it as a uniqueness key here would quietly fuse John Smith
#: and Jane Smith into one contact on import. It is stored alongside anyway,
#: indexed, because it is the right key for pulling homonym candidates out.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS contacts (
    id          INTEGER PRIMARY KEY,
    norm_name   TEXT NOT NULL UNIQUE,
    name_key    TEXT NOT NULL,
    name        TEXT NOT NULL,
    role        TEXT NOT NULL DEFAULT '',
    company     TEXT NOT NULL DEFAULT '',
    profile_url TEXT NOT NULL DEFAULT '',
    connected_on TEXT NOT NULL DEFAULT '',
    added_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_contacts_name ON contacts (name);
CREATE INDEX IF NOT EXISTS idx_contacts_key  ON contacts (name_key);

CREATE TABLE IF NOT EXISTS operator (
    id      INTEGER PRIMARY KEY CHECK (id = 1),
    name    TEXT NOT NULL,
    context TEXT NOT NULL DEFAULT '',
    set_at  TEXT NOT NULL
);
"""

#: LinkedIn has shipped these header spellings across export generations, and
#: localised exports use the same columns in another language often enough that
#: matching on position alone would be wrong. Matched case- and space-folded.
_FIRST = ("first name", "firstname", "given name", "vorname", "prénom", "prenom", "nome")
_LAST = ("last name", "lastname", "surname", "family name", "nachname", "nom", "cognome")
_ROLE = ("position", "title", "job title", "headline", "role")
_COMPANY = ("company", "organization", "organisation", "current company", "employer")
_URL = ("url", "profile url", "public profile url", "linkedin url", "profile link")
_CONNECTED = ("connected on", "connected_on", "date connected", "connection date")
#: A single-column export exists too ("Full Name"), and some CRMs re-export one.
_FULL = ("full name", "name", "contact name", "display name")


@dataclass(frozen=True)
class Contact:
    name: str
    role: str = ""
    company: str = ""
    profile_url: str = ""
    connected_on: str = ""
    added_at: Optional[datetime] = None

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "norm_name": normalize_name(self.name),
            "role": self.role,
            "company": self.company,
            "profile_url": self.profile_url,
            "connected_on": self.connected_on,
            "added_at": iso_z(self.added_at) if self.added_at else None,
        }


class CsvFormatError(ValueError):
    """The upload is not recognisably a LinkedIn Connections export."""


def _fold_header(value: str) -> str:
    return value.strip().strip('"').replace("_", " ").casefold()


def _pick(headers: list[str], candidates: Iterable[str]) -> Optional[int]:
    folded = [_fold_header(h) for h in headers]
    for want in candidates:
        if want in folded:
            return folded.index(want)
    return None


def parse_linkedin_csv(raw: bytes | str) -> list[Contact]:
    """Rows out of a LinkedIn Connections export.

    LinkedIn prepends a three-line "Notes:" preamble before the real header, and
    which line it lands on has changed between export generations — so the header
    is *found* by looking for the name columns rather than assumed at row 0 or
    row 3. Everything after it is parsed by ``csv`` proper, not split on commas:
    roles routinely contain commas inside quotes ("VP, Engineering") and a naive
    split silently shifts every later column on exactly those rows.
    """
    if isinstance(raw, bytes):
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:  # pragma: no cover - latin-1 decodes any byte string
            raise CsvFormatError("could not decode the file as text")
    else:
        text = raw

    if not text.strip():
        raise CsvFormatError("the file is empty")

    rows = list(csv.reader(io.StringIO(text)))
    header_idx = -1
    for i, row in enumerate(rows[:20]):
        if not row:
            continue
        folded = [_fold_header(c) for c in row]
        if any(f in folded for f in _FIRST) and any(l in folded for l in _LAST):
            header_idx = i
            break
        if any(f in folded for f in _FULL) and len(row) > 1:
            header_idx = i
            break
    if header_idx == -1:
        raise CsvFormatError(
            "no LinkedIn header row found in the first 20 lines — expected columns "
            "'First Name' and 'Last Name'. Export Connections only, and upload "
            "Connections.csv from the zip rather than the zip itself."
        )

    headers = rows[header_idx]
    i_first = _pick(headers, _FIRST)
    i_last = _pick(headers, _LAST)
    i_full = _pick(headers, _FULL)
    i_role = _pick(headers, _ROLE)
    i_company = _pick(headers, _COMPANY)
    i_url = _pick(headers, _URL)
    i_conn = _pick(headers, _CONNECTED)

    def cell(row: list[str], idx: Optional[int]) -> str:
        if idx is None or idx >= len(row):
            return ""
        return row[idx].strip()

    seen: set[str] = set()
    contacts: list[Contact] = []
    for row in rows[header_idx + 1 :]:
        if not row or not any(c.strip() for c in row):
            continue
        if i_first is not None or i_last is not None:
            name = f"{cell(row, i_first)} {cell(row, i_last)}".strip()
        else:
            name = cell(row, i_full)
        if not name:
            continue
        key = normalize_name(name)
        # A LinkedIn export can carry the same person twice after a merge; the
        # DB would reject the second insert anyway, so drop it here where the
        # count shown to the user is computed.
        if not key or key in seen:
            continue
        seen.add(key)
        contacts.append(
            Contact(
                name=name,
                role=cell(row, i_role),
                company=cell(row, i_company),
                profile_url=cell(row, i_url),
                connected_on=cell(row, i_conn),
            )
        )
    if not contacts:
        raise CsvFormatError("the header was found but no rows carried a name")
    return contacts


class NetworkStore:
    """SQLite roster. Same directory and threading discipline as RelationCache.

    Writes go through a lock and run in a worker thread: the routes that call
    this are async, and a CSV of 5,000 rows is long enough that doing the insert
    on the event loop would stall every in-flight crawl.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.path = Path(cache_dir) / "network.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # -- operator -----------------------------------------------------------
    def get_operator(self) -> Optional[dict[str, str]]:
        with self._connect() as conn:
            row = conn.execute("SELECT name, context, set_at FROM operator WHERE id = 1").fetchone()
        if row is None:
            return None
        return {"name": row["name"], "context": row["context"], "set_at": row["set_at"]}

    async def set_operator(self, name: str, context: str = "") -> dict[str, str]:
        set_at = iso_z(utcnow())
        async with self._lock:
            await asyncio.to_thread(self._write_operator, name, context, set_at)
        return {"name": name, "context": context, "set_at": set_at}

    def _write_operator(self, name: str, context: str, set_at: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO operator (id, name, context, set_at) VALUES (1, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET name=excluded.name, "
                "context=excluded.context, set_at=excluded.set_at",
                (name, context, set_at),
            )

    # -- contacts -----------------------------------------------------------
    def contacts(self, search: str = "", limit: int = 0) -> list[Contact]:
        sql = "SELECT * FROM contacts"
        params: list[object] = []
        if search.strip():
            sql += " WHERE name LIKE ? OR company LIKE ? OR role LIKE ?"
            like = f"%{search.strip()}%"
            params += [like, like, like]
        sql += " ORDER BY name COLLATE NOCASE"
        if limit > 0:
            sql += " LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            Contact(
                name=r["name"],
                role=r["role"],
                company=r["company"],
                profile_url=r["profile_url"],
                connected_on=r["connected_on"],
                added_at=_parse_iso(r["added_at"]),
            )
            for r in rows
        ]

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM contacts").fetchone()[0])

    def match(self, names: Iterable[str]) -> dict[str, dict[str, str]]:
        """Which of these names are people in the roster.

        Two strengths, reported separately rather than blended, because they are
        not the same claim. ``exact`` is the same normalised name — titles and
        accents folded, nothing else. ``possible`` is ``could_be_same_name``,
        which resolves nicknames and initials and is explicitly documented as
        "could denote one person, not evidence that they do".

        Neither is an identity assertion, and neither reaches the graph. This
        answers "is a name on this route also a name in my CSV" — a fact about
        the CSV, surfaced beside a route, never inside one.
        """
        with self._connect() as conn:
            rows = conn.execute("SELECT name, norm_name, name_key, role, company FROM contacts")
            roster = [dict(r) for r in rows]
        by_norm = {r["norm_name"]: r for r in roster}
        # Bucketed by *surname*, not by name_key. name_key is given-initial +
        # surname, so bucketing on it would hide "Bob Vance" from "Robert
        # Vance" — the nickname case could_be_same_name exists to resolve.
        # Surname is the correct bucket and loses nothing: that function
        # already refuses any pair whose surnames differ.
        by_surname: dict[str, list[dict[str, str]]] = {}
        for r in roster:
            by_surname.setdefault(surname(r["name"]), []).append(r)

        out: dict[str, dict[str, str]] = {}
        for name in names:
            norm = normalize_name(name)
            if not norm:
                continue
            hit = by_norm.get(norm)
            if hit is not None:
                out[name] = {
                    "basis": "exact",
                    "name": hit["name"],
                    "role": hit["role"],
                    "company": hit["company"],
                }
                continue
            # Only contacts sharing the coarse key are worth the pairwise call;
            # a full scan per pivot name would be O(routes x roster).
            hits = [
                cand
                for cand in by_surname.get(surname(name), ())
                if could_be_same_name(name, cand["name"])
            ]
            if not hits:
                continue
            if len(hits) > 1:
                # "J. Smith" against a roster holding John and Jane Smith. Naming
                # one of them would be a coin flip presented as a lookup, so the
                # count goes out instead and the caller names nobody.
                out[name] = {
                    "basis": "ambiguous",
                    "name": "",
                    "role": "",
                    "company": "",
                    "candidates": str(len(hits)),
                }
                continue
            out[name] = {
                "basis": "possible",
                "name": hits[0]["name"],
                "role": hits[0]["role"],
                "company": hits[0]["company"],
            }
        return out

    async def add_many(self, contacts: list[Contact]) -> dict[str, int]:
        async with self._lock:
            return await asyncio.to_thread(self._write_contacts, contacts)

    def _write_contacts(self, contacts: list[Contact]) -> dict[str, int]:
        added_at = iso_z(utcnow())
        created = updated = skipped = 0
        with self._connect() as conn:
            for c in contacts:
                norm = normalize_name(c.name)
                if not norm:
                    skipped += 1
                    continue
                existing = conn.execute(
                    "SELECT role, company, profile_url FROM contacts WHERE norm_name = ?", (norm,)
                ).fetchone()
                if existing is None:
                    conn.execute(
                        "INSERT INTO contacts (norm_name, name_key, name, role, company, "
                        "profile_url, connected_on, added_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            norm,
                            name_key(c.name),
                            c.name,
                            c.role,
                            c.company,
                            c.profile_url,
                            c.connected_on,
                            added_at,
                        ),
                    )
                    created += 1
                    continue
                # A re-upload months later carries fresher titles. Only ever
                # fill in or replace with something non-empty, so a sparser
                # export cannot blank out detail an earlier one supplied.
                role = c.role or existing["role"]
                company = c.company or existing["company"]
                url = c.profile_url or existing["profile_url"]
                if (role, company, url) == (
                    existing["role"],
                    existing["company"],
                    existing["profile_url"],
                ):
                    skipped += 1
                    continue
                conn.execute(
                    "UPDATE contacts SET role = ?, company = ?, profile_url = ? WHERE norm_name = ?",
                    (role, company, url, norm),
                )
                updated += 1
        return {"created": created, "updated": updated, "skipped": skipped}

    async def clear(self) -> int:
        async with self._lock:
            return await asyncio.to_thread(self._clear)

    def _clear(self) -> int:
        with self._connect() as conn:
            n = int(conn.execute("SELECT COUNT(*) FROM contacts").fetchone()[0])
            conn.execute("DELETE FROM contacts")
        return n


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
