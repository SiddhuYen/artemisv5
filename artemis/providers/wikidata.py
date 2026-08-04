"""Wikidata — structured claims about a person, used to find pages worth reading.

Free, no key, and the highest-precision person index that exists: a claim like
"P108 employer → Q95" is curated, not inferred from prose. ArtemisV2 turned
those claims straight into edges by writing a sentence for each one; this does
not, for the reason in this package's docstring — a sentence we wrote ourselves
is not a page asserting a relationship.

This provider does both jobs. `discover()` follows the pointing: if Wikidata
says this person co-founded that company or married that person, those entities
have articles, and those articles are prose written by someone else stating the
relationship in their own words — so a claim buys a URL and the ordinary fetch
-> extract -> ground path does the rest. That remains the preferred evidence.

`assert_relations()` admits the claim itself when no such page turns up. Not the
same as v2's synthesized sentence: nothing is invented here, the claim is a
curated and cited record, and it arrives with QIDs for both ends where the rest
of this system has only names. Those hops are labelled STRUCTURED_CLAIM and a
route resting on one ranks below an equal-length route built from prose.

Two guards carried over from v2 unchanged, each earned:

  * Educated-at (P69) and political party (P102) are NOT followed. Sharing a
    university or a party is not a relationship, and treating it as one produced
    mass false "classmate" and "colleague" edges.
  * The subject must be a human (P31 → Q5). Name search on Wikidata returns
    ships, songs and racehorses for ordinary human names.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Sequence

import httpx

from artemis.providers import Assertion, Discovery

_API = "https://www.wikidata.org/w/api.php"
_WIKIPEDIA = "https://en.wikipedia.org/wiki/{title}"
_ENTITY = "https://www.wikidata.org/wiki/{qid}"

#: Wikidata property -> how the target relates to the subject. Phrased for the
#: job log, so an operator can see why a URL was fetched.
#:
#: P69 (educated at) and P102 (member of political party) are deliberately
#: absent: co-attendance and co-membership at that scale are not relationships,
#: and following them floods the frontier with strangers.
#: Every entry here points at a person or an organisation. Properties that name
#: a *role* rather than an entity are deliberately absent — P39 (position held),
#: P1308 (officeholder), and on a person P488/P169/P112 resolve to articles like
#: "Chief technology officer" and "Chair (officer)", which are encyclopedia
#: entries about job titles and can never state that two people know each other.
#: Founder, CEO and chair are recovered in the right direction by _reverse().
_PROPERTIES: dict[str, str] = {
    "P26": "spouse of",
    "P22": "father of",
    "P25": "mother of",
    "P40": "parent of",
    "P3373": "sibling of",
    "P108": "employed by",
    "P1066": "student of",
    "P802": "taught",
    "P463": "member of",
}

_SPARQL = "https://query.wikidata.org/sparql"
#: Organisations the subject founded, leads, or chairs. These are stored on the
#: *organisation* (an org has a founder; a person does not have a "founded"),
#: so reading them forward off the person yields job titles instead.
_REVERSE_QUERY = """
SELECT ?org ?orgLabel ?rel ?article WHERE {
  VALUES (?p ?rel) {
    (wdt:P112 "founded") (wdt:P169 "is chief executive of")
    (wdt:P488 "chairs") (wdt:P3320 "sits on the board of")
  }
  ?org ?p wd:%s .
  OPTIONAL {
    ?article schema:about ?org ; schema:isPartOf <https://en.wikipedia.org/> .
  }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 25
"""

_HUMAN = "Q5"
_QID_ONLY = re.compile(r"^Q\d+$")
#: wbgetentities accepts 50 ids per call; one round trip covers a whole person.
_BATCH = 50
#: A person with 200 claimed positions is an institution, not a lead worth
#: chasing. Cap what one person contributes to the frontier.
_MAX_TARGETS = 24


class WikidataProvider:
    name = "wikidata"

    def __init__(self, settings: Any) -> None:
        self.s = settings

    def available(self) -> bool:
        return bool(getattr(self.s, "wikidata_enabled", True))

    async def discover(self, *, person: str, orgs: Sequence[str]) -> list[Discovery]:
        headers = {"User-Agent": self.s.user_agent}
        async with httpx.AsyncClient(
            timeout=20.0, headers=headers, follow_redirects=True
        ) as client:
            qid = await self._resolve_person(client, person, orgs)
            if qid is None:
                return []
            entity = await self._entity(client, qid)
            if entity is None:
                return []

            out: list[Discovery] = []
            own = _sitelink(entity)
            if own:
                # The subject's own article is the densest single page about
                # them, and it states their relationships in someone else's
                # prose rather than in a claim.
                out.append(
                    Discovery(
                        url=_WIKIPEDIA.format(title=own),
                        provider=self.name,
                        why=f"Wikipedia article for {person} ({qid})",
                    )
                )

            await asyncio.sleep(0.1)  # Wikimedia asks for polite pacing
            out.extend(await self._reverse(client, qid, person))

            targets = _targets(entity)
            if not targets:
                return out
            await asyncio.sleep(0.1)
            labels = await self._labels(client, [q for q, _ in targets])

        for target_qid, relation in targets:
            title, label = labels.get(target_qid, ("", ""))
            if not title:
                continue  # no English article: nothing to fetch
            out.append(
                Discovery(
                    url=_WIKIPEDIA.format(title=title),
                    provider=self.name,
                    why=f"{person} — {relation} — {label or title}",
                )
            )
        return out

    async def assert_relations(
        self, *, person: str, orgs: Sequence[str]
    ) -> list[Assertion]:
        """The same claims discover() follows, admitted as relationships.

        A claim states the relationship outright and carries QIDs for both ends,
        so it does not need a page to corroborate it. discover() still runs: the
        article is worth reading anyway, and a hop grounded in prose outranks one
        grounded here.
        """
        headers = {"User-Agent": self.s.user_agent}
        async with httpx.AsyncClient(
            timeout=20.0, headers=headers, follow_redirects=True
        ) as client:
            qid = await self._resolve_person(client, person, orgs)
            if qid is None:
                return []
            entity = await self._entity(client, qid)
            if entity is None:
                return []
            subject = _label(entity) or person

            targets = _targets(entity)
            await asyncio.sleep(0.1)
            labels = await self._labels(client, [q for q, _ in targets])
            out = [
                Assertion(
                    subject=subject,
                    object=label,
                    relation=relation,
                    source_url=_ENTITY.format(qid=qid),
                    source_title=f"Wikidata: {subject} ({qid})",
                    provider=self.name,
                    subject_id=qid,
                    object_id=target_qid,
                )
                for target_qid, relation in targets
                for _title, label in [labels.get(target_qid, ("", ""))]
                if label and label.casefold() != subject.casefold()
            ]

            await asyncio.sleep(0.1)
            out.extend(await self._reverse_claims(client, qid, subject))
        return out

    async def _reverse_claims(
        self, client: httpx.AsyncClient, qid: str, subject: str
    ) -> list[Assertion]:
        try:
            resp = await client.get(
                _SPARQL,
                params={"query": _REVERSE_QUERY % qid, "format": "json"},
                headers={"Accept": "application/sparql-results+json"},
            )
            if resp.status_code != 200:
                return []
            rows = (resp.json().get("results") or {}).get("bindings", []) or []
        except Exception:
            return []

        out: list[Assertion] = []
        seen: set[str] = set()
        for row in rows:
            org_uri = ((row.get("org") or {}).get("value") or "").strip()
            org_qid = org_uri.rsplit("/", 1)[-1] if org_uri else ""
            label = ((row.get("orgLabel") or {}).get("value") or "").strip()
            if _QID_ONLY.match(label):
                article = ((row.get("article") or {}).get("value") or "").strip()
                label = article.rsplit("/", 1)[-1].replace("_", " ") if article else ""
            if not label or label in seen or label.casefold() == subject.casefold():
                continue
            seen.add(label)
            out.append(
                Assertion(
                    subject=subject,
                    object=label,
                    relation=((row.get("rel") or {}).get("value") or "is involved with").strip(),
                    source_url=_ENTITY.format(qid=org_qid or qid),
                    source_title=f"Wikidata: {label} ({org_qid})",
                    provider=self.name,
                    subject_id=qid,
                    object_id=org_qid,
                )
            )
        return out

    # -- lookups ------------------------------------------------------------
    async def _resolve_person(
        self, client: httpx.AsyncClient, person: str, orgs: Sequence[str]
    ) -> str | None:
        try:
            resp = await client.get(
                _API,
                params={
                    "action": "wbsearchentities",
                    "search": person,
                    "language": "en",
                    "type": "item",
                    "limit": 5,
                    "format": "json",
                },
            )
            if resp.status_code != 200:
                return None
            hits = resp.json().get("search", []) or []
        except Exception:
            return None
        if not hits:
            return None

        wanted = {o.casefold() for o in orgs if o}
        fallback: str | None = None
        for hit in hits[:5]:
            qid = str(hit.get("id") or "")
            if not qid:
                continue
            entity = await self._entity(client, qid)
            if entity is None or not _is_human(entity):
                continue
            if fallback is None:
                fallback = qid
            if not wanted:
                return qid
            # Corroborate against what the graph already believes. Name search
            # on Wikidata is a homonym minefield exactly as it is on OpenAlex,
            # and an unrelated namesake's article poisons the frontier with a
            # complete set of the wrong person's relationships.
            described = str(hit.get("description") or "").casefold()
            if any(w in described for w in wanted):
                return qid
        # Orgs were supplied and none matched. Prefer the first human over
        # nothing: a description is a one-line summary and routinely omits the
        # employer, so absence here is weak evidence, unlike OpenAlex where the
        # affiliation is a structured field.
        return fallback

    async def _reverse(
        self, client: httpx.AsyncClient, qid: str, person: str
    ) -> list[Discovery]:
        """Organisations that name this person as founder, CEO, chair or board."""
        try:
            resp = await client.get(
                _SPARQL,
                params={"query": _REVERSE_QUERY % qid, "format": "json"},
                headers={"Accept": "application/sparql-results+json"},
            )
            if resp.status_code != 200:
                return []
            rows = (resp.json().get("results") or {}).get("bindings", []) or []
        except Exception:
            return []

        out: list[Discovery] = []
        seen: set[str] = set()
        for row in rows:
            article = ((row.get("article") or {}).get("value") or "").strip()
            if not article or article in seen:
                continue  # no English article: nothing to fetch
            seen.add(article)
            label = ((row.get("orgLabel") or {}).get("value") or "").strip()
            # The label service falls back to the bare QID when an entity has no
            # English label. The article title is a better name than "Q2616400".
            if _QID_ONLY.match(label):
                label = article.rsplit("/", 1)[-1].replace("_", " ")
            relation = ((row.get("rel") or {}).get("value") or "is involved with").strip()
            out.append(
                Discovery(
                    url=article,
                    provider=self.name,
                    why=f"{person} {relation} {label or article}",
                )
            )
        return out

    async def _entity(self, client: httpx.AsyncClient, qid: str) -> dict | None:
        try:
            resp = await client.get(
                _API,
                params={
                    "action": "wbgetentities",
                    "ids": qid,
                    "props": "claims|sitelinks|labels",
                    "languages": "en",
                    "sitefilter": "enwiki",
                    "format": "json",
                },
            )
            if resp.status_code != 200:
                return None
            return (resp.json().get("entities") or {}).get(qid)
        except Exception:
            return None

    async def _labels(
        self, client: httpx.AsyncClient, qids: Sequence[str]
    ) -> dict[str, tuple[str, str]]:
        """QID -> (english wikipedia title, label). Missing article = ("", label)."""
        found: dict[str, tuple[str, str]] = {}
        for start in range(0, len(qids), _BATCH):
            chunk = list(qids[start : start + _BATCH])
            try:
                resp = await client.get(
                    _API,
                    params={
                        "action": "wbgetentities",
                        "ids": "|".join(chunk),
                        "props": "sitelinks|labels",
                        "languages": "en",
                        "sitefilter": "enwiki",
                        "format": "json",
                    },
                )
                if resp.status_code != 200:
                    continue
                entities = resp.json().get("entities") or {}
            except Exception:
                continue
            for qid, entity in entities.items():
                if not isinstance(entity, dict):
                    continue
                label = str(
                    ((entity.get("labels") or {}).get("en") or {}).get("value", "")
                )
                found[qid] = (_sitelink(entity), label)
            await asyncio.sleep(0.1)
        return found


def _is_human(entity: dict) -> bool:
    for claim in (entity.get("claims") or {}).get("P31", []) or []:
        if _claim_qid(claim) == _HUMAN:
            return True
    return False


def _label(entity: dict) -> str:
    return str(((entity.get("labels") or {}).get("en") or {}).get("value", ""))


def _sitelink(entity: dict) -> str:
    link = (entity.get("sitelinks") or {}).get("enwiki") or {}
    title = str(link.get("title") or "")
    return title.replace(" ", "_")


def _claim_qid(claim: dict) -> str:
    value = ((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value")
    if isinstance(value, dict):
        return str(value.get("id") or "")
    return ""


def _targets(entity: dict) -> list[tuple[str, str]]:
    """(qid, relation phrase) for every followed property, deduped, capped."""
    claims = entity.get("claims") or {}
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for prop, relation in _PROPERTIES.items():
        for claim in claims.get(prop, []) or []:
            qid = _claim_qid(claim)
            if not qid or qid in seen:
                continue
            seen.add(qid)
            out.append((qid, relation))
            if len(out) >= _MAX_TARGETS:
                return out
    return out


