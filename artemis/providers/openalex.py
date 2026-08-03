"""OpenAlex — academic co-authorship.

Free, no key, person-centric: resolve a name to an author, then collect the
works they appear on. High recall for anyone who has ever published — which in
practice includes many executives, clinicians, and researchers, not only
academics.

A paper's author list is a roster in exactly the sense co-listing means: the
work states that these people are its authors, and nothing about whether they
know each other. So this returns the work's landing page for the ordinary
fetch -> extract -> ground path, where the author list grounds as co-listings
and any prose in the abstract grounds as assertions.
"""

from __future__ import annotations

import asyncio
from typing import Any, Sequence

import httpx

from artemis.providers import Discovery

_AUTHORS = "https://api.openalex.org/authors"
_WORKS = "https://api.openalex.org/works"

_MAX_WORKS = 10
_MIN_AUTHORS = 2
#: A 300-author physics collaboration says nothing useful about any pair in it.
_MAX_AUTHORS = 30

#: Institution words that carry no identifying signal.
_INST_NOISE = {
    "university", "universite", "college", "institute", "school", "of", "the",
    "and", "for", "center", "centre", "department", "dept", "lab", "laboratory",
    "hospital", "medical", "research", "national", "state", "system", "inc",
}


def _tokens(name: str) -> set[str]:
    return {
        t.strip(",.")
        for t in name.casefold().split()
        if t.strip(",.") and t.strip(",.") not in _INST_NOISE and len(t.strip(",.")) > 2
    }


class OpenAlexProvider:
    name = "openalex"

    def __init__(self, settings: Any) -> None:
        self.s = settings

    def available(self) -> bool:
        return bool(getattr(self.s, "openalex_enabled", True))

    async def discover(self, *, person: str, orgs: Sequence[str]) -> list[Discovery]:
        headers = {"User-Agent": self.s.user_agent}
        async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
            author = await self._resolve_author(client, person, orgs)
            if author is None:
                return []
            author_id, display = author
            await asyncio.sleep(0.1)  # OpenAlex asks for polite pacing
            works = await self._works(client, author_id)

        out: list[Discovery] = []
        for work in works:
            authorships = work.get("authorships") or []
            if not (_MIN_AUTHORS <= len(authorships) <= _MAX_AUTHORS):
                continue
            url = (
                (work.get("primary_location") or {}).get("landing_page_url")
                or work.get("doi")
                or work.get("id")
            )
            if not url:
                continue
            title = str(work.get("display_name") or "")[:90]
            out.append(
                Discovery(
                    url=url,
                    provider=self.name,
                    why=f"{display} co-authored {title!r} with {len(authorships) - 1} others",
                )
            )
        return out

    async def _resolve_author(
        self, client: httpx.AsyncClient, person: str, orgs: Sequence[str]
    ) -> tuple[str, str] | None:
        try:
            resp = await client.get(_AUTHORS, params={"search": person, "per-page": 5})
            if resp.status_code != 200:
                return None
            results = resp.json().get("results", []) or []
        except Exception:
            return None
        if not results:
            return None

        # Prefer an author whose institution matches something we already know;
        # name search alone on OpenAlex is a homonym minefield.
        #
        # Match on significant tokens, not whole strings: institutions are
        # written out in full here ("University of California, Berkeley") while
        # our attributes are colloquial ("UC Berkeley"), so substring matching
        # finds nothing even when it is plainly the same place.
        wanted = {t for o in orgs for t in _tokens(o)}
        for item in results:
            affiliation_tokens = {
                t
                for inst in (item.get("last_known_institutions") or [])
                for t in _tokens(str((inst or {}).get("display_name", "")))
            }
            if wanted and wanted & affiliation_tokens:
                return str(item.get("id", "")), str(item.get("display_name", person))

        # Orgs were supplied and none of them matched: this is a different
        # person with the same name. Searching "Diana Hu" here returns a
        # paediatric vaccine researcher, and a relevance-ranked fallback happily
        # accepts her — nine papers on infant immunisation attributed to a YC
        # partner. When we have something to check against and it fails, stop.
        if wanted:
            return None

        # Nothing to corroborate with. Take the top hit only when it is
        # unambiguous; otherwise decline rather than guess between homonyms.
        top = results[0]
        if len(results) > 1 and top.get("works_count", 0) < 3:
            return None
        return str(top.get("id", "")), str(top.get("display_name", person))

    async def _works(self, client: httpx.AsyncClient, author_id: str) -> list[dict]:
        try:
            resp = await client.get(
                _WORKS,
                params={
                    "filter": f"author.id:{author_id}",
                    "per-page": _MAX_WORKS,
                    "sort": "cited_by_count:desc",
                },
            )
            if resp.status_code != 200:
                return []
            return resp.json().get("results", []) or []
        except Exception:
            return []
