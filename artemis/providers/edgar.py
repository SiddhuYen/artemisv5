"""SEC EDGAR full-text search — public-company filings.

Free, no key, but the SEC requires a declared User-Agent carrying contact info
and rate-limits aggressively.

The high-value case is not the subject's own filings — a private-company founder
has none. It is *other* companies' 8-K exhibits: a licensing or financing press
release filed as EX-99.1 names executives on both sides of the deal, in prose,
which is exactly the kind of stated relationship this tool can ground. Searching
the org name rather than the person is what surfaces those.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Sequence

import httpx

from artemis.providers import Discovery

_FTS = "https://efts.sec.gov/LATEST/search-index"
_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}"

_MAX_HITS_PER_QUERY = 8
_MAX_TOTAL = 20
#: Exhibit and report types whose bodies are prose rather than tables of numbers.
_PROSE_FORMS = ("EX-99", "8-K", "S-1", "424B", "DEF 14A", "10-K", "20-F", "6-K")
_TICKER = re.compile(r"\s\([A-Z]{1,6}\)")


class EdgarProvider:
    name = "edgar"

    def __init__(self, settings: Any) -> None:
        self.s = settings

    def available(self) -> bool:
        return bool(getattr(self.s, "edgar_enabled", True))

    async def discover(self, *, person: str, orgs: Sequence[str]) -> list[Discovery]:
        queries: list[tuple[str, str]] = [(f'"{person}"', f"named in filings as {person}")]
        for org in list(orgs)[:2]:
            queries.append((f'"{org}"', f"filings mentioning {org}"))

        out: list[Discovery] = []
        seen: set[str] = set()
        headers = {"User-Agent": self.s.edgar_user_agent, "Accept-Encoding": "gzip, deflate"}

        async with httpx.AsyncClient(timeout=25.0, headers=headers) as client:
            for query, why in queries:
                if len(out) >= _MAX_TOTAL:
                    break
                for hit in await self._hits(client, query):
                    url = self._filing_url(hit)
                    if not url or url in seen:
                        continue
                    source = hit.get("_source", {}) or {}
                    if not self._is_prose(source):
                        continue
                    seen.add(url)
                    filer = _TICKER.sub("", (source.get("display_names") or [""])[0]).strip()
                    out.append(
                        Discovery(
                            url=url,
                            provider=self.name,
                            why=f"{why}: {source.get('file_type', '?')} "
                                f"{source.get('file_date', '')} filed by {filer}",
                        )
                    )
                    if len(out) >= _MAX_TOTAL:
                        break
                await asyncio.sleep(0.2)  # SEC fair-access rate limit
        return out

    async def _hits(self, client: httpx.AsyncClient, query: str) -> list[dict]:
        try:
            resp = await client.get(_FTS, params={"q": query})
            if resp.status_code != 200:
                return []
            hits = resp.json().get("hits", {}).get("hits", [])
        except Exception:
            return []
        return hits[:_MAX_HITS_PER_QUERY]

    def _is_prose(self, source: dict) -> bool:
        form = str(source.get("file_type") or "")
        return any(form.startswith(p) for p in _PROSE_FORMS)

    def _filing_url(self, hit: dict) -> str:
        # _id is "<accession>:<filename>"; the archive path drops the dashes.
        accession, _, filename = str(hit.get("_id") or "").partition(":")
        ciks = (hit.get("_source", {}) or {}).get("ciks") or []
        if not (accession and filename and ciks):
            return ""
        return _ARCHIVE.format(
            cik=str(ciks[0]).lstrip("0"),
            accession=accession.replace("-", ""),
            filename=filename,
        )
