"""ProPublica Nonprofit Explorer — IRS Form 990 boards and officers.

Free, no key. The API exposes aggregate financials and organisation identity but
not officer names, so the useful artefact is the organisation's own page, whose
990 filings list trustees, directors, and officers with titles.

Only helps when an organisation is a US nonprofit: a for-profit company returns
nothing here, which is the correct answer rather than a failure.
"""

from __future__ import annotations

from typing import Any, Sequence

import httpx

from artemis.identity.normalize import fold
from artemis.providers import Discovery

_SEARCH = "https://projects.propublica.org/nonprofits/api/v2/search.json"
_ORG_PAGE = "https://projects.propublica.org/nonprofits/organizations/{ein}"

_MAX_ORGS = 3
_STOPWORDS = {"the", "of", "and", "for", "inc", "llc", "foundation", "trust",
              "institute", "association", "society", "center", "centre"}


def _significant(name: str) -> set[str]:
    return {t for t in fold(name).replace(",", " ").split() if t not in _STOPWORDS and len(t) > 2}


class ProPublicaProvider:
    name = "propublica"

    def __init__(self, settings: Any) -> None:
        self.s = settings

    def available(self) -> bool:
        return bool(getattr(self.s, "propublica_enabled", True))

    async def discover(self, *, person: str, orgs: Sequence[str]) -> list[Discovery]:
        if not orgs:
            return []
        out: list[Discovery] = []
        seen: set[str] = set()
        async with httpx.AsyncClient(
            timeout=20.0, headers={"User-Agent": self.s.user_agent}
        ) as client:
            for org in list(orgs)[:_MAX_ORGS]:
                for ein, matched_name in await self._resolve(client, org):
                    url = _ORG_PAGE.format(ein=ein)
                    if url in seen:
                        continue
                    seen.add(url)
                    out.append(
                        Discovery(
                            url=url,
                            provider=self.name,
                            why=f"IRS 990 filings for {matched_name} (matched {org!r})",
                        )
                    )
        return out

    async def _resolve(self, client: httpx.AsyncClient, org: str) -> list[tuple[str, str]]:
        """EINs whose name genuinely overlaps `org` — the API matches loosely."""
        try:
            resp = await client.get(_SEARCH, params={"q": org})
            if resp.status_code != 200:
                return []
            found = resp.json().get("organizations", []) or []
        except Exception:
            return []

        # A one-word organisation name is not specific enough to match on: with
        # a single token allowed to differ, "Host" matched "Phantom Host" and
        # "Calexico Host", and the crawl went off fetching unrelated 990s.
        #
        # Count raw words, not significant ones — nonprofits are overwhelmingly
        # named "<Something> Foundation" or "<Something> Trust", and both of
        # those words are stopwords here. Gating on significant tokens would
        # have switched this provider off for most of its own domain.
        if len(org.split()) < 2:
            return []
        wanted = _significant(org)
        if not wanted:
            return []
        matches: list[tuple[str, str]] = []
        for item in found[:10]:
            name = str(item.get("name") or "")
            ein = str(item.get("ein") or "")
            found_tokens = _significant(name)
            # Require real overlap in BOTH directions. Containment alone let the
            # one-word org "Niantic" match "Niantic Christian Church" — and
            # Niantic is also a town, so the org wasn't even the subject.
            if not (ein and wanted <= found_tokens):
                continue
            if len(found_tokens - wanted) > 1:
                continue
            matches.append((ein, name))
            if len(matches) >= 2:
                break
        return matches
