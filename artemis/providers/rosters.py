"""Organisation team pages — located by convention, not by search.

The densest person-to-person document in existence is an org's own team page:
one heading, twenty names, all genuinely affiliated. Co-listing can ground it,
but only if something fetches it, and person-shaped web queries never return one.

Rather than spend search credits, this exploits the fact that team pages live at
conventional paths. It resolves an organisation name to a likely domain, checks
the domain actually answers, then offers the handful of paths teams live at. The
fetcher's soft-404 and paywall detection discards the misses, so a wrong guess
costs one request and nothing else.

Guessing a domain is a heuristic, and a wrong guess could point at an unrelated
company. That is safe here for the reason every provider in this package is
safe: a provider only ever nominates a URL. If the page turns out to be some
other company's team, the anchor person will not be listed on it and
`ground_co_listings` discards the whole roster.
"""

from __future__ import annotations

import asyncio
from typing import Any, Sequence

import httpx

from artemis.identity.normalize import fold
from artemis.providers import Discovery

#: Where teams live, in rough order of how often they are found there.
_PATHS = (
    "/team", "/our-team", "/people", "/about/team", "/leadership",
    "/partners", "/who-we-are", "/staff", "/about-us/team", "/about",
)
_TLDS = (".com", ".io", ".co", ".vc", ".org")

_MAX_ORGS = 2
_MAX_PATHS = 6
#: Words that are part of an org's legal name but never its domain.
_DOMAIN_NOISE = {
    "the", "inc", "llc", "ltd", "corp", "corporation", "company", "co",
    "group", "holdings", "plc", "gmbh", "limited", "and",
}


def domain_candidates(org: str) -> list[str]:
    """Likely hosts for an organisation name, most probable first."""
    tokens = [t for t in fold(org).replace(",", " ").replace(".", " ").split()
              if t and t not in _DOMAIN_NOISE]
    if not tokens:
        return []
    joined = "".join(tokens)
    hyphenated = "-".join(tokens)
    stems = [joined] if joined == hyphenated else [joined, hyphenated]
    return [f"{stem}{tld}" for stem in stems for tld in _TLDS]


class RosterProvider:
    name = "rosters"

    def __init__(self, settings: Any) -> None:
        self.s = settings

    def available(self) -> bool:
        return bool(getattr(self.s, "rosters_enabled", True))

    async def discover(self, *, person: str, orgs: Sequence[str]) -> list[Discovery]:
        if not orgs:
            return []
        out: list[Discovery] = []
        headers = {"User-Agent": self.s.user_agent}
        async with httpx.AsyncClient(
            timeout=12.0, headers=headers, follow_redirects=True
        ) as client:
            for org in list(orgs)[:_MAX_ORGS]:
                host = await self._live_host(client, org)
                if host is None:
                    continue
                for path in _PATHS[:_MAX_PATHS]:
                    out.append(
                        Discovery(
                            url=f"{host}{path}",
                            provider=self.name,
                            why=f"conventional team-page path for {org}",
                        )
                    )
        return out

    async def _live_host(self, client: httpx.AsyncClient, org: str) -> str | None:
        """First candidate domain that actually answers with HTML."""
        for candidate in domain_candidates(org)[:6]:
            url = f"https://{candidate}"
            try:
                resp = await client.get(url)
            except Exception:
                continue
            finally:
                await asyncio.sleep(0.05)
            if resp.status_code < 400 and "html" in resp.headers.get("content-type", ""):
                # Keep whatever the redirect settled on (www., trailing slash).
                return str(resp.url).rstrip("/")
        return None
