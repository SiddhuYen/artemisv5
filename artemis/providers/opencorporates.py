"""OpenCorporates — company officer networks from statutory registries.

Requires OPENCORPORATES_API_TOKEN (free tier available; anonymous access is
throttled to the point of uselessness). Absent token => provider reports itself
unavailable and the crawl proceeds without it.

Returns links to company pages whose officer lists name people alongside the
subject. As with the other structured providers, the page is fetched and
extracted normally — co-officership is not itself treated as a stated
relationship.
"""

from __future__ import annotations

from typing import Any, Sequence

import httpx

from artemis.providers import Discovery

_BASE = "https://api.opencorporates.com/v0.4"
_COMPANY_PAGE = "https://opencorporates.com/companies/{jurisdiction}/{number}"

_MAX_COMPANIES = 5


class OpenCorporatesProvider:
    name = "opencorporates"

    def __init__(self, settings: Any) -> None:
        self.s = settings

    @property
    def _token(self) -> str:
        return (getattr(self.s, "opencorporates_api_token", "") or "").strip()

    def available(self) -> bool:
        return bool(self._token)

    async def discover(self, *, person: str, orgs: Sequence[str]) -> list[Discovery]:
        out: list[Discovery] = []
        seen: set[str] = set()
        async with httpx.AsyncClient(
            timeout=20.0, headers={"User-Agent": self.s.user_agent}
        ) as client:
            for jurisdiction, number, company in await self._companies(client, person):
                url = _COMPANY_PAGE.format(jurisdiction=jurisdiction, number=number)
                if url in seen:
                    continue
                seen.add(url)
                out.append(
                    Discovery(
                        url=url,
                        provider=self.name,
                        why=f"{person} listed as an officer of {company}",
                    )
                )
        return out

    async def _companies(
        self, client: httpx.AsyncClient, person: str
    ) -> list[tuple[str, str, str]]:
        try:
            resp = await client.get(
                f"{_BASE}/officers/search",
                params={"q": person, "per_page": 20, "api_token": self._token},
            )
            if resp.status_code != 200:
                return []
            officers = resp.json().get("results", {}).get("officers", []) or []
        except Exception:
            return []

        found: list[tuple[str, str, str]] = []
        for entry in officers:
            officer = entry.get("officer", {}) or {}
            company = officer.get("company", {}) or {}
            jurisdiction = company.get("jurisdiction_code")
            number = company.get("company_number")
            if jurisdiction and number:
                found.append((jurisdiction, number, company.get("name", "")))
            if len(found) >= _MAX_COMPANIES:
                break
        return found
