"""Podcast episodes — host interviewed guest.

An episode asserts exactly one relationship: this host interviewed this guest.
That is demonstrated and on the record, and it is the kind of link that rarely
appears anywhere else for otherwise-obscure professionals.

It asserts **nothing** about two guests of the same show, who have typically
never met — so this only ever surfaces episode pages, and the ordinary
extractor decides what the episode text actually supports. The show's back
catalogue is not a roster and must never be treated as one.

Uses the free iTunes Search API to locate shows, then their public RSS feeds.
No key.
"""

from __future__ import annotations

import re
from typing import Any, Sequence
from xml.etree import ElementTree

import httpx

from artemis.identity.normalize import fold
from artemis.providers import Discovery

_ITUNES_SEARCH = "https://itunes.apple.com/search"

_MAX_SHOWS = 3
_MAX_EPISODES = 6
_TAG = re.compile(r"<[^>]+>")


class PodcastProvider:
    name = "podcasts"

    def __init__(self, settings: Any) -> None:
        self.s = settings

    def available(self) -> bool:
        return bool(getattr(self.s, "podcasts_enabled", True))

    async def discover(self, *, person: str, orgs: Sequence[str]) -> list[Discovery]:
        headers = {"User-Agent": self.s.user_agent}
        out: list[Discovery] = []
        seen: set[str] = set()
        needle = fold(person)

        async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
            for feed_url, show in await self._shows(client, person):
                for link, title in await self._episodes(client, feed_url, needle):
                    if link in seen:
                        continue
                    seen.add(link)
                    out.append(
                        Discovery(
                            url=link,
                            provider=self.name,
                            why=f"{show!r} episode naming {person}: {title[:70]!r}",
                        )
                    )
                    if len(out) >= _MAX_EPISODES:
                        return out
        return out

    async def _shows(self, client: httpx.AsyncClient, person: str) -> list[tuple[str, str]]:
        try:
            resp = await client.get(
                _ITUNES_SEARCH,
                params={"term": person, "media": "podcast", "limit": _MAX_SHOWS},
            )
            if resp.status_code != 200:
                return []
            results = resp.json().get("results", []) or []
        except Exception:
            return []
        return [
            (str(r.get("feedUrl")), str(r.get("collectionName", "")))
            for r in results
            if r.get("feedUrl")
        ]

    async def _episodes(
        self, client: httpx.AsyncClient, feed_url: str, needle: str
    ) -> list[tuple[str, str]]:
        """Episodes whose title or description actually names the person."""
        try:
            resp = await client.get(feed_url)
            if resp.status_code != 200:
                return []
            root = ElementTree.fromstring(resp.content)
        except Exception:
            return []

        found: list[tuple[str, str]] = []
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            summary = _TAG.sub(" ", item.findtext("description") or "")
            link = (item.findtext("link") or "").strip()
            if not link:
                continue
            if needle and needle not in fold(f"{title} {summary}"):
                continue
            found.append((link, title))
            if len(found) >= _MAX_EPISODES:
                break
        return found
