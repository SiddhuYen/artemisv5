"""Structured sources that discover URLs the ordinary web search misses.

These are **URL-discovery** providers, not evidence providers. Each one queries a
structured index (SEC filings, IRS 990s, company registries) and returns links to
real documents; those documents then go through exactly the same fetch ->
extract -> ground path as a page found on Google. Every resulting edge still
carries a byte-exact span from a real source.

That is a deliberate departure from ArtemisV2, which renders structured records
into synthesized sentences ("{subject} coworker of {name}.") and extracts from
those. That works in v2's evidence model but not here: a sentence we wrote
ourselves is not a page asserting a relationship, and presenting it as one would
break the invariant the whole tool rests on. Co-membership of a board is not, on
its own, a stated relationship — so these providers surface the filings where
such relationships are actually *described*, and let the extractor decide.

Serper remains the only general search provider. These are narrow, structured
indexes queried by name, not a second door to the open web.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Discovery:
    """A URL worth fetching, plus why this provider thinks so (for the job log)."""

    url: str
    provider: str
    why: str


class StructuredProvider(Protocol):
    name: str

    def available(self) -> bool: ...

    async def discover(self, *, person: str, orgs: Sequence[str]) -> list[Discovery]: ...


from artemis.providers.edgar import EdgarProvider  # noqa: E402
from artemis.providers.opencorporates import OpenCorporatesProvider  # noqa: E402
from artemis.providers.propublica import ProPublicaProvider  # noqa: E402


def build_providers(settings) -> list[StructuredProvider]:  # type: ignore[no-untyped-def]
    """Every configured provider, in the order they should be consulted."""
    candidates: list[StructuredProvider] = [
        EdgarProvider(settings),
        ProPublicaProvider(settings),
        OpenCorporatesProvider(settings),
    ]
    return [p for p in candidates if p.available()]


__all__ = [
    "Discovery",
    "StructuredProvider",
    "EdgarProvider",
    "ProPublicaProvider",
    "OpenCorporatesProvider",
    "build_providers",
]
