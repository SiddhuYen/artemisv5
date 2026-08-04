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


@dataclass(frozen=True)
class Assertion:
    """A relationship a curated record states outright, with no page to read.

    The exception to this package's URL-only rule, and only for sources that
    genuinely assert rather than co-list. A Wikidata claim or a registry
    officership is written by an editor or filed by a company; it is not a
    sentence we invented to feed the extractor, which is what the docstring
    above rejects.

    Two things make these worth admitting despite carrying no span. They are
    curated and cited at the source. And they arrive with a canonical entity id
    — a QID, a company number — where every other path through this system has
    only a name, so on identity they are stronger than anything the extractor
    can produce. They are labelled STRUCTURED_CLAIM throughout, and a route
    resting on one ranks below an equal-length route built from prose.
    """

    subject: str
    object: str
    #: Rendered as the hop's statement, e.g. "founded" or "is an officer of".
    relation: str
    #: The record itself — a Wikidata entity page, a registry filing.
    source_url: str
    source_title: str
    provider: str
    #: Canonical ids where the provider has them. These make the identity claim
    #: checkable rather than name-based.
    subject_id: str = ""
    object_id: str = ""


class StructuredProvider(Protocol):
    name: str

    def available(self) -> bool: ...

    async def discover(self, *, person: str, orgs: Sequence[str]) -> list[Discovery]: ...


class AssertingProvider(StructuredProvider, Protocol):
    """A provider whose records assert relationships, not merely locate them."""

    async def assert_relations(
        self, *, person: str, orgs: Sequence[str]
    ) -> list[Assertion]: ...


from artemis.providers.edgar import EdgarProvider  # noqa: E402
from artemis.providers.openalex import OpenAlexProvider  # noqa: E402
from artemis.providers.opencorporates import OpenCorporatesProvider  # noqa: E402
from artemis.providers.podcasts import PodcastProvider  # noqa: E402
from artemis.providers.propublica import ProPublicaProvider  # noqa: E402
from artemis.providers.rosters import RosterProvider  # noqa: E402
from artemis.providers.wikidata import WikidataProvider  # noqa: E402


def build_providers(settings) -> list[StructuredProvider]:  # type: ignore[no-untyped-def]
    """Every configured provider, in the order they should be consulted."""
    candidates: list[StructuredProvider] = [
        WikidataProvider(settings),    # curated claims — highest precision, free
        RosterProvider(settings),      # org team pages — densest person-to-person source
        EdgarProvider(settings),       # SEC filings
        OpenAlexProvider(settings),    # academic co-authorship
        PodcastProvider(settings),     # host interviewed guest
        ProPublicaProvider(settings),  # IRS 990 boards
        OpenCorporatesProvider(settings),
    ]
    return [p for p in candidates if p.available()]


__all__ = [
    "Assertion",
    "AssertingProvider",
    "Discovery",
    "StructuredProvider",
    "WikidataProvider",
    "RosterProvider",
    "EdgarProvider",
    "OpenAlexProvider",
    "PodcastProvider",
    "ProPublicaProvider",
    "OpenCorporatesProvider",
    "build_providers",
]
