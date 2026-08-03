"""Domain models for ARTEMIS.

Everything the service returns is built from these types. Three invariants are
enforced here rather than left to convention:

1. An edge without a grounding span cannot be constructed. ``Edge`` requires an
   ``Extraction``, and ``Extraction`` requires a verbatim ``span_text`` plus the
   offsets it was found at.
2. ``resolved_statement`` is always labelled as derived. It never occupies the
   same field as source text.
3. Identity basis is an enum with a total order, so "weakest basis on the route"
   is computed, not asserted.

Search-layer wire models (Query, SearchHit, SearchResults) live in
``artemis/search/base.py`` beside the provider protocol — they never escape that
layer. ``PageDocument`` lives here because its character offsets are the
substrate every returned span is checked against.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(dt: datetime) -> str:
    """RFC 3339 with a literal Z, matching the documented result schema."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ResolutionBasis(str, Enum):
    """How the extractor tied a mention in the span to a full name."""

    DIRECT = "direct"
    PRONOUN = "pronoun"
    DEFINITE_DESCRIPTION = "definite_description"
    APPOSITION = "apposition"
    LIST_CONTINUATION = "list_continuation"
    #: Not an asserted relationship. The page lists both people under one
    #: affiliation (a team page, a board roster, a cohort). The span is their
    #: verbatim listing and `context_before` is the heading that establishes
    #: what they share. Always labelled distinctly in output — co-membership is
    #: a fact about a page, not a claim that these two people know each other.
    CO_LISTING = "co_listing"


#: Hops whose evidence is co-membership rather than a stated relationship.
CO_MEMBERSHIP_BASES = frozenset({ResolutionBasis.CO_LISTING})


class IdentityBasis(str, Enum):
    """Evidence that two observations of a name are the same human."""

    SHARED_PAGE = "shared_page"
    CANONICAL_URL = "canonical_url"
    ATTRIBUTE_MATCH = "attribute_match"
    NAME_ONLY = "name_only"


# Higher is stronger. Used to compute a route's weakest pivot.
IDENTITY_BASIS_STRENGTH: dict[IdentityBasis, int] = {
    IdentityBasis.SHARED_PAGE: 4,
    IdentityBasis.CANONICAL_URL: 3,
    IdentityBasis.ATTRIBUTE_MATCH: 2,
    IdentityBasis.NAME_ONLY: 1,
}


def weakest_basis(bases: list[IdentityBasis]) -> IdentityBasis:
    """Weakest link in a chain of identity claims. Empty chain reads as name_only."""
    if not bases:
        return IdentityBasis.NAME_ONLY
    return min(bases, key=lambda b: IDENTITY_BASIS_STRENGTH[b])


class MergeDecisionKind(str, Enum):
    MERGED = "merged"
    BLOCKED = "blocked"
    HELD_SEPARATE = "held_separate"


class MergeBasis(str, Enum):
    """Rungs of the merge ladder, in the order resolve.py tries them."""

    SAME_CANONICAL_URL = "same_canonical_url"
    SINGLE_REFERENT_CO_OCCURRENCE = "single_referent_co_occurrence"
    #: The page was returned by a query naming this person (and, for an
    #: endpoint, their disambiguator). Search-engine provenance is evidence
    #: about who the page is about — weaker than a canonical URL, stronger
    #: than a bare name match.
    SEARCH_PROVENANCE = "search_provenance"
    ATTRIBUTES_PLUS_SHARED_NEIGHBOR = "attributes_plus_shared_neighbor"
    RARE_NAME_ATTRIBUTES = "rare_name_attributes"
    # Non-merging outcomes, recorded with the same vocabulary.
    NAME_STRING_ONLY = "name_string_only"
    CONFLICTING_ATTRIBUTES = "conflicting_attributes"
    NAME_TOO_COMMON = "name_too_common"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Endpoint(str, Enum):
    """Which side of the bidirectional search a node/frontier belongs to."""

    A = "a"
    B = "b"


ATTRIBUTE_KEYS: tuple[str, ...] = (
    "employer",
    "role",
    "institution",
    "location",
    "field",
)


# ---------------------------------------------------------------------------
# Page substrate
# ---------------------------------------------------------------------------


class Sentence(BaseModel):
    """A sentence with offsets into the parent document's extracted text.

    ``text == document.text[start:end]`` is an invariant maintained by
    extract_text.py and re-checked whenever a span is verified.
    """

    index: int
    start: int
    end: int
    text: str

    @model_validator(mode="after")
    def _offsets_match_text(self) -> "Sentence":
        if self.end - self.start != len(self.text):
            raise ValueError(
                f"sentence {self.index}: offset width {self.end - self.start} "
                f"!= len(text) {len(self.text)}"
            )
        return self


class PageDocument(BaseModel):
    """Fetched page reduced to clean text with stable offsets.

    Paragraph boundaries are preserved in ``text``; the extraction layer depends
    on sentence adjacency for referent resolution.
    """

    url: str
    final_url: str
    title: Optional[str] = None
    text: str
    sentences: list[Sentence] = Field(default_factory=list)
    retrieved_at: datetime = Field(default_factory=utcnow)
    text_sha256: str = ""
    from_cache: bool = False

    @model_validator(mode="after")
    def _fill_hash(self) -> "PageDocument":
        if not self.text_sha256:
            object.__setattr__(self, "text_sha256", content_hash(self.text))
        return self

    def slice(self, start: int, end: int) -> str:
        return self.text[start:end]

    def contains_span(self, span_text: str, start: int, end: int) -> bool:
        """Byte-exact check that a span is really at those offsets."""
        return 0 <= start <= end <= len(self.text) and self.text[start:end] == span_text


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class RawExtraction(BaseModel):
    """Exactly what Claude is allowed to return, before verification.

    Claude names a sentence index and copies the span; it never invents
    character offsets. grounding.py locates ``span_text`` inside that sentence's
    range, computes offsets, and only then constructs an ``Extraction``. A span
    that cannot be located is discarded and logged.
    """

    model_config = ConfigDict(extra="forbid")

    subject_name: str
    object_name: str
    sentence_index: int = Field(ge=0)
    span_text: str = Field(min_length=1)
    context_before: str = ""
    resolved_statement: str
    resolution_basis: ResolutionBasis
    subject_attributes: list[str] = Field(default_factory=list)
    object_attributes: list[str] = Field(default_factory=list)


class RawCoListing(BaseModel):
    """A roster the model found: people sharing one affiliation on one page.

    Kept separate from RawExtraction because the evidence is different in kind.
    Nobody asserted a relationship here; a page put these names in one list, and
    the affiliation is what they share.
    """

    model_config = ConfigDict(extra="forbid")

    affiliation: str  # "Y Combinator Group Partners", "Board of Trustees", ...
    affiliation_kind: str  # employer | board | cohort | investors | event | other
    #: Index of the heading establishing the affiliation, or -1 when the page
    #: states it only in its title. Many team pages open straight into names,
    #: with the organisation named only in <title> and the URL.
    context_sentence_index: int = Field(ge=-1)
    context_text: str  # verbatim heading, or the verbatim page title when index is -1
    members: list["RawCoListingMember"] = Field(default_factory=list)


class RawCoListingMember(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    sentence_index: int = Field(ge=0)
    span_text: str = Field(min_length=1)
    role: str = ""


class Extraction(BaseModel):
    """A verified relationship assertion tied to exact page offsets."""

    model_config = ConfigDict(extra="forbid")

    subject_name: str  # resolved, full name
    object_name: str  # resolved, full name
    span_text: str = Field(min_length=1)  # VERBATIM; must match page text exactly
    span_start: int = Field(ge=0)  # char offset into extracted text
    span_end: int = Field(ge=0)
    context_before: str = ""  # verbatim preceding sentences used for resolution
    resolved_statement: str  # span with referents substituted — DERIVED, not source text
    resolution_basis: ResolutionBasis
    subject_attributes: list[str] = Field(default_factory=list)
    object_attributes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_invariants(self) -> "Extraction":
        if self.span_end <= self.span_start:
            raise ValueError("span_end must be greater than span_start")
        if self.span_end - self.span_start != len(self.span_text):
            raise ValueError(
                "span offsets do not match span_text length "
                f"({self.span_end - self.span_start} != {len(self.span_text)})"
            )
        if self.resolution_basis is not ResolutionBasis.DIRECT and not self.context_before.strip():
            raise ValueError(
                f"resolution_basis={self.resolution_basis.value} requires a non-empty "
                "context_before containing the antecedent"
            )
        if self.subject_name.strip().casefold() == self.object_name.strip().casefold():
            raise ValueError("subject and object resolve to the same name")
        return self

    def verify_against(self, page: PageDocument) -> bool:
        """Byte-exact substring check. Callers discard the extraction on False."""
        return page.contains_span(self.span_text, self.span_start, self.span_end)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    """One sighting of a person on one page."""

    url: str
    page_title: Optional[str] = None
    span_text: str
    span_start: int
    span_end: int
    retrieved_at: datetime = Field(default_factory=utcnow)
    attributes: dict[str, set[str]] = Field(default_factory=dict)

    @field_serializer("attributes")
    def _sorted_attributes(self, v: dict[str, set[str]]) -> dict[str, list[str]]:
        return {k: sorted(vals) for k, vals in sorted(v.items())}

    @field_serializer("retrieved_at")
    def _ts(self, v: datetime) -> str:
        return iso_z(v)


class Node(BaseModel):
    """A person. Deliberately not a name string.

    Default is DO NOT MERGE: two same-name observations stay separate nodes
    until resolve.py finds evidence on the ladder.
    """

    node_id: str
    display_name: str
    name_variants: set[str] = Field(default_factory=set)
    attributes: dict[str, set[str]] = Field(default_factory=dict)
    canonical_urls: set[str] = Field(default_factory=set)
    observations: list[Observation] = Field(default_factory=list)
    endpoint: Optional[Endpoint] = None  # which frontier discovered it
    discovered_at_depth: int = 0

    @field_serializer("name_variants", "canonical_urls")
    def _sorted_set(self, v: set[str]) -> list[str]:
        return sorted(v)

    @field_serializer("attributes")
    def _sorted_attributes(self, v: dict[str, set[str]]) -> dict[str, list[str]]:
        return {k: sorted(vals) for k, vals in sorted(v.items())}

    @property
    def source_urls(self) -> set[str]:
        return {o.url for o in self.observations}


class MergeDecision(BaseModel):
    """Every merge decision goes in the job log with its basis."""

    decision: MergeDecisionKind
    basis: MergeBasis
    name: str
    left_node_id: str
    right_node_id: str
    detail: str = ""
    claude_assisted: bool = False
    decided_at: datetime = Field(default_factory=utcnow)

    @field_serializer("decided_at")
    def _ts(self, v: datetime) -> str:
        return iso_z(v)


class DisambiguationCandidate(BaseModel):
    """One reading of an ambiguous endpoint name, returned instead of guessing."""

    display_name: str
    attributes: dict[str, list[str]] = Field(default_factory=dict)
    source_url: str
    source_title: Optional[str] = None
    snippet: Optional[str] = None


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


class Edge(BaseModel):
    """A grounded relationship. Cannot exist without its span.

    No strength score, no edge type, no confidence weight — by design.
    """

    edge_id: str
    subject_node_id: str
    object_node_id: str
    extraction: Extraction
    source_url: str
    source_title: Optional[str] = None
    retrieved_at: datetime = Field(default_factory=utcnow)

    @field_serializer("retrieved_at")
    def _ts(self, v: datetime) -> str:
        return iso_z(v)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


class HopEndpoint(BaseModel):
    name: str
    node_id: str


class Hop(BaseModel):
    """One hop of a returned route, in the documented wire shape.

    ``from`` is a Python keyword, so the field is ``from_`` with an alias.
    FastAPI serialises response models with ``by_alias=True`` by default; dump
    manually with ``model_dump(by_alias=True)`` outside a route.
    """

    model_config = ConfigDict(populate_by_name=True)

    from_: HopEndpoint = Field(alias="from")
    to: HopEndpoint
    span_text: str
    span_start: int
    span_end: int
    context_before: str = ""
    resolved_statement: str
    resolution_basis: ResolutionBasis
    source_url: str
    source_title: Optional[str] = None
    retrieved_at: datetime
    from_identity_basis: IdentityBasis
    to_identity_basis: IdentityBasis

    @field_serializer("retrieved_at")
    def _ts(self, v: datetime) -> str:
        return iso_z(v)


class Route(BaseModel):
    length: int
    weakest_identity_basis: IdentityBasis
    identity_warnings: list[str] = Field(default_factory=list)
    hops: list[Hop]

    @model_validator(mode="after")
    def _length_matches(self) -> "Route":
        if self.length != len(self.hops):
            raise ValueError(f"length {self.length} != {len(self.hops)} hops")
        return self


class Stats(BaseModel):
    serper_queries: int = 0
    serper_credits_used: int = 0
    pages_fetched: int = 0
    claude_calls: int = 0
    nodes_expanded: int = 0
    merges: int = 0
    merges_blocked: int = 0
    elapsed_s: float = 0.0


class Result(BaseModel):
    found: bool
    routes: list[Route] = Field(default_factory=list)
    disambiguation: Optional[list[DisambiguationCandidate]] = None
    stats: Stats = Field(default_factory=Stats)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


class Budget(BaseModel):
    """Per-request budget overrides. Unset fields fall back to config defaults."""

    model_config = ConfigDict(extra="forbid")

    max_depth_a: Optional[int] = Field(default=None, ge=1, le=6)
    max_depth_b: Optional[int] = Field(default=None, ge=0, le=6)
    max_nodes_expanded: Optional[int] = Field(default=None, ge=1)
    max_serper_credits: Optional[int] = Field(default=None, ge=1)
    max_fetches: Optional[int] = Field(default=None, ge=1)
    max_claude_calls: Optional[int] = Field(default=None, ge=0)
    wall_clock_s: Optional[float] = Field(default=None, gt=0)


class ConnectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    person_a: str = Field(min_length=1)
    person_b: str = Field(min_length=1)
    context_a: Optional[str] = None  # free-text disambiguator
    context_b: Optional[str] = None
    max_depth: Optional[int] = Field(default=None, ge=1, le=6)
    budget: Optional[Budget] = None


class ConnectAccepted(BaseModel):
    job_id: str


class LogEntry(BaseModel):
    """Structured event emitted as the crawl runs; a poller watches these live."""

    ts: datetime = Field(default_factory=utcnow)
    level: LogLevel = LogLevel.INFO
    event: str  # e.g. "query.issued", "page.fetched", "edge.grounded", "merge.decided"
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)

    @field_serializer("ts")
    def _ts(self, v: datetime) -> str:
        return iso_z(v)


class JobState(BaseModel):
    id: str
    status: JobStatus = JobStatus.QUEUED
    request: ConnectRequest
    log: list[LogEntry] = Field(default_factory=list)
    result: Optional[Result] = None
    warnings: list[str] = Field(default_factory=list)
    stats: Stats = Field(default_factory=Stats)
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @field_serializer("created_at", "updated_at")
    def _ts(self, v: datetime) -> str:
        return iso_z(v)


class JobView(BaseModel):
    """GET /jobs/{id} payload."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    status: JobStatus
    log: list[LogEntry]
    result: Optional[Result] = None
    warnings: list[str] = Field(default_factory=list)
    stats: Stats


# Degradation warnings are string constants so tests can assert on them.
WARN_NO_REFERENT_RESOLUTION = "degraded: no referent resolution"
WARN_NAME_ONLY_PIVOT = "route contains a name_only pivot: hops are individually grounded but the pivot may be two different people"
WARN_CO_LISTING_HOP = (
    "route contains a co_listing hop: the source lists both people under one "
    "affiliation but does not state that they know each other"
)
