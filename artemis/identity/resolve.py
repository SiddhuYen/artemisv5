"""Merge ladder, rarity gate, conflict blocking. Rules are code, not model judgment.

Default is DO NOT MERGE. Two same-name observations stay separate nodes unless
merge evidence exists, in this order:

  1. same canonical profile URL
  2. co-occurrence on one page as a single referent
  3. compatible attributes + a shared neighbour
  4. compatible attributes only, and the name is rare
  5. name string alone -> do not merge

Claude assists on 3 and 4 only, and only ever to *veto*: a "no" blocks a merge
that the rules allowed. It cannot promote a merge the rules refused.

Deviation worth knowing: the spec blocks merges on attributes that conflict over
an overlapping period. Public pages rarely date their claims, so this build
treats a conflict as blocking for rungs 2-4 but lets rung 1 (same canonical
profile URL) through — one personal site is stronger evidence than an undated
employer mismatch, which is usually just a job change.
"""

from __future__ import annotations

from typing import Optional

from artemis.config import Settings
from artemis.extract.client import ClaudeClient
from artemis.graph.store import GraphStore
from artemis.identity.normalize import could_be_same_name, fold, name_key
from artemis.models import (
    Endpoint,
    MergeBasis,
    MergeDecision,
    MergeDecisionKind,
    Node,
    Observation,
)
from artemis.runtime import JobLog

_ROLE_MARKERS = (
    "ceo", "cto", "cfo", "coo", "chief", "founder", "cofounder", "co-founder",
    "president", "chair", "chairman", "director", "professor", "lecturer",
    "engineer", "scientist", "researcher", "partner", "principal", "head of",
    "manager", "editor", "surgeon", "physician", "cardiologist", "attorney",
    "analyst", "advisor", "trustee", "dean", "provost", "curator", "designer",
)
_INSTITUTION_MARKERS = (
    "university", "college", "institute", "school", "hospital", "academy",
    "laboratory", "labs", "clinic", "foundation", "museum",
)
_FIELD_MARKERS = (
    "cardiology", "oncology", "neurology", "physics", "chemistry", "biology",
    "economics", "law", "medicine", "engineering", "machine learning",
    "artificial intelligence", "robotics", "genomics", "climate",
)
# Keys where two different values are treated as a conflict.
_EXCLUSIVE_KEYS = ("employer", "institution")

#: Words that can sit inside an organisation's name in lower case without the
#: name stopping being a proper noun: "Bank of England", "Sidley Austin LLP".
_ORG_FUNCTION_WORDS = frozenset(
    {"of", "the", "and", "for", "at", "in", "on", "de", "du", "la", "le", "von",
     "van", "der", "den", "&", "-", "y", "el"}
)

#: Capitalised things that are emphatically not employers. Each of these was
#: observed in the `employer` bucket on a real job, where — being an exclusive
#: key — it declared two observations of one person to be two people.
#: Political affiliation and rich-lists are the two that recur.
_NOT_EMPLOYERS = frozenset(
    {"republican", "republicans", "democrat", "democrats", "democratic",
     "independent", "conservative", "conservatives", "labour", "labor", "tory",
     "tories", "gop", "libertarian", "green party",
     "forbes 400", "fortune 500", "time 100", "forbes", "fortune"}
)

#: An organisation's name is a name, not a sentence. "hosting hit reality show
#: The Apprentice" arrived as an employer because it contains a capital letter.
_MAX_ORG_TOKENS = 6


def looks_like_org(value: str) -> bool:
    """Could this string be the name of a place someone works?

    ``_nameable`` used to answer this with ``any(c.isupper() for c in v)``,
    which its own docstring did not describe: an organisation is a proper noun
    and is capitalised *wherever* it is written, not merely somewhere. One
    capital anywhere let a six-word verb phrase through.

    So every content token has to be capitalised (function words inside a name
    may be lower case), the whole thing has to be short enough to be a name,
    and a small set of capitalised non-employers is refused outright.
    """
    tokens = value.split()
    if not tokens or len(tokens) > _MAX_ORG_TOKENS:
        return False
    if fold(value).strip() in _NOT_EMPLOYERS:
        return False
    content = [t for t in tokens if fold(t) not in _ORG_FUNCTION_WORDS]
    if not content:
        return False
    # Digits are fine inside a name ("Section 9", "3M"); they just cannot be
    # the thing that makes it look like a proper noun.
    return all(t[0].isupper() or t[0].isdigit() for t in content) and any(
        c.isupper() for c in value
    )


def classify_attribute(value: str) -> str:
    """Bucket a free-text attribute into one of the Node.attributes keys.

    The fallthrough used to be ``employer``, which is one of two
    ``_EXCLUSIVE_KEYS`` — so anything the markers failed to recognise landed in
    the bucket with the most power to declare two people different. 'son',
    'Republican' and 'billionaire businessman' all became employers, and since
    they do not overlap each other, every pair of observations positively
    contradicted every other. That is what split Donald Trump into four
    readings and aborted the job before it searched.

    Unrecognised text now falls to ``other``, which is in no exclusive key and
    in no compatibility check: it is recorded, and it gets no vote on identity.
    """
    v = fold(value)
    if any(m in v for m in _ROLE_MARKERS):
        return "role"
    if any(m in v for m in _INSTITUTION_MARKERS):
        return "institution"
    if any(m in v for m in _FIELD_MARKERS):
        return "field"
    return "employer" if looks_like_org(value) else "other"


def bucket_attributes(values: list[str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for value in values:
        cleaned = value.strip()
        if cleaned:
            out.setdefault(classify_attribute(cleaned), set()).add(cleaned)
    return out


def _values(attrs: dict[str, set[str]], key: str) -> set[str]:
    return {fold(v) for v in attrs.get(key, set())}


def _compact(value: str) -> str:
    """Comparison form: letters and digits only.

    "Pantheon Prep" and "PantheonPrep" are the same employer written two ways —
    a page title against a directory listing. Compared literally they look like
    a conflict, which blocked the two real Abhimanyu Sharma nodes from merging
    and left his colleagues stranded on a node the search never reached.
    """
    return "".join(c for c in value if c.isalnum())


def _overlaps(left: set[str], right: set[str]) -> bool:
    for a in left:
        for b in right:
            ca, cb = _compact(a), _compact(b)
            if not ca or not cb:
                continue
            if ca == cb or (len(ca) > 3 and len(cb) > 3 and (ca in cb or cb in ca)):
                return True
    return False


def _nameable(values: set[str]) -> set[str]:
    """Values that could actually be an organisation's name.

    A stray descriptor in an exclusive key becomes a merge-blocking "conflict"
    — "subordinate" versus "PantheonPrep" blocked a real merge this way. An
    organisation is a proper noun and is capitalised wherever it is written, so
    a bare lowercase word is a description of a role, not the name of a place
    to work, and gets no vote on whether two people are different.

    Still applied even though ``classify_attribute`` now keeps non-names out of
    ``employer``: ``institution`` is the other exclusive key, and it is filled
    by keyword match ("the clinic", "a teaching hospital") rather than by shape.
    """
    return {v for v in values if looks_like_org(v)}


def attributes_conflict(a: dict[str, set[str]], b: dict[str, set[str]]) -> Optional[str]:
    """Mutually exclusive values on the same key. Returns the offending key.

    Only genuine organisation names can conflict. This keeps the real signal —
    "Linguistics" (a JNU academic) against "PantheonPrep" still blocks, because
    those are two different people — while dropping the noise.
    """
    for key in _EXCLUSIVE_KEYS:
        left = _nameable(a.get(key, set()))
        right = _nameable(b.get(key, set()))
        if left and right and not _overlaps({fold(v) for v in left}, {fold(v) for v in right}):
            return key
    return None


def attributes_inconsistent(a: dict[str, set[str]], b: dict[str, set[str]]) -> bool:
    """Do these two observations positively contradict each other?

    Absence of shared attributes is NOT inconsistency. A page that names someone
    with no context around the mention is uninformative, not evidence of a
    second human — treating it as such split Donald Trump into four people, two
    of them with no attributes at all.

    Use this for *clustering* (how many distinct people have we seen), and
    `attributes_compatible` for *merging* (may these two be joined), which
    rightly demands positive evidence.
    """
    return attributes_conflict(a, b) is not None


def attributes_compatible(a: dict[str, set[str]], b: dict[str, set[str]]) -> bool:
    """At least one genuinely corresponding attribute, and no conflict."""
    if attributes_conflict(a, b):
        return False
    for key in ("employer", "institution", "field", "role", "location"):
        if _overlaps(_values(a, key), _values(b, key)):
            return True
    return False


class IdentityResolver:
    def __init__(
        self,
        store: GraphStore,
        settings: Settings,
        claude: ClaudeClient,
        log: JobLog,
    ) -> None:
        self.store = store
        self.s = settings
        self.claude = claude
        self.log = log
        self.merges = 0
        self.merges_blocked = 0
        self.decisions: list[MergeDecision] = []

    # -- rarity -------------------------------------------------------------
    def cluster_count(self, name: str) -> int:
        """How many mutually inconsistent people we have already seen under this name.

        Estimated from our own results, not a name-frequency table.
        """
        nodes = self.store.candidates_for(name)
        clusters: list[Node] = []
        for node in nodes:
            if all(
                attributes_inconsistent(node.attributes, c.attributes) for c in clusters
            ):
                clusters.append(node)
        return max(len(clusters), 1 if nodes else 0)

    # -- the ladder ---------------------------------------------------------
    async def resolve(
        self,
        name: str,
        observation: Observation,
        *,
        canonical_urls: Optional[set[str]] = None,
        endpoint: Optional[Endpoint] = None,
        depth: int = 0,
        page_url: Optional[str] = None,
        provenance_name: Optional[str] = None,
    ) -> Node:
        """Attach an observation to an existing node, or open a new one.

        `provenance_name` is the person the query that surfaced this page was
        about — search-engine provenance, used by rung 2b.
        """
        canonical_urls = canonical_urls or set()
        incoming_attrs = observation.attributes

        for candidate in self.store.candidates_for(name):
            if not could_be_same_name(name, candidate.display_name) and not any(
                could_be_same_name(name, v) for v in candidate.name_variants
            ):
                continue

            basis, allowed, detail, assisted = await self._evaluate(
                candidate, name, incoming_attrs, canonical_urls, page_url, provenance_name
            )
            decision = MergeDecision(
                decision=MergeDecisionKind.MERGED if allowed else MergeDecisionKind.BLOCKED,
                basis=basis,
                name=name,
                left_node_id=candidate.node_id,
                right_node_id="(new observation)",
                detail=detail,
                claude_assisted=assisted,
            )
            self.decisions.append(decision)
            self.log(
                "merge.decided",
                f"{'merged' if allowed else 'held separate'}: {name} ({basis.value})",
                node_id=candidate.node_id,
                basis=basis.value,
                detail=detail,
                claude_assisted=assisted,
            )

            if allowed:
                self.merges += 1
                self.store.record_observation(candidate.node_id, observation)
                for url in canonical_urls:
                    self.store.add_canonical_url(candidate.node_id, url)
                candidate.name_variants.add(name)
                if candidate.endpoint is None:
                    candidate.endpoint = endpoint
                return candidate
            self.merges_blocked += 1

        node = self.store.add_node(
            name,
            endpoint=endpoint,
            depth=depth,
            observation=observation,
            canonical_urls=canonical_urls,
        )
        self.log("node.created", name, node_id=node.node_id, url=observation.url,
                 name_key=name_key(name))
        return node

    async def _evaluate(
        self,
        candidate: Node,
        name: str,
        attrs: dict[str, set[str]],
        canonical_urls: set[str],
        page_url: Optional[str],
        provenance_name: Optional[str] = None,
    ) -> tuple[MergeBasis, bool, str, bool]:
        # Rung 1 — same canonical profile URL.
        shared_urls = canonical_urls & candidate.canonical_urls
        if shared_urls:
            return (
                MergeBasis.SAME_CANONICAL_URL,
                True,
                f"shared canonical url {sorted(shared_urls)[0]}",
                False,
            )

        conflict_key = attributes_conflict(candidate.attributes, attrs)
        if conflict_key:
            left = sorted(candidate.attributes.get(conflict_key, set()))
            right = sorted(attrs.get(conflict_key, set()))
            return (
                MergeBasis.CONFLICTING_ATTRIBUTES,
                False,
                f"{conflict_key} conflict: {left} vs {right}",
                False,
            )

        # Rung 2 — co-occurrence on one page as a single referent.
        if page_url and any(o.url == page_url for o in candidate.observations):
            return (
                MergeBasis.SINGLE_REFERENT_CO_OCCURRENCE,
                True,
                f"same name, same page: {page_url}",
                False,
            )

        # Rung 2b — search provenance. This page came back from a query that
        # named this person; the search engine matching it to them is evidence
        # about who it is about, on top of the name string.
        #
        # Without this, an endpoint fragments: every page mentioning them with
        # no nearby attributes falls to rung 5 and becomes an isolated node
        # holding one or two of their edges. Drew Glover split into a dozen
        # nodes this way and the path search never saw most of his network.
        if provenance_name and could_be_same_name(provenance_name, name):
            return (
                MergeBasis.SEARCH_PROVENANCE,
                True,
                f"page returned by a query naming {provenance_name!r}",
                False,
            )

        compatible = attributes_compatible(candidate.attributes, attrs)
        clusters = self.cluster_count(name)

        # Rung 3 — compatible attributes + a shared neighbour.
        if compatible and self._shares_neighbor(candidate, attrs):
            allowed, assisted, note = await self._claude_veto(candidate, name, attrs)
            return (
                MergeBasis.ATTRIBUTES_PLUS_SHARED_NEIGHBOR,
                allowed,
                f"compatible attributes and a shared neighbour{note}",
                assisted,
            )

        # Rung 4 — compatible attributes only, and the name is rare.
        if compatible:
            if clusters >= self.s.common_name_cluster_threshold:
                return (
                    MergeBasis.NAME_TOO_COMMON,
                    False,
                    f"{clusters} inconsistent clusters seen for this name; "
                    "attribute-only merge refused",
                    False,
                )
            allowed, assisted, note = await self._claude_veto(candidate, name, attrs)
            return (
                MergeBasis.RARE_NAME_ATTRIBUTES,
                allowed,
                f"compatible attributes, rare name ({clusters} cluster(s)){note}",
                assisted,
            )

        # Rung 5 — nothing but the name string.
        return (MergeBasis.NAME_STRING_ONLY, False, "name string alone is not evidence", False)

    def _shares_neighbor(self, candidate: Node, attrs: dict[str, set[str]]) -> bool:
        """Weak proxy: the candidate has a graph neighbour whose org matches ours."""
        incoming = set()
        for key in ("employer", "institution"):
            incoming |= _values(attrs, key)
        if not incoming:
            return False
        for neighbor_id, _edge_id in self.store.neighbors(candidate.node_id):
            neighbor = self.store.get(neighbor_id)
            if neighbor is None:
                continue
            theirs = _values(neighbor.attributes, "employer") | _values(
                neighbor.attributes, "institution"
            )
            if _overlaps(incoming, theirs):
                return True
        return False

    async def _claude_veto(
        self, candidate: Node, name: str, attrs: dict[str, set[str]]
    ) -> tuple[bool, bool, str]:
        """Advisory second opinion on rungs 3 and 4. Only a 'no' changes the outcome."""
        if not self.claude.enabled:
            return True, False, " (no adjudication available)"
        left = {
            "attributes": {k: sorted(v) for k, v in candidate.attributes.items()},
            "spans": [o.span_text[:300] for o in candidate.observations[:3]],
            "urls": sorted({o.url for o in candidate.observations})[:3],
        }
        right = {"attributes": {k: sorted(v) for k, v in attrs.items()}}
        verdict = await self.claude.adjudicate_identity(name, left, right)
        if verdict.same_person == "no":
            return False, True, f" — adjudicator disagreed: {verdict.reason}"
        return True, True, f" — adjudicator: {verdict.same_person}"
