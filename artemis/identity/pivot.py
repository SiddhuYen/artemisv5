"""Route-time pivot verification: is url1's P the same human as url2's P?

Runs on candidate routes only, right before returning, because this is the only
place an identity failure actually costs anything. Every hop can be individually
real and the route still fiction — hop 2's Jane Smith and hop 3's Jane Smith
being different people is not a hypothetical, it is the default outcome of
name-as-identity.

This is cross-document coreference without a knowledge base to link against:
clustering, not linking. It is not solved, it is managed — so the basis is
reported rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from artemis.config import Settings
from artemis.extract.client import ClaudeClient
from artemis.graph.store import GraphStore
from artemis.identity.resolve import _overlaps, _values
from artemis.models import IdentityBasis, Node
from artemis.runtime import JobLog


@dataclass
class PivotVerdict:
    node_id: str
    name: str
    basis: IdentityBasis
    detail: str


class PivotVerifier:
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

    async def verify(
        self,
        node_id: str,
        *,
        url_in: str,
        url_out: str,
        prev_node_id: Optional[str] = None,
        next_node_id: Optional[str] = None,
    ) -> PivotVerdict:
        node = self.store.get(node_id)
        if node is None:
            return PivotVerdict(node_id, "?", IdentityBasis.NAME_ONLY, "node missing")

        # shared_page — one page carries the pivot alongside both neighbours.
        if url_in == url_out:
            return self._done(node, IdentityBasis.SHARED_PAGE,
                              f"both hops grounded on {url_in}")
        shared = self._page_with_all(node_id, prev_node_id, next_node_id)
        if shared:
            return self._done(node, IdentityBasis.SHARED_PAGE,
                              f"{shared} mentions the pivot with both neighbours")

        # canonical_url — both sides grounded on this person's own profile pages.
        if url_in in node.canonical_urls and url_out in node.canonical_urls:
            return self._done(node, IdentityBasis.CANONICAL_URL,
                              "both sources are canonical profile pages for this person")

        # attribute_match — employer/institution/field consistent across sources.
        left = self._attrs_on(node, url_in)
        right = self._attrs_on(node, url_out)
        matched = self._matching_keys(left, right)
        if matched:
            return self._done(node, IdentityBasis.ATTRIBUTE_MATCH,
                              f"consistent across sources on {', '.join(matched)}")

        # name_only — ask the stronger model; it can lift this to attribute_match.
        if self.claude.enabled:
            verdict = await self.claude.verify_pivot(
                node.display_name,
                {"url": url_in, "attributes": {k: sorted(v) for k, v in left.items()},
                 "spans": self._spans_on(node, url_in)},
                {"url": url_out, "attributes": {k: sorted(v) for k, v in right.items()},
                 "spans": self._spans_on(node, url_out)},
            )
            if verdict.is_yes:
                return self._done(node, IdentityBasis.ATTRIBUTE_MATCH,
                                  f"adjudicated same person: {verdict.reason}")
            return self._done(node, IdentityBasis.NAME_ONLY,
                              f"adjudicator said {verdict.same_person}: {verdict.reason}")

        return self._done(node, IdentityBasis.NAME_ONLY,
                          "nothing but a matching name string links the two sources")

    # -- helpers ------------------------------------------------------------
    def _done(self, node: Node, basis: IdentityBasis, detail: str) -> PivotVerdict:
        self.log("pivot.verified", f"{node.display_name}: {basis.value}",
                 node_id=node.node_id, basis=basis.value, detail=detail)
        return PivotVerdict(node.node_id, node.display_name, basis, detail)

    def _page_with_all(
        self, node_id: str, prev_id: Optional[str], next_id: Optional[str]
    ) -> Optional[str]:
        if not prev_id or not next_id:
            return None
        sets = []
        for nid in (node_id, prev_id, next_id):
            n = self.store.get(nid)
            if n is None:
                return None
            sets.append({o.url for o in n.observations})
        common = set.intersection(*sets)
        return sorted(common)[0] if common else None

    def _attrs_on(self, node: Node, url: str) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for obs in node.observations:
            if obs.url != url:
                continue
            for key, values in obs.attributes.items():
                out.setdefault(key, set()).update(values)
        return out

    def _spans_on(self, node: Node, url: str) -> list[str]:
        return [o.span_text[:300] for o in node.observations if o.url == url][:3]

    def _matching_keys(self, a: dict[str, set[str]], b: dict[str, set[str]]) -> list[str]:
        return [
            key
            for key in ("employer", "institution", "field", "role", "location")
            if _overlaps(_values(a, key), _values(b, key))
        ]
