"""Nodes, edges, observations.

The store is dumb on purpose: it holds structure and does no identity
reasoning. Deciding that two observations are one person is resolve.py's job;
the store only executes the merge it is told to perform.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime
from typing import Iterable, Optional

from artemis.identity.normalize import name_key, normalize_name, variants
from artemis.models import Edge, Endpoint, Extraction, Node, Observation


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class GraphStore:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}
        self._incident: dict[str, set[str]] = defaultdict(set)  # node_id -> edge_ids
        self._by_name_key: dict[str, list[str]] = defaultdict(list)
        self._alias: dict[str, str] = {}  # merged-away node_id -> surviving node_id

    # -- lookup -------------------------------------------------------------
    def resolve_id(self, node_id: str) -> str:
        seen: set[str] = set()
        while node_id in self._alias and node_id not in seen:
            seen.add(node_id)
            node_id = self._alias[node_id]
        return node_id

    def get(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(self.resolve_id(node_id))

    def candidates_for(self, name: str) -> list[Node]:
        """Nodes whose name *could* denote the same person. Not a merge decision."""
        key = name_key(name)
        return [self.nodes[nid] for nid in self._by_name_key.get(key, []) if nid in self.nodes]

    # -- mutation -----------------------------------------------------------
    def add_node(
        self,
        display_name: str,
        *,
        endpoint: Optional[Endpoint] = None,
        depth: int = 0,
        observation: Optional[Observation] = None,
        canonical_urls: Iterable[str] = (),
    ) -> Node:
        node = Node(
            node_id=_new_id("n"),
            display_name=display_name.strip(),
            name_variants=variants(display_name),
            endpoint=endpoint,
            discovered_at_depth=depth,
            canonical_urls=set(canonical_urls),
        )
        self.nodes[node.node_id] = node
        self._by_name_key[name_key(display_name)].append(node.node_id)
        if observation is not None:
            self.record_observation(node.node_id, observation)
        return node

    def record_observation(self, node_id: str, observation: Observation) -> None:
        node = self.nodes[self.resolve_id(node_id)]
        node.observations.append(observation)
        for key, values in observation.attributes.items():
            node.attributes.setdefault(key, set()).update(values)

    def add_canonical_url(self, node_id: str, url: str) -> None:
        self.nodes[self.resolve_id(node_id)].canonical_urls.add(url)

    def merge(self, keep_id: str, drop_id: str) -> Node:
        """Fold `drop` into `keep`. Callers have already justified this."""
        keep_id, drop_id = self.resolve_id(keep_id), self.resolve_id(drop_id)
        if keep_id == drop_id:
            return self.nodes[keep_id]

        keep, drop = self.nodes[keep_id], self.nodes[drop_id]
        keep.name_variants |= drop.name_variants | {normalize_name(drop.display_name)}
        keep.canonical_urls |= drop.canonical_urls
        keep.observations.extend(drop.observations)
        for key, values in drop.attributes.items():
            keep.attributes.setdefault(key, set()).update(values)
        if keep.endpoint is None:
            keep.endpoint = drop.endpoint
        keep.discovered_at_depth = min(keep.discovered_at_depth, drop.discovered_at_depth)

        for edge_id in list(self._incident.get(drop_id, ())):
            edge = self.edges[edge_id]
            if edge.subject_node_id == drop_id:
                edge.subject_node_id = keep_id
            if edge.object_node_id == drop_id:
                edge.object_node_id = keep_id
            self._incident[keep_id].add(edge_id)
        self._incident.pop(drop_id, None)

        del self.nodes[drop_id]
        self._alias[drop_id] = keep_id
        return keep

    def add_edge(
        self,
        subject_node_id: str,
        object_node_id: str,
        extraction: Extraction,
        *,
        source_url: str,
        source_title: Optional[str],
        retrieved_at: datetime,
    ) -> Optional[Edge]:
        subject_node_id = self.resolve_id(subject_node_id)
        object_node_id = self.resolve_id(object_node_id)
        if subject_node_id == object_node_id:
            return None  # a merge turned this into a self-loop

        fingerprint = (subject_node_id, object_node_id, source_url,
                       extraction.span_start, extraction.span_end)
        for edge_id in self._incident.get(subject_node_id, ()):
            existing = self.edges[edge_id]
            if (existing.subject_node_id, existing.object_node_id, existing.source_url,
                    existing.extraction.span_start, existing.extraction.span_end) == fingerprint:
                return existing

        edge = Edge(
            edge_id=_new_id("e"),
            subject_node_id=subject_node_id,
            object_node_id=object_node_id,
            extraction=extraction,
            source_url=source_url,
            source_title=source_title,
            retrieved_at=retrieved_at,
        )
        self.edges[edge.edge_id] = edge
        self._incident[subject_node_id].add(edge.edge_id)
        self._incident[object_node_id].add(edge.edge_id)
        return edge

    # -- traversal ----------------------------------------------------------
    def neighbors(self, node_id: str) -> list[tuple[str, str]]:
        """[(neighbour_node_id, edge_id)] — edges traverse in both directions."""
        node_id = self.resolve_id(node_id)
        out: list[tuple[str, str]] = []
        for edge_id in self._incident.get(node_id, ()):
            edge = self.edges[edge_id]
            other = edge.object_node_id if edge.subject_node_id == node_id else edge.subject_node_id
            if other != node_id:
                out.append((other, edge_id))
        return out

    def observations_on(self, node_id: str, url: str) -> list[Observation]:
        node = self.get(node_id)
        if node is None:
            return []
        return [o for o in node.observations if o.url == url]
