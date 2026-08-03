"""SearchProvider protocol and the wire models the search layer speaks.

These types stay inside the search layer; the graph layer consumes URLs and
titles, never provider-shaped objects.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

from pydantic import BaseModel, Field

from artemis.search.templates import QueryTemplate


class Query(BaseModel):
    template: QueryTemplate
    rendered: str
    subject_name: str
    #: Node this query was issued on behalf of, for the job log.
    node_id: Optional[str] = None


class SearchHit(BaseModel):
    title: str = ""
    link: str
    snippet: str = ""
    position: Optional[int] = None


class SearchResults(BaseModel):
    query: Query
    hits: list[SearchHit] = Field(default_factory=list)
    #: Serper's knowledgeGraph block when present — a cheap identity signal.
    knowledge_graph: Optional[dict[str, Any]] = None
    credits_used: int = 0
    from_cache: bool = False
    error: Optional[str] = None


class SearchProvider(Protocol):
    async def search(self, queries: list[Query]) -> list[SearchResults]: ...
