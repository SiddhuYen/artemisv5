"""Serper implementation: batched POST, credit accounting, disk cache, backoff.

Serper accepts a JSON array of query objects in one request; the credit
accounting is the same as issuing them separately, so batching is a free
round-trip saving. Credits are a first-class budget: every query is counted and
the provider hard-stops at the ceiling.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

import httpx

from artemis.config import Settings
from artemis.runtime import BudgetLedger, JobLog
from artemis.scrape.cache import DiskCache
from artemis.search.base import Query, SearchHit, SearchResults
from artemis.search.templates import normalize_query

_RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class SerperUnavailable(RuntimeError):
    """Total provider failure. Serper is the only door to the web; the job fails."""


class SerperProvider:
    def __init__(
        self,
        settings: Settings,
        cache: DiskCache,
        ledger: BudgetLedger,
        log: JobLog,
        *,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.s = settings
        self.cache = cache
        self.ledger = ledger
        self.log = log
        self._client = client
        self._owns_client = client is None
        self.calls_attempted = 0
        self.calls_failed = 0

    async def __aenter__(self) -> "SerperProvider":
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- public -------------------------------------------------------------
    async def search(self, queries: list[Query]) -> list[SearchResults]:
        if not queries:
            return []
        if not self.s.serper_api_key:
            raise SerperUnavailable("no Serper API key configured")

        results: list[Optional[SearchResults]] = [None] * len(queries)
        pending: list[tuple[int, Query]] = []

        for i, q in enumerate(queries):
            cached = await self.cache.get("serper", self._cache_key(q))
            if cached is not None:
                results[i] = self._parse(cached, q, from_cache=True)
                self.log("query.cached", q.rendered, template=q.template.value,
                         hits=len(results[i].hits))  # type: ignore[union-attr]
            else:
                pending.append((i, q))

        for start in range(0, len(pending), self.s.serper_batch_size):
            chunk = pending[start : start + self.s.serper_batch_size]
            affordable: list[tuple[int, Query]] = []
            for idx, q in chunk:
                if self.ledger.try_spend_serper(1):
                    affordable.append((idx, q))
                else:
                    results[idx] = SearchResults(query=q, error="budget: max_serper_credits")
            if not affordable:
                continue

            payloads = await self._post([q for _, q in affordable])
            for (idx, q), payload in zip(affordable, payloads):
                if payload is None:
                    results[idx] = SearchResults(query=q, error="serper request failed")
                    continue
                await self.cache.set("serper", self._cache_key(q), payload)
                parsed = self._parse(payload, q, from_cache=False)
                results[idx] = parsed
                self.log(
                    "query.issued",
                    q.rendered,
                    template=q.template.value,
                    node_id=q.node_id,
                    hits=len(parsed.hits),
                    credits_used=self.ledger.serper_credits_used,
                )

        if self.calls_attempted and self.calls_failed == self.calls_attempted:
            raise SerperUnavailable(
                f"all {self.calls_attempted} Serper requests failed; no other route to the web"
            )

        return [r if r is not None else SearchResults(query=q, error="not executed")
                for r, q in zip(results, queries)]

    # -- internals ----------------------------------------------------------
    def _cache_key(self, q: Query) -> str:
        return "|".join(
            [
                normalize_query(q.rendered),
                str(self.s.serper_results_per_query),
                self.s.serper_gl,
                self.s.serper_hl,
            ]
        )

    def _body(self, queries: list[Query]) -> Any:
        items = [
            {
                "q": q.rendered,
                "num": self.s.serper_results_per_query,
                "gl": self.s.serper_gl,
                "hl": self.s.serper_hl,
            }
            for q in queries
        ]
        return items if len(items) > 1 else items[0]

    async def _post(self, queries: list[Query]) -> list[Optional[dict]]:
        assert self._client is not None
        headers = {
            "X-API-KEY": self.s.serper_api_key or "",
            "Content-Type": "application/json",
        }
        delay = 1.0
        for attempt in range(self.s.serper_max_retries + 1):
            self.calls_attempted += 1
            try:
                resp = await self._client.post(
                    self.s.serper_endpoint, headers=headers, json=self._body(queries)
                )
            except Exception as exc:
                self.calls_failed += 1
                self.log.warn("serper.error", f"{type(exc).__name__}: {exc}", attempt=attempt)
                if attempt >= self.s.serper_max_retries:
                    return [None] * len(queries)
                await asyncio.sleep(delay)
                delay *= 2
                continue

            if resp.status_code in _RETRY_STATUS and attempt < self.s.serper_max_retries:
                self.calls_failed += 1
                self.log.warn("serper.retry", f"HTTP {resp.status_code}", attempt=attempt)
                await asyncio.sleep(delay)
                delay *= 2
                continue

            if resp.status_code >= 400:
                self.calls_failed += 1
                self.log.warn("serper.failed", f"HTTP {resp.status_code}: {resp.text[:200]}")
                return [None] * len(queries)

            try:
                data = resp.json()
            except json.JSONDecodeError:
                self.calls_failed += 1
                self.log.warn("serper.failed", "response was not JSON")
                return [None] * len(queries)

            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list) or len(data) != len(queries):
                # Shape drift: keep whatever aligns, drop the rest.
                self.log.warn(
                    "serper.shape",
                    f"expected {len(queries)} results, got {type(data).__name__} "
                    f"len={len(data) if isinstance(data, list) else 'n/a'}",
                )
                data = (data if isinstance(data, list) else [])[: len(queries)]
                data += [None] * (len(queries) - len(data))
            return data
        return [None] * len(queries)

    def _parse(self, payload: dict, query: Query, *, from_cache: bool) -> SearchResults:
        organic = payload.get("organic") or []
        hits: list[SearchHit] = []
        for item in organic:
            link = (item or {}).get("link")
            if not link:
                continue
            hits.append(
                SearchHit(
                    title=item.get("title") or "",
                    link=link,
                    snippet=item.get("snippet") or "",
                    position=item.get("position"),
                )
            )
        return SearchResults(
            query=query,
            hits=hits,
            knowledge_graph=payload.get("knowledgeGraph"),
            credits_used=0 if from_cache else 1,
            from_cache=from_cache,
        )
