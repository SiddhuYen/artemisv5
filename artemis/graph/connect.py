"""Bidirectional BFS orchestration and path reconstruction.

Symmetric expansion from both endpoints, meeting in the middle. After each
level the two frontiers are checked for intersection; on a hit we run exactly
one more level to collect alternates, then return up to K distinct routes
sorted by hop count.

Parent pointers are recomputed by a fresh BFS over the whole store after each
level rather than threaded through discovery. Edges arrive out of order — a page
fetched for X routinely asserts a relationship between two people who are not X
— so recomputing is both simpler and actually correct about shortest paths.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence

from artemis.config import Settings
from artemis.extract.client import ClaudeClient
from artemis.graph.expand import ExpansionPolicy, SymmetricPolicy
from artemis.graph.store import GraphStore
from artemis.identity.normalize import could_be_same_name, fold, looks_canonical_for
from artemis.identity.pivot import PivotVerdict, PivotVerifier
from artemis.identity.resolve import (
    IdentityResolver,
    attributes_inconsistent,
    bucket_attributes,
)
from artemis.providers import build_providers
from artemis.models import (
    CO_MEMBERSHIP_BASES,
    WARN_CO_LISTING_HOP,
    WARN_NAME_ONLY_PIVOT,
    WARN_NO_REFERENT_RESOLUTION,
    ConnectRequest,
    DisambiguationCandidate,
    Endpoint,
    Hop,
    HopEndpoint,
    IdentityBasis,
    Node,
    Observation,
    Result,
    Route,
    utcnow,
    weakest_basis,
)
from artemis.graph.relations import RelationCache
from artemis.runtime import BudgetLedger, JobLog
from artemis.scrape.fetcher import Fetcher
from artemis.search.base import Query, SearchResults
from artemis.search.serper import SerperProvider, SerperUnavailable


@dataclass
class Traversal:
    dist: dict[str, int] = field(default_factory=dict)
    parent: dict[str, tuple[str, str]] = field(default_factory=dict)  # node -> (prev, edge)


def traverse(store: GraphStore, seed_id: str, max_depth: int) -> Traversal:
    # The seed may have been merged away since it was chosen; follow the alias.
    seed_id = store.resolve_id(seed_id)
    t = Traversal(dist={seed_id: 0})
    queue: deque[str] = deque([seed_id])
    while queue:
        current = queue.popleft()
        if t.dist[current] >= max_depth:
            continue
        for neighbor, edge_id in store.neighbors(current):
            if neighbor not in t.dist:
                t.dist[neighbor] = t.dist[current] + 1
                t.parent[neighbor] = (current, edge_id)
                queue.append(neighbor)
    return t


class Connector:
    def __init__(
        self,
        request: ConnectRequest,
        settings: Settings,
        provider: SerperProvider,
        fetcher: Fetcher,
        claude: ClaudeClient,
        ledger: BudgetLedger,
        log: JobLog,
        *,
        policy: Optional[ExpansionPolicy] = None,
    ) -> None:
        self.req = request
        self.s = settings
        self.provider = provider
        self.fetcher = fetcher
        self.claude = claude
        self.ledger = ledger
        self.log = log
        self.store = GraphStore()
        self.resolver = IdentityResolver(self.store, settings, claude, log)
        self.pivots = PivotVerifier(self.store, settings, claude, log)
        self.policy = policy or SymmetricPolicy(settings)
        self.providers = build_providers(settings)
        self.relations = RelationCache(
            settings.cache_dir,
            enabled=settings.relation_cache_enabled,
            ttl_s=settings.relation_cache_ttl_s,
        )
        self.warnings: list[str] = []
        self._expanded: set[str] = set()
        self._urls_seen: set[str] = set()
        # Set once in run(); each side steers toward the other's notability.
        self._a_is_famous = True
        self._b_is_famous = True

        budget = request.budget
        self.depth_a = (
            (budget.max_depth_a if budget else None)
            or request.max_depth
            or settings.max_depth_a
        )
        self.depth_b = (
            (budget.max_depth_b if budget else None)
            if (budget and budget.max_depth_b is not None)
            else (request.max_depth or settings.max_depth_b)
        )

    def release(self) -> None:
        """Drop the crawl's working set once the run is over.

        Cancelling a job leaves `run()`'s coroutine frame alive for the rest of
        the process — measured on CPython 3.12, and unaffected by dropping the
        Task or clearing tracebacks — and that frame holds `self`. So the caller
        cannot free the graph by releasing its own reference to the connector;
        the connector has to let go of the graph itself. A cancelled run used to
        strand a full GraphStore per job, which is what walked the container
        into its memory ceiling.

        Safe only after run() has returned or raised: the result is already
        built by then, and nothing reads these afterwards.
        """
        self.store = GraphStore()
        self.resolver = None  # type: ignore[assignment]
        self.pivots = None  # type: ignore[assignment]
        self.providers = ()
        self._expanded = set()
        self._urls_seen = set()

    # -- entry point --------------------------------------------------------
    async def run(self) -> Result:
        if not self.claude.enabled:
            self.warnings.append(WARN_NO_REFERENT_RESOLUTION)
            self.log.warn("degraded", "no Anthropic key: strict same-sentence extraction only")

        self.log(
            "job.started",
            f"{self.req.person_a} <-> {self.req.person_b}",
            depth_a=self.depth_a,
            depth_b=self.depth_b,
            structured_providers=[p.name for p in self.providers],
        )

        fame_a, why_a = await self.claude.classify_fame(
            self.req.person_a, self.req.context_a or ""
        )
        fame_b, why_b = await self.claude.classify_fame(
            self.req.person_b, self.req.context_b or ""
        )
        self._a_is_famous = fame_a == "famous"
        self._b_is_famous = fame_b == "famous"
        self.log(
            "notability.classified",
            f"{self.req.person_a}: {fame_a}; {self.req.person_b}: {fame_b}",
            person_a=fame_a, why_a=why_a, person_b=fame_b, why_b=why_b,
            a_expands_toward="famous" if self._b_is_famous else "obscure",
            b_expands_toward="famous" if self._a_is_famous else "obscure",
        )

        seed_a, disambiguation = await self._seed(
            self.req.person_a, self.req.context_a, Endpoint.A, self.req.person_b
        )
        if disambiguation:
            return self._result(found=False, disambiguation=disambiguation)
        seed_b, disambiguation = await self._seed(
            self.req.person_b, self.req.context_b, Endpoint.B, self.req.person_a
        )
        if disambiguation:
            return self._result(found=False, disambiguation=disambiguation)

        if seed_a is None or seed_b is None:
            missing = self.req.person_a if seed_a is None else self.req.person_b
            self.warnings.append(f"no grounded mention of {missing!r} was found on any fetched page")
            return self._result(found=False)

        await self._search_loop(seed_a, seed_b)
        await self._enrich_with_providers(seed_a, seed_b)
        routes = await self._build_routes(seed_a, seed_b)

        if not routes:
            for limit in self.ledger.limits_hit:
                self.warnings.append(f"budget exhausted: {limit}")
            ta = traverse(self.store, seed_a, self.depth_a + 2)
            tb = traverse(self.store, seed_b, self.depth_b + 2)
            self.warnings.append(
                f"frontier A reached depth {max(ta.dist.values(), default=0)} "
                f"across {len(ta.dist)} people; frontier B reached depth "
                f"{max(tb.dist.values(), default=0)} across {len(tb.dist)} people"
            )
        return self._result(found=bool(routes), routes=routes)

    # -- seeding ------------------------------------------------------------
    async def _seed(
        self, name: str, context: Optional[str], endpoint: Endpoint, other: str
    ) -> tuple[Optional[str], Optional[list[DisambiguationCandidate]]]:
        # Anything we already know about this person, replayed before spending
        # anything on the network.
        await self._replay_cached(name)

        probe = Node(node_id="seed-probe", display_name=name)
        queries = self.policy.plan_queries(
            probe, context=context, other_endpoint_name=other, is_seed=True
        )
        results = await self.provider.search(queries)
        await self._ingest(self._urls_from(results))

        matches = [
            n
            for n in self.store.nodes.values()
            if could_be_same_name(name, n.display_name)
            or any(could_be_same_name(name, v) for v in n.name_variants)
        ]
        if not matches:
            return None, None

        clusters: list[list[Node]] = []
        for node in sorted(matches, key=lambda n: -len(n.observations)):
            for cluster in clusters:
                # Only a positive contradiction starts a new reading. A mention
                # carrying no attributes is uninformative, not a second person.
                if not attributes_inconsistent(cluster[0].attributes, node.attributes):
                    cluster.append(node)
                    break
            else:
                clusters.append([node])

        is_famous = self._a_is_famous if endpoint is Endpoint.A else self._b_is_famous

        # A famous endpoint is fused rather than split. When one referent
        # dominates a name in public coverage, same-name observations are
        # overwhelmingly that person, and the fragments are pages describing
        # him in terms too varied to reconcile attribute-wise — "Republican",
        # "The Trump Organization", "hosting The Apprentice". Splitting there
        # does not protect against a homonym; it strands most of his real
        # relationships on nodes the search never reaches.
        #
        # Scoped deliberately to the endpoint, which is the person the caller
        # named. It is NOT extended to interior nodes: a wrongly fused pivot
        # invents a route that does not exist, and BFS returning shortest
        # paths would then prefer it. Downstream, every node still climbs the
        # ordinary merge ladder.
        #
        # And only when no disambiguator was given. A caller who writes
        # "Michael Jordan" + "the Berkeley machine learning professor" has said
        # which reading they mean, and both readings of that name are famous —
        # fusing there would hand them the basketball player's network. Context
        # means honour the context, via _context_fit below.
        if len(clusters) > 1 and is_famous and not context:
            keep = max(matches, key=lambda n: (len(n.observations), len(n.attributes)))
            fused = 0
            for node in matches:
                if self.store.resolve_id(node.node_id) != self.store.resolve_id(keep.node_id):
                    self.store.merge(keep.node_id, node.node_id)
                    fused += 1
            keep = self.store.get(self.store.resolve_id(keep.node_id)) or keep
            self.log.warn(
                "seed.fused_famous",
                f"{name!r} is classified famous: fused {len(clusters)} readings into one seed",
                clusters=len(clusters),
                nodes_fused=fused,
                observations=len(keep.observations),
                endpoint=endpoint.value,
            )
            keep.endpoint = endpoint
            return keep.node_id, None

        # Multiple inconsistent readings and no disambiguator: stop and ask,
        # rather than silently guessing which Michael Chen was meant.
        if len(clusters) > 1 and not context:
            self.log.warn(
                "disambiguation.required",
                f"{len(clusters)} inconsistent readings of {name!r}",
                clusters=len(clusters),
            )
            return None, [self._candidate(c[0]) for c in clusters[:6]]

        best = max(
            matches,
            key=lambda n: (self._context_fit(n, context), len(n.observations), len(n.attributes)),
        )
        best.endpoint = endpoint
        self.log("seed.selected", best.display_name, node_id=best.node_id,
                 endpoint=endpoint.value, observations=len(best.observations))
        return best.node_id, None

    def _context_fit(self, node: Node, context: Optional[str]) -> int:
        """How well a candidate matches the disambiguator the caller supplied.

        The caller said which person they meant. Until now that steered the
        queries but not the choice of node, so with three "Abhimanyu Sharma"
        nodes at one observation each the seed was picked arbitrarily — and
        landed on a university academic instead of the Pantheon Prep one.

        Counts attribute matches and source-URL matches, both on significant
        tokens so "Pantheon Prep" matches pantheonprep.com.
        """
        if not context:
            return 0
        wanted = {t for t in fold(context).replace(",", " ").split() if len(t) > 2}
        if not wanted:
            return 0

        score = 0
        for values in node.attributes.values():
            for value in values:
                if wanted & set(fold(value).split()):
                    score += 2
        for url in {o.url for o in node.observations} | node.canonical_urls:
            folded = fold(url)
            # Strip separators so "Pantheon Prep" matches pantheonprep.com.
            compact = folded.replace("-", "").replace(".", "").replace("/", "")
            if all(w in compact for w in wanted):
                score += 3
            elif any(w in folded for w in wanted):
                score += 1
        return score

    def _candidate(self, node: Node) -> DisambiguationCandidate:
        obs = node.observations[0] if node.observations else None
        return DisambiguationCandidate(
            display_name=node.display_name,
            attributes={k: sorted(v) for k, v in node.attributes.items()},
            source_url=obs.url if obs else "",
            source_title=obs.page_title if obs else None,
            snippet=obs.span_text[:300] if obs else None,
        )

    # -- the search loop ----------------------------------------------------
    async def _search_loop(self, seed_a: str, seed_b: str) -> None:
        allowed_levels = max(self.depth_a, self.depth_b)
        level = 0
        bonus_used = False

        while level < allowed_levels:
            if self.ledger.out_of_time():
                self.log.warn("budget", "wall clock exhausted")
                break

            frontier: list[tuple[str, Endpoint]] = []
            ta = traverse(self.store, seed_a, self.depth_a)
            tb = traverse(self.store, seed_b, self.depth_b)
            # Each side steers toward the fame level of the person it is trying
            # to reach: A is hunting B, so it follows B's notability, and vice
            # versa.
            if level < self.depth_a:
                frontier += [
                    (n, Endpoint.A)
                    for n in self._frontier(ta, level, toward_famous=self._b_is_famous)
                ]
            if level < self.depth_b:
                frontier += [
                    (n, Endpoint.B)
                    for n in self._frontier(tb, level, toward_famous=self._a_is_famous)
                ]

            frontier = [(n, e) for n, e in frontier if n not in self._expanded]
            if not frontier:
                self.log("level.empty", f"nothing left to expand at level {level}")
                break

            self.log("level.started", f"level {level}", frontier=len(frontier))
            await self._expand(frontier)

            ta = traverse(self.store, seed_a, self.depth_a + 2)
            tb = traverse(self.store, seed_b, self.depth_b + 2)
            meeting = set(ta.dist) & set(tb.dist)
            if meeting and not bonus_used:
                bonus_used = True
                allowed_levels = min(allowed_levels, level + 2)
                self.log("frontiers.met", f"{len(meeting)} meeting point(s)",
                         at_level=level, continuing_one_more_level=True)
            level += 1

    def _frontier(self, t: Traversal, level: int, *, toward_famous: bool = True) -> list[str]:
        scored = [
            (node_id, len({o.url for o in node.observations}))
            for node_id, depth in t.dist.items()
            if depth == level and (node := self.store.get(node_id)) is not None
        ]
        return self.policy.rank_frontier(
            scored, self.s.frontier_cap_per_level, toward_famous=toward_famous
        )

    async def _expand(self, frontier: list[tuple[str, Endpoint]]) -> None:
        queries: list[Query] = []
        # A node reachable from both sides appears twice in one level's frontier;
        # expanding it twice buys nothing and costs two queries every time.
        seen_this_level: set[str] = set()
        for node_id, endpoint in frontier:
            node = self.store.get(node_id)
            if node is None or node.node_id in seen_this_level:
                continue
            seen_this_level.add(node.node_id)
            if not self.ledger.try_spend_node():
                self.log.warn("budget", "max_nodes_expanded reached")
                break
            self._expanded.add(node_id)
            await self._replay_cached(node.display_name)

            # Which angle of this person's network is most likely to bridge to
            # the far endpoint? The model picks one of a fixed enum; it never
            # writes query text, so a wrong pick only wastes a query.
            target = self.req.person_b if endpoint is Endpoint.A else self.req.person_a
            angle, why = await self.claude.choose_angle(
                node.display_name,
                target,
                {k: sorted(v) for k, v in node.attributes.items()},
            )
            self.log("strategy.angle", f"{node.display_name}: {angle}",
                     node_id=node_id, angle=angle, why=why)
            queries += self.policy.plan_queries(node, angle=angle)

        if not queries:
            return
        results = await self.provider.search(queries)
        await self._ingest(self._urls_from(results))

    async def _enrich_with_providers(self, seed_a: str, seed_b: str) -> None:
        """Structured sources, run once the web crawl has settled.

        Deliberately a second phase rather than interleaved. The web pass
        establishes who is actually in play and what organisations they belong
        to; only then is it worth asking a filings index or an academic graph
        about them. Interleaved, these fired on every expanded node with any
        org-shaped attribute and pulled in dozens of tangential documents before
        the graph knew what mattered.

        Everything discovered here goes through the ordinary fetch -> extract ->
        ground path, so provider-sourced edges carry the same verbatim spans.
        """
        if not self.providers:
            return
        if self.ledger.out_of_time():
            return

        ta = traverse(self.store, seed_a, self.depth_a + 2)
        tb = traverse(self.store, seed_b, self.depth_b + 2)
        # Closest to an endpoint first — that is where a new document can still
        # change whether the two sides meet.
        ranked = sorted(
            set(ta.dist) | set(tb.dist),
            key=lambda n: min(ta.dist.get(n, 99), tb.dist.get(n, 99)),
        )

        targets: list[tuple[str, list[str]]] = []
        for node_id in ranked[: self.s.provider_people]:
            node = self.store.get(node_id)
            if node is None:
                continue
            orgs = [
                o
                for o in sorted(node.attributes.get("employer", set()))
                + sorted(node.attributes.get("institution", set()))
                if len(o.split()) >= 2
            ]
            targets.append((node.display_name, orgs[:2]))

        self.log("providers.started", f"{len(targets)} people across "
                 f"{len(self.providers)} providers",
                 providers=[p.name for p in self.providers])

        # Fan out across people. Each provider already paces itself internally
        # (SEC fair-access sleeps, OpenAlex politeness), so the sequential loop
        # this replaces was serialising several hundred independent HTTP round
        # trips — minutes of wall clock on every run for no reason.
        sem = asyncio.Semaphore(self.s.provider_concurrency)

        async def lookup(person: str, orgs: list[str]) -> list[tuple[str, str]]:
            async with sem:
                return await self._structured_urls(person, orgs)

        batches = await asyncio.gather(
            *(lookup(person, orgs) for person, orgs in targets), return_exceptions=True
        )
        urls: list[tuple[str, str]] = []
        for batch in batches:
            if isinstance(batch, BaseException):
                self.log.warn("provider.error", f"{type(batch).__name__}: {batch}")
                continue
            urls += batch

        await self._ingest(urls)
        self.log("providers.finished", f"{len(urls)} documents discovered")

    async def _replay_cached(self, person: str) -> int:
        """Re-enter previously grounded relations for this person, free.

        The evidence replays intact — same span, same URL, original
        `retrieved_at`. Identity does not: each relation goes through this job's
        own merge ladder, so a cached relation about a different person with the
        same name is held apart on this job's evidence, not trusted because it
        was trusted before.
        """
        cached = await self.relations.lookup(person)
        if not cached:
            return 0
        added = 0
        for relation in cached:
            extraction = relation.to_extraction()
            if extraction is None:
                continue
            retrieved = _parse_iso(relation.retrieved_at)
            subject = await self.resolver.resolve(
                extraction.subject_name,
                Observation(
                    url=relation.source_url, page_title=relation.source_title,
                    span_text=extraction.span_text, span_start=extraction.span_start,
                    span_end=extraction.span_end, retrieved_at=retrieved,
                    attributes=bucket_attributes(extraction.subject_attributes),
                ),
                page_url=relation.source_url, provenance_name=person,
            )
            obj = await self.resolver.resolve(
                extraction.object_name,
                Observation(
                    url=relation.source_url, page_title=relation.source_title,
                    span_text=extraction.span_text, span_start=extraction.span_start,
                    span_end=extraction.span_end, retrieved_at=retrieved,
                    attributes=bucket_attributes(extraction.object_attributes),
                ),
                page_url=relation.source_url, provenance_name=person,
            )
            if self.store.add_edge(
                subject.node_id, obj.node_id, extraction,
                source_url=relation.source_url, source_title=relation.source_title,
                retrieved_at=retrieved,
            ) is not None:
                added += 1
        if added:
            self.relations.replayed += added
            self.log("relations.replayed", f"{added} cached relations for {person}",
                     person=person, count=added)
        return added

    async def _structured_urls(
        self, person: str, orgs: Sequence[str]
    ) -> list[tuple[str, str]]:
        """Documents the open web search misses: SEC filings, 990s, registries.

        These are fetched and grounded exactly like a page found on Google — the
        providers only decide what is worth fetching, never what it says.
        """
        urls: list[tuple[str, str]] = []
        for provider in self.providers:
            try:
                found = await provider.discover(person=person, orgs=list(orgs))
            except Exception as exc:
                self.log.warn("provider.error", f"{provider.name}: {type(exc).__name__}: {exc}")
                continue
            for discovery in found[: self.s.provider_urls_per_node]:
                if discovery.url in self._urls_seen:
                    continue
                self._urls_seen.add(discovery.url)
                urls.append((discovery.url, person))
                self.log("provider.discovered", f"{discovery.provider}: {discovery.why}",
                         url=discovery.url, provider=discovery.provider)
        return urls

    def _urls_from(self, results: list[SearchResults]) -> list[tuple[str, str]]:
        """[(url, person the query was about)] — provenance for the merge ladder."""
        urls: list[tuple[str, str]] = []
        for res in results:
            if res.error:
                continue
            for hit in res.hits[: self.s.fetch_top_n_per_node]:
                if hit.link not in self._urls_seen:
                    self._urls_seen.add(hit.link)
                    urls.append((hit.link, res.query.subject_name))
        return urls

    # -- ingestion ----------------------------------------------------------
    async def _ingest(self, urls: list[tuple[str, str]]) -> None:
        if not urls:
            return
        provenance = {url: name for url, name in urls}
        outcomes = await self.fetcher.fetch_many([url for url, _ in urls])
        pages = [o.page for o in outcomes if o.page is not None]
        if not pages:
            return

        endpoints = [self.req.person_a, self.req.person_b]
        sem = asyncio.Semaphore(self.s.extraction_concurrency)

        async def extract(page):  # type: ignore[no-untyped-def]
            async with sem:
                # The page's own provenance person leads: rosters are grounded
                # against whoever this page was fetched for, not against
                # person_a regardless of context.
                origin = provenance.get(page.url) or provenance.get(page.final_url)
                anchors = ([origin] if origin else []) + endpoints
                return page, await self.claude.extract_page(
                    page, anchors, is_known=self._is_known_person
                )

        # Extraction fans out; graph mutation stays serial so merges see a
        # consistent store.
        #
        # Streamed rather than gathered. `gather` waits for the whole level
        # before touching the graph, so `people` and `merges` sat frozen for
        # minutes while the job worked — indistinguishable from a hang, and the
        # reason a working crawl kept looking stalled. `as_completed` resolves
        # each page the moment its extraction lands, so every counter advances
        # continuously and the store is never more than one page behind.
        tasks = [asyncio.create_task(extract(p)) for p in pages]
        ingested = 0
        for finished in asyncio.as_completed(tasks):
            try:
                page, extractions = await finished
            except Exception as exc:
                self.log.warn("extract.failed", f"{type(exc).__name__}: {exc}")
                continue
            ingested += 1
            if ingested % 10 == 0 or ingested == len(tasks):
                self.log(
                    "ingest.progress",
                    f"{ingested}/{len(tasks)} pages resolved",
                    pages=ingested,
                    of=len(tasks),
                    nodes=len(self.store.nodes),
                    edges=len(self.store.edges),
                )
            origin = provenance.get(page.url) or provenance.get(page.final_url)
            for extraction in extractions:
                if not self._connects_to_graph(extraction, origin):
                    self.log(
                        "edge.skipped",
                        f"island: {extraction.subject_name} -> {extraction.object_name}",
                        url=page.url,
                    )
                    continue
                subject = await self._attach(page, extraction.subject_name,
                                             extraction.subject_attributes, extraction,
                                             provenance_name=origin)
                obj = await self._attach(page, extraction.object_name,
                                         extraction.object_attributes, extraction,
                                         provenance_name=origin)
                edge = self.store.add_edge(
                    subject.node_id,
                    obj.node_id,
                    extraction,
                    source_url=page.final_url,
                    source_title=page.title,
                    retrieved_at=page.retrieved_at,
                )
                if edge is not None:
                    await self.relations.add(
                        extraction,
                        source_url=page.final_url,
                        source_title=page.title,
                        retrieved_at=page.retrieved_at,
                    )

    def _is_known_person(self, name: str) -> bool:
        """Is this name already someone in the graph?

        Lets a roster that does not name the anchor still be used, hung off a
        member we already know — "Y Combinator founders" does not list Diana Hu
        but does list Paul Graham, and it genuinely asserts that he founded YC
        with Livingston, Blackwell and Morris.
        """
        return bool(self.store.candidates_for(name))

    def _connects_to_graph(self, extraction, origin: Optional[str]) -> bool:  # type: ignore[no-untyped-def]
        """Would this edge attach to what we already know, or float free?

        A page fetched while researching one person routinely asserts
        relationships between two other people entirely — a podcast page naming
        its guests, a news article's cast of characters. Those edges are real,
        but both endpoints are strangers to the graph, so they form an island
        that no path can ever cross. Last run, 67 of 83 discovered people were
        unreachable from either endpoint for exactly this reason.

        Admitting them costs extraction, identity resolution, and frontier slots
        for something that cannot contribute a route.
        """
        if not self.s.require_graph_connection or not self.store.nodes:
            return True  # seeding: the graph has to start somewhere
        for name in (extraction.subject_name, extraction.object_name):
            if origin and could_be_same_name(origin, name):
                return True
            if self.store.candidates_for(name):
                return True
        return False

    async def _attach(  # type: ignore[no-untyped-def]
        self, page, name: str, attributes: list[str], extraction,
        *, provenance_name: Optional[str] = None,
    ) -> Node:
        observation = Observation(
            url=page.final_url,
            page_title=page.title,
            span_text=extraction.span_text,
            span_start=extraction.span_start,
            span_end=extraction.span_end,
            retrieved_at=page.retrieved_at,
            attributes=bucket_attributes(attributes),
        )
        canonical = {page.final_url} if looks_canonical_for(page.final_url, name) else set()
        return await self.resolver.resolve(
            name,
            observation,
            canonical_urls=canonical,
            page_url=page.final_url,
            provenance_name=provenance_name,
        )

    # -- routes -------------------------------------------------------------
    async def _build_routes(self, seed_a: str, seed_b: str) -> list[Route]:
        ta = traverse(self.store, seed_a, self.depth_a + 2)
        tb = traverse(self.store, seed_b, self.depth_b + 2)
        meeting = set(ta.dist) & set(tb.dist)
        if not meeting:
            return []

        raw: list[tuple[list[str], list[str]]] = []
        seen_paths: set[tuple[str, ...]] = set()
        for node_id in sorted(meeting, key=lambda n: ta.dist[n] + tb.dist[n]):
            path = self._reconstruct(node_id, ta, tb)
            if path is None:
                continue
            nodes, edges = path
            key = tuple(nodes)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            raw.append((nodes, edges))
            if len(raw) >= self.s.max_routes_returned * 3:
                break

        routes: list[Route] = []
        for nodes, edge_ids in raw:
            route = await self._to_route(nodes, edge_ids)
            if route is not None:
                routes.append(route)

        routes.sort(key=lambda r: (r.length, -_basis_rank(r.weakest_identity_basis)))
        return routes[: self.s.max_routes_returned]

    def _reconstruct(
        self, meet: str, ta: Traversal, tb: Traversal
    ) -> Optional[tuple[list[str], list[str]]]:
        left_nodes: list[str] = [meet]
        left_edges: list[str] = []
        cursor = meet
        while cursor in ta.parent:
            prev, edge_id = ta.parent[cursor]
            left_nodes.append(prev)
            left_edges.append(edge_id)
            cursor = prev
        left_nodes.reverse()
        left_edges.reverse()

        right_nodes: list[str] = []
        right_edges: list[str] = []
        cursor = meet
        while cursor in tb.parent:
            nxt, edge_id = tb.parent[cursor]
            right_nodes.append(nxt)
            right_edges.append(edge_id)
            cursor = nxt

        nodes = left_nodes + right_nodes
        edges = left_edges + right_edges
        if len(set(nodes)) != len(nodes) or not edges:
            return None  # not a simple path
        return nodes, edges

    async def _to_route(self, node_ids: list[str], edge_ids: list[str]) -> Optional[Route]:
        edges = [self.store.edges[e] for e in edge_ids if e in self.store.edges]
        if len(edges) != len(edge_ids):
            return None

        # Verify every interior node: is url_in's P the same human as url_out's P?
        verdicts: dict[str, PivotVerdict] = {}
        for i in range(1, len(node_ids) - 1):
            verdicts[node_ids[i]] = await self.pivots.verify(
                node_ids[i],
                url_in=edges[i - 1].source_url,
                url_out=edges[i].source_url,
                prev_node_id=node_ids[i - 1],
                next_node_id=node_ids[i + 1],
            )

        bases = [v.basis for v in verdicts.values()]
        # A one-hop route has no pivot, so there is no pivot risk to report.
        weakest = weakest_basis(bases) if bases else IdentityBasis.SHARED_PAGE

        if not self.claude.enabled and weakest in (
            IdentityBasis.ATTRIBUTE_MATCH,
            IdentityBasis.NAME_ONLY,
        ):
            self.log(
                "route.dropped",
                "degraded mode accepts only shared_page and canonical_url pivots",
                weakest=weakest.value,
            )
            return None
        if weakest is IdentityBasis.NAME_ONLY and self.s.drop_name_only_pivots:
            self.log("route.dropped", "name_only pivot and drop_name_only_pivots is set")
            return None

        warnings = [
            f"{WARN_NAME_ONLY_PIVOT} — pivot {v.name!r}: {v.detail}"
            for v in verdicts.values()
            if v.basis is IdentityBasis.NAME_ONLY
        ]
        co_listed = [e for e in edges if e.extraction.resolution_basis in CO_MEMBERSHIP_BASES]
        for edge in co_listed:
            warnings.append(
                f"{WARN_CO_LISTING_HOP} — {edge.extraction.subject_name} / "
                f"{edge.extraction.object_name} via {edge.source_url}"
            )

        hops: list[Hop] = []
        for i, edge in enumerate(edges):
            src, dst = node_ids[i], node_ids[i + 1]
            src_node, dst_node = self.store.get(src), self.store.get(dst)
            if src_node is None or dst_node is None:
                return None
            hops.append(
                Hop(
                    **{"from": HopEndpoint(name=src_node.display_name, node_id=src)},
                    to=HopEndpoint(name=dst_node.display_name, node_id=dst),
                    span_text=edge.extraction.span_text,
                    span_start=edge.extraction.span_start,
                    span_end=edge.extraction.span_end,
                    context_before=edge.extraction.context_before,
                    resolved_statement=edge.extraction.resolved_statement,
                    resolution_basis=edge.extraction.resolution_basis,
                    source_url=edge.source_url,
                    source_title=edge.source_title,
                    retrieved_at=edge.retrieved_at,
                    from_identity_basis=self._basis_for(src, verdicts),
                    to_identity_basis=self._basis_for(dst, verdicts),
                )
            )

        return Route(
            length=len(hops),
            weakest_identity_basis=weakest,
            identity_warnings=warnings,
            hops=hops,
        )

    def _basis_for(self, node_id: str, verdicts: dict[str, PivotVerdict]) -> IdentityBasis:
        if node_id in verdicts:
            return verdicts[node_id].basis
        node = self.store.get(node_id)  # terminal node: how well pinned is the endpoint?
        if node is None:
            return IdentityBasis.NAME_ONLY
        if node.canonical_urls:
            return IdentityBasis.CANONICAL_URL
        if any(node.attributes.get(k) for k in ("employer", "institution", "field")):
            return IdentityBasis.ATTRIBUTE_MATCH
        return IdentityBasis.NAME_ONLY

    # -- assembly -----------------------------------------------------------
    def _result(
        self,
        *,
        found: bool,
        routes: Optional[list[Route]] = None,
        disambiguation: Optional[list[DisambiguationCandidate]] = None,
    ) -> Result:
        return Result(
            found=found,
            routes=routes or [],
            disambiguation=disambiguation,
            stats=self.ledger.snapshot(
                merges=self.resolver.merges, merges_blocked=self.resolver.merges_blocked
            ),
            warnings=self.warnings,
        )


def _parse_iso(value: str) -> "datetime":
    """Original fetch time of a cached relation; falls back to now if unreadable."""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return utcnow()


def _basis_rank(basis: IdentityBasis) -> int:
    from artemis.models import IDENTITY_BASIS_STRENGTH

    return IDENTITY_BASIS_STRENGTH[basis]


__all__ = ["Connector", "SerperUnavailable", "traverse", "Traversal"]
