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
from typing import Any, Optional, Sequence

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
    CLAIM_BASES,
    CO_MEMBERSHIP_BASES,
    WARN_CO_LISTING_HOP,
    WARN_NAME_ONLY_PIVOT,
    WARN_NO_REFERENT_RESOLUTION,
    WARN_STRUCTURED_CLAIM_HOP,
    ConnectRequest,
    DisambiguationCandidate,
    Endpoint,
    Extraction,
    Hop,
    HopEndpoint,
    IdentityBasis,
    Node,
    Observation,
    ResolutionBasis,
    Result,
    Route,
    SearchTier,
    TrackedNode,
    utcnow,
    weakest_basis,
)
from artemis.graph.relations import RelationCache
from artemis.runtime import BudgetLedger, JobLog, RunControl
from artemis.scrape.fetcher import Fetcher
from artemis.search.base import Query, SearchResults
from artemis.search.templates import QueryTemplate, sanitise_hypothesis
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
        control: Optional[RunControl] = None,
    ) -> None:
        self.control = control or RunControl()
        #: Set once seeding succeeds; the mid-level route check needs both ends
        #: and _expand does not take them as arguments.
        self._seed_a: Optional[str] = None
        self._seed_b: Optional[str] = None
        self._published: tuple[str, ...] = ()
        #: Set by _expand when a mid-level checkpoint says stop, so _search_loop
        #: does not start another level before noticing.
        self._stop_mid_level = False
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
        #: Breadth-first picture of the search, published as it fills.
        self._tiers: list[SearchTier] = []
        #: An unrecognised value runs bidirectional rather than failing a job
        #: over a typo in an env var; run() logs which one is in force either way.
        self._staged = (settings.search_strategy or "").strip().lower() == "staged"
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

    # -- early routes -------------------------------------------------------
    async def _checkpoint(self) -> bool:
        """Publish any route that exists yet. True means stop crawling.

        Cheap until it matters: the meeting-point test is a set intersection
        over two in-memory traversals, so this costs nothing on the overwhelming
        majority of calls, when the frontiers have not met. Only once they have
        does it build routes, and pivot verdicts for an unchanged pivot come
        back from the Claude disk cache rather than the model.
        """
        target = self._target_node()
        if self._seed_a is None or target is None:
            return False
        ta = traverse(self.store, self._seed_a, self.depth_a + 2)
        tb = traverse(self.store, target, self.depth_b + 2)
        if not (set(ta.dist) & set(tb.dist)):
            return self.control.should_stop()

        routes = await self._build_routes(self._seed_a, target)
        if routes:
            # Republishing an unchanged set would restart the console's prompt
            # under the operator every few seconds.
            fingerprint = tuple(
                "|".join(h.to.node_id for h in r.hops) for r in routes
            )
            if fingerprint != self._published:
                self._published = fingerprint
                self.control.publish(routes)
                self.log(
                    "routes.preview",
                    f"{len(routes)} route(s) available; still looking for a shorter one",
                    routes=len(routes),
                    shortest=min(r.length for r in routes),
                )
            if self.s.auto_stop_on_first_route:
                self.log("routes.auto_stop", "auto_stop_on_first_route is set")
                return True
        return self.control.should_stop()

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
        # The registry holds its own copies; these are the crawl's working set.
        self._tiers = []

    # -- entry point --------------------------------------------------------
    async def run(self) -> Result:
        if not self.claude.enabled:
            self.warnings.append(WARN_NO_REFERENT_RESOLUTION)
            self.log.warn("degraded", "no Anthropic key: strict same-sentence extraction only")

        configured = (self.s.search_strategy or "").strip().lower()
        if configured not in ("bidirectional", "staged"):
            self.log.warn(
                "strategy.unknown",
                f"search_strategy={self.s.search_strategy!r} is not recognised; "
                "running bidirectional",
                configured=self.s.search_strategy,
            )
        self.log(
            "job.started",
            f"{self.req.person_a} <-> {self.req.person_b}",
            strategy="staged" if self._staged else "bidirectional",
            depth_a=self.depth_a,
            depth_b=self.depth_b,
            structured_providers=[p.name for p in self.providers],
        )

        if self._staged:
            # The tracker has nothing to show until tier 0 opens, and seeding is
            # not quick, so say what is happening rather than render an empty
            # box for minutes.
            self._tiers = [
                SearchTier(
                    level=-1,
                    label="seeding the origin",
                    parents=[self.req.person_a],
                    status="running",
                )
            ]
            self._publish_tiers()

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
        # Staged never expands the target — it walks out from the origin and
        # asks who reaches him. Crawling him anyway cost 203 seconds and 685
        # merge decisions off 7 pages before a single node was expanded, and
        # bought nothing: the whole graph it built was on the side the strategy
        # does not traverse. The context box is the target description the
        # ranker and the reachability pass actually read.
        #
        # He still becomes a node — the moment the origin side grounds an edge
        # naming him, the resolver creates him like anyone else. _target_node()
        # looks for that, so routes form exactly as before, just without paying
        # to build his network first.
        seed_b: Optional[str] = None
        if self._staged:
            self.log(
                "staged.target_not_seeded",
                f"{self.req.person_b} will not be crawled; "
                f"context: {self.req.context_b or '(none given)'}",
                target=self.req.person_b, context=self.req.context_b or "",
            )
        else:
            seed_b, disambiguation = await self._seed(
                self.req.person_b, self.req.context_b, Endpoint.B, self.req.person_a
            )
            if disambiguation:
                return self._result(found=False, disambiguation=disambiguation)

        if seed_a is None or (seed_b is None and not self._staged):
            missing = self.req.person_a if seed_a is None else self.req.person_b
            self.warnings.append(f"no grounded mention of {missing!r} was found on any fetched page")
            return self._result(found=False)

        self._seed_a, self._seed_b = seed_a, seed_b

        # Both loops take (origin, target) and return "did a checkpoint stop
        # us"; everything after this point is strategy-agnostic.
        stopped = await (
            self._staged_loop(seed_a, seed_b) if self._staged
            else self._search_loop(seed_a, seed_b)
        )
        # Enrichment is another minutes-long phase. Someone who has already
        # accepted a route is not waiting through it.
        if not stopped:
            await self._enrich_with_providers(seed_a, seed_b)
        # Staged never seeded him, so resolve him now: he is in the graph only
        # if the origin side actually reached him.
        final_target = self._target_node() or seed_b
        routes = (
            await self._build_routes(seed_a, final_target) if final_target else []
        )
        if stopped:
            self.log("run.stopped_early", f"finishing with {len(routes)} route(s)",
                     routes=len(routes))

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
    async def _search_loop(self, seed_a: str, seed_b: str) -> bool:
        """Expand until the depth budget runs out. True if stopped early."""
        allowed_levels = max(self.depth_a, self.depth_b)
        level = 0
        bonus_used = False

        while level < allowed_levels:
            if self.ledger.out_of_time():
                self.log.warn("budget", "wall clock exhausted")
                break
            if await self._checkpoint():
                return True

            frontier: list[tuple[str, Endpoint]] = []
            ta = traverse(self.store, seed_a, self.depth_a)
            tb = traverse(self.store, seed_b, self.depth_b)
            # Each side steers toward the fame level of the person it is trying
            # to reach: A is hunting B, so it follows B's notability, and vice
            # versa.
            if level < self.depth_a:
                # A is hunting B, so B is what its candidates are ranked toward.
                frontier += [
                    (n, Endpoint.A)
                    for n in await self._frontier(
                        ta, level, toward_famous=self._b_is_famous,
                        target_node_id=seed_b, target_name=self.req.person_b,
                    )
                ]
            if level < self.depth_b:
                frontier += [
                    (n, Endpoint.B)
                    for n in await self._frontier(
                        tb, level, toward_famous=self._a_is_famous,
                        target_node_id=seed_a, target_name=self.req.person_a,
                    )
                ]

            frontier = [(n, e) for n, e in frontier if n not in self._expanded]
            if not frontier:
                self.log("level.empty", f"nothing left to expand at level {level}")
                break

            self.log("level.started", f"level {level}", frontier=len(frontier))
            await self._expand(frontier)
            if self._stop_mid_level or await self._checkpoint():
                return True

            ta = traverse(self.store, seed_a, self.depth_a + 2)
            tb = traverse(self.store, seed_b, self.depth_b + 2)
            meeting = set(ta.dist) & set(tb.dist)
            if meeting and not bonus_used:
                bonus_used = True
                allowed_levels = min(allowed_levels, level + 2)
                self.log("frontiers.met", f"{len(meeting)} meeting point(s)",
                         at_level=level, continuing_one_more_level=True)
            level += 1
        return False

    # -- staged search ------------------------------------------------------
    async def _staged_loop(self, origin: str, target: str) -> bool:
        """Bottom-up and one-directional: origin -> best 12 -> best 12 of each.

        The bidirectional loop assumes both endpoints have a searchable network
        for the two frontiers to meet in. When the target's side of the web is
        thin, B's frontier stalls and A's depth budget is spent guessing which
        branch to follow. This walks outward from the origin only, ranking hard
        at every step, and then asks — once, over everyone it is holding —
        which of them plausibly reach the target, instead of deciding that one
        branch at a time on the way out.

        Returns True if a checkpoint said stop, exactly like _search_loop.
        """
        cap = self.s.frontier_cap_per_level
        per_parent = max(1, self.s.staged_keep_per_parent)
        target_name = self.req.person_b

        # Three expansions deep by construction (origin -> tier 1 -> pursuit).
        # A shallower depth budget would build a graph whose far end no
        # traversal in _checkpoint or _build_routes ever reaches, so the routes
        # would exist in the store and never be found.
        if self.depth_a < 3:
            self.log("staged.depth_raised", f"depth_a {self.depth_a} -> 3 for the staged plan",
                     was=self.depth_a)
            self.depth_a = 3

        parents_allowed, pursue_allowed = self._staged_budget(cap, self.s.staged_pursue_cap)

        # -- tier 0: the origin, expanded completely ------------------------
        tier0 = self._tier(0, "origin", parents=[self._display(origin)])
        await self._expand([(origin, Endpoint.A)])
        if self._stop_mid_level or await self._checkpoint():
            self._close_tiers()
            return True
        surfaced = self._surfaced_by(origin, {origin, target})
        ranked = await self._rank(
            self._scored(surfaced), cap, toward_famous=self._b_is_famous,
            target_node_id=target, target_name=target_name, where="the origin",
        )
        tier0.found, tier0.candidates = len(surfaced), self._tracked(ranked)
        tier0.kept, tier0.status = len(tier0.candidates), "done"
        self.log("tier.done", f"origin surfaced {tier0.found}, kept {tier0.kept}",
                 tier=0, found=tier0.found, kept=tier0.kept,
                 top=[c.name for c in tier0.candidates[:3]])
        self._publish_tiers()

        # -- tier 1: each of those, expanded completely ---------------------
        parents = tier0.candidates[:parents_allowed]
        tier1 = self._tier(1, "tier 1", parents=[p.name for p in parents])
        seen = {origin, target} | {c.node_id for c in tier0.candidates}
        for parent in parents:
            if self.ledger.out_of_time():
                self.log.warn("budget", "wall clock exhausted mid tier 1",
                              expanded=sum(1 for p in parents if p.expanded), of=len(parents))
                break
            await self._expand([(parent.node_id, Endpoint.A)])
            parent.expanded = True
            if self._stop_mid_level or await self._checkpoint():
                self._close_tiers()
                return True
            children = self._surfaced_by(parent.node_id, seen)
            kept = await self._rank(
                self._scored(children), per_parent, toward_famous=self._b_is_famous,
                target_node_id=target, target_name=target_name, where=parent.name,
            )
            seen |= {node_id for node_id, _ in kept}
            # Renumbered across the tier rather than restarted per parent: rank
            # 1 is the best candidate of the best parent, which is the order
            # pursuit falls back on when reachability cannot separate them.
            tier1.candidates += self._tracked(kept, start=len(tier1.candidates) + 1)
            tier1.found += len(children)
            tier1.kept = len(tier1.candidates)
            self._publish_tiers()
        self.log("tier.done", f"tier 1 surfaced {tier1.found}, kept {tier1.kept}",
                 tier=1, parents=len(parents), found=tier1.found, kept=tier1.kept)

        # -- tier 2: which of them reach the target, and only those ---------
        pursued = await self._pursuable(tier1.candidates, target, pursue_allowed)
        tier1.status = "done"
        tier2 = self._tier(2, "tier 2", parents=[p.name for p in pursued])
        for candidate in pursued:
            if self.ledger.out_of_time():
                self.log.warn("budget", "wall clock exhausted mid tier 2",
                              expanded=sum(1 for p in pursued if p.expanded), of=len(pursued))
                break
            await self._expand([(candidate.node_id, Endpoint.A)])
            candidate.expanded = True
            self._publish_tiers()
            if self._stop_mid_level or await self._checkpoint():
                self._close_tiers()
                return True

        reached: list[str] = []
        for candidate in pursued:
            reached += self._surfaced_by(candidate.node_id, seen | set(reached))
        ranked = await self._rank(
            self._scored(reached), cap, toward_famous=self._b_is_famous,
            target_node_id=target, target_name=target_name, where="tier 2",
        )
        tier2.found, tier2.candidates = len(reached), self._tracked(ranked)
        tier2.kept, tier2.status = len(tier2.candidates), "done"
        self.log("tier.done", f"tier 2 surfaced {tier2.found}, kept {tier2.kept}",
                 tier=2, parents=len(pursued), found=tier2.found, kept=tier2.kept)
        self._publish_tiers()

        # -- tier 3: the target's side, finally ------------------------------
        if await self._probe_target_links(tier1.candidates + tier2.candidates):
            self._close_tiers()
            return True
        self._close_tiers()
        return False

    async def _probe_target_links(self, pool: list[TrackedNode]) -> bool:
        """Ask where the target might attach to anyone we hold, then check.

        The only point at which the target's own side is touched, and it is a
        guess-then-verify rather than a crawl: his network is never expanded,
        so the cost is one search per plausible tie instead of the hundreds of
        pages that seeding a famous person costs. Grounding is unchanged — a
        proposed tie becomes a hop only if a fetched page states it.

        Returns True if a checkpoint said stop.
        """
        limit = max(0, int(getattr(self.s, "target_link_probes", 0) or 0))
        if not pool or limit <= 0:
            return False

        tier3 = self._tier(3, "target links", parents=[self.req.person_b])
        facts = [f for f in (self._candidate_facts(c.node_id) for c in pool) if f]
        links = await self.claude.propose_target_links(
            facts,
            self.req.person_b,
            (self.req.context_b or "").strip(),
            limit,
            batch_size=max(1, self.s.reachability_batch_size),
        )
        if not links:
            self.log.warn("target_links.none",
                          f"no plausible tie proposed between {self.req.person_b} "
                          f"and any of the {len(facts)} people held")
            tier3.status = "done"
            self._publish_tiers()
            return False

        by_id = {c.node_id: c for c in pool}
        queries: list[Query] = []
        tracked: list[TrackedNode] = []
        for node_id, connection, raw in links:
            rendered = sanitise_hypothesis(raw)
            candidate = by_id.get(node_id)
            if not rendered or candidate is None:
                continue
            marked = candidate.model_copy(update={
                "rank": len(tracked) + 1,
                "reaches_target": "possible",
                "reaches_target_why": connection,
            })
            tracked.append(marked)
            self.log("target_link.proposed",
                     f"{self.req.person_b} <- {candidate.name}: {connection}",
                     candidate=candidate.name, query=rendered)
            queries.append(
                Query(
                    template=QueryTemplate.HYPOTHESIS,
                    rendered=rendered,
                    subject_name=candidate.name,
                    node_id=node_id,
                )
            )

        tier3.candidates, tier3.found, tier3.kept = tracked, len(links), len(tracked)
        self._publish_tiers()
        if not queries:
            tier3.status = "done"
            self._publish_tiers()
            return False

        self.log("target_links.verifying", f"{len(queries)} tie(s) to check by search",
                 queries=len(queries))
        results = await self.provider.search(queries)
        await self._ingest(self._urls_from(results))
        tier3.status = "done"
        self._publish_tiers()
        return self._stop_mid_level or await self._checkpoint()

    async def _pursuable(
        self, candidates: list[TrackedNode], target: str, limit: int
    ) -> list[TrackedNode]:
        """Ask which of the held candidates reach the target; return who to expand.

        The verdicts are stamped on the candidates themselves, so the console
        shows what was decided about all of them, not only the survivors.

        "no" is the only answer that removes anyone, and only when some other
        candidate scored better: an assessment that rejects everybody is far
        more likely a bad batch than 144 true negatives, so it degrades to rank
        order rather than ending the search.
        """
        if not candidates:
            return []

        target_attributes = self._target_facts()

        facts = [f for f in (self._candidate_facts(c.node_id) for c in candidates) if f]
        verdicts = await self.claude.assess_reachability(
            facts, self.req.person_b, target_attributes
        )
        by_id = {c.node_id: c for c in candidates}
        tally = {"yes": 0, "no": 0, "unknown": 0}
        for node_id, answer, why in verdicts:
            tracked = by_id.get(node_id)
            if tracked is None:
                continue
            tracked.reaches_target = answer
            tracked.reaches_target_why = why
            tally[answer] = tally.get(answer, 0) + 1
        self._publish_tiers()

        self.log(
            "reachability.assessed",
            f"{tally['yes']} yes / {tally['no']} no / {tally['unknown']} unknown "
            f"of {len(candidates)} toward {self.req.person_b}",
            **tally, candidates=len(candidates),
        )
        if not tally["yes"] and not tally["no"]:
            self.log.warn(
                "reachability.degraded",
                "no reachability judgement available; pursuing by rank",
                candidates=len(candidates),
            )

        order = {"yes": 0, "unknown": 1, "no": 2}
        ranked = sorted(
            candidates, key=lambda c: (order.get(c.reaches_target or "unknown", 1), c.rank)
        )
        pursued = [c for c in ranked if c.reaches_target != "no"][:limit]
        if not pursued:
            self.log.warn(
                "reachability.rejected_all",
                f"every candidate was judged unable to reach {self.req.person_b}; "
                "pursuing the top-ranked ones anyway",
                candidates=len(candidates),
            )
            pursued = ranked[:limit]
        self.log("tier.pursuing", f"{len(pursued)} of {len(candidates)} candidates",
                 pursuing=[c.name for c in pursued[:5]])
        return pursued

    def _staged_budget(self, parents: int, pursue: int) -> tuple[int, int]:
        """Trim the staged plan to what the ledger can still pay for.

        One expansion costs `_expand_cost()` Serper credits and one of
        max_nodes_expanded, and the plan is 1 origin + `parents` + `pursue` of
        them. Measured against what is LEFT rather than the configured ceiling,
        because seeding has already spent some of both.

        Pursuit is trimmed first: an unexpanded tier-1 parent costs a whole
        group of candidates that were never surfaced at all, where an unpursued
        tier-2 candidate was at least surfaced, ranked and assessed. Never
        silent — a short plan is logged with the arithmetic that produced it.
        """
        cost = self._expand_cost()
        credits_left = max(0, self.ledger.max_serper_credits - self.ledger.serper_credits_used)
        nodes_left = max(0, self.ledger.max_nodes_expanded - self.ledger.nodes_expanded)
        affordable = min(credits_left // cost, nodes_left)
        wanted = 1 + parents + pursue

        if affordable >= wanted:
            self.log(
                "staged.plan",
                f"{wanted} expansions at ~{cost} Serper credits each "
                f"({wanted * cost} of {credits_left} remaining)",
                parents=parents, pursue=pursue, credits=wanted * cost,
                credits_left=credits_left, nodes_left=nodes_left,
            )
            return parents, pursue

        spare = max(0, affordable - 1)  # the origin is expanded before anything else
        got_parents = min(parents, spare)
        got_pursue = min(pursue, spare - got_parents)
        self.log.warn(
            "staged.budget_shortfall",
            f"plan wants {wanted} expansions (~{wanted * cost} Serper credits) but only "
            f"{affordable} are affordable: {credits_left} credits and {nodes_left} node "
            f"expansions remain at ~{cost} credits each — expanding {got_parents} of "
            f"{parents} tier-1 parents and pursuing {got_pursue} of {pursue}",
            wanted=wanted, affordable=affordable, cost_per_node=cost,
            credits_left=credits_left, nodes_left=nodes_left,
            parents=got_parents, parents_planned=parents,
            pursue=got_pursue, pursue_planned=pursue,
        )
        return got_parents, got_pursue

    def _expand_cost(self) -> int:
        """Serper credits one expansion spends. Planning only; the ledger rules.

        The shape of plan_queries on a non-seed node: COLLEAGUES always, plus at
        most one angle template, then one search per bridge hypothesis. Credits
        are charged per query, not per POST, so batching does not change it.
        """
        return 2 + max(0, self.s.bridge_hypotheses_per_node)

    def _surfaced_by(self, node_id: str, exclude: set[str]) -> list[str]:
        """Who this expansion left the node adjacent to, minus who we already hold.

        Read back off the store rather than tracked during ingestion: edges
        arrive out of order and identity merges rewrite ids underneath, so the
        store is the only thing that knows who a node ended up next to.
        """
        anchor = self.store.resolve_id(node_id)
        blocked = {self.store.resolve_id(n) for n in exclude} | {anchor}
        out: list[str] = []
        for neighbor, _edge in self.store.neighbors(anchor):
            resolved = self.store.resolve_id(neighbor)
            if resolved in blocked:
                continue
            blocked.add(resolved)
            out.append(resolved)
        return out

    def _scored(self, node_ids: Sequence[str]) -> list[tuple[str, int]]:
        """(id, distinct source URLs) — the input rank_frontier and _rank expect."""
        return [
            (node_id, len({o.url for o in node.observations}))
            for node_id in node_ids
            if (node := self.store.get(node_id)) is not None
        ]

    def _tracked(
        self, ranked: Sequence[tuple[str, str]], *, start: int = 1
    ) -> list[TrackedNode]:
        out: list[TrackedNode] = []
        for offset, (node_id, why) in enumerate(ranked):
            node = self.store.get(node_id)
            out.append(
                TrackedNode(
                    node_id=node_id,
                    name=node.display_name if node else node_id,
                    rank=start + offset,
                    sources=len({o.url for o in node.observations}) if node else 0,
                    why=why,
                )
            )
        return out

    def _tier(self, level: int, label: str, *, parents: list[str]) -> SearchTier:
        tier = SearchTier(level=level, label=label, parents=parents)
        self._tiers.append(tier)
        # Published empty, on purpose: the console polls every 1.2s and a tier
        # that only appears once it is full looks like nothing happening for the
        # several minutes it takes to fill.
        self._publish_tiers()
        return tier

    def _target_node(self) -> Optional[str]:
        """The target's node, once the origin side has grounded a mention of him.

        Staged never seeds him, so he exists only from the moment somebody's
        page says something about him — which is also the only moment a route
        through him becomes possible.
        """
        if self._seed_b is not None:
            return self._seed_b
        for node in self.store.nodes.values():
            if could_be_same_name(self.req.person_b, node.display_name) or any(
                could_be_same_name(self.req.person_b, v) for v in node.name_variants
            ):
                self._seed_b = node.node_id
                self.log("staged.target_reached", f"{node.display_name} is now in the graph",
                         node_id=node.node_id)
                return self._seed_b
        return None

    def _target_facts(self) -> dict[str, list[str]]:
        """What the ranker is steering toward.

        Grounded attributes when the target is in the graph; otherwise the
        caller's context box. That fallback is the whole point of not crawling
        him: a line like "US President" is enough to point candidates in the
        right direction, and it costs nothing to obtain.
        """
        node_id = self._target_node()
        if node_id and (node := self.store.get(node_id)) is not None and node.attributes:
            return {k: sorted(v) for k, v in node.attributes.items() if v}
        context = (self.req.context_b or "").strip()
        return {"description": [context]} if context else {}

    def _publish_tiers(self) -> None:
        self.control.publish_tiers(self._tiers)

    def _close_tiers(self) -> None:
        """Mark every tier finished and publish, so nothing is left reading "running"."""
        for tier in self._tiers:
            tier.status = "done"
        self._publish_tiers()

    async def _frontier(
        self,
        t: Traversal,
        level: int,
        *,
        toward_famous: bool = True,
        target_node_id: Optional[str] = None,
        target_name: str = "",
    ) -> list[str]:
        scored = [
            (node_id, len({o.url for o in node.observations}))
            for node_id, depth in t.dist.items()
            if depth == level and (node := self.store.get(node_id)) is not None
        ]
        ranked = await self._rank(
            scored,
            self.s.frontier_cap_per_level,
            toward_famous=toward_famous,
            target_node_id=target_node_id,
            target_name=target_name,
            where=f"level {level}",
        )
        return [node_id for node_id, _ in ranked]

    async def _rank(
        self,
        scored: list[tuple[str, int]],
        cap: int,
        *,
        toward_famous: bool = True,
        target_node_id: Optional[str] = None,
        target_name: str = "",
        where: str = "",
    ) -> list[tuple[str, str]]:
        """Best `cap` candidates toward the target, each with why it is there.

        Shared by both strategies: the bidirectional loop ranks one BFS level,
        the staged loop ranks what a single parent surfaced. Same call, same
        cap, same degradation — a staged tier must not quietly rank by a
        different rule than a level does.
        """
        heuristic = self.policy.rank_frontier(scored, cap, toward_famous=toward_famous)
        by_sources = dict(scored)

        def unranked(node_ids: list[str]) -> list[tuple[str, str]]:
            # No model judgement was applied, so do not attribute one: say what
            # the recurrence heuristic actually ordered on.
            return [(n, f"{by_sources.get(n, 0)} distinct sources") for n in node_ids]

        if not self.s.frontier_selection_enabled or len(scored) < 2:
            return unranked(heuristic)

        # Rank BEFORE the cap, not after. The heuristic orders by how many
        # distinct sources mention someone, which is prominence — so ranking
        # only what it already admitted would hide exactly the candidate this
        # exists to find: a modest bridge into the target's world.
        pool = self.policy.rank_frontier(
            scored, self.s.frontier_rank_pool, toward_famous=toward_famous
        )
        if len(scored) > len(pool):
            self.log(
                "frontier.pool_capped",
                f"ranking {len(pool)} of {len(scored)} candidates at {where}",
                candidates=len(scored), ranked=len(pool),
            )

        candidates = [c for c in (self._candidate_facts(n) for n in pool) if c]
        target_attributes: dict[str, list[str]] = {}
        if target_node_id and (node := self.store.get(target_node_id)) is not None:
            target_attributes = {k: sorted(v) for k, v in node.attributes.items() if v}
        if not target_attributes and target_name == self.req.person_b:
            target_attributes = self._target_facts()
            target_name = target_name or node.display_name

        ordered, why = await self.claude.choose_frontier(
            candidates, target_name, target_attributes
        )
        if ordered[:cap] != pool[:cap]:
            self.log(
                "frontier.reranked",
                why or f"reordered {len(ordered)} candidates toward {target_name}",
                target=target_name, top=[self._display(n) for n in ordered[:3]],
            )
        kept = ordered[:cap]
        if not kept:
            return unranked(heuristic)
        # The model writes one sentence about its top pick, so only the top pick
        # can honestly carry it.
        return [
            (n, why if i == 0 and why else f"ranked {i + 1} of {len(ordered)} toward {target_name}")
            for i, n in enumerate(kept)
        ]

    async def _bridge_queries(self, node: Node, target: str) -> list[Query]:
        """Ask who specifically might connect this node to the far endpoint.

        The templates say "find people near this person"; this says "find a page
        about this person and *that* one". It is a guess, and it is labelled as
        one — QueryTemplate.HYPOTHESIS, plus a log line per bridge naming who was
        proposed and why, so a run can be read back and the guesses judged.
        """
        limit = int(getattr(self.s, "bridge_hypotheses_per_node", 0) or 0)
        if limit <= 0:
            return []

        target_node = self.store.get(self._seed_b if target == self.req.person_b
                                     else self._seed_a) if self._seed_a else None
        bridges = await self.claude.propose_bridges(
            node.display_name,
            {k: sorted(v) for k, v in node.attributes.items() if v},
            target,
            {k: sorted(v) for k, v in (target_node.attributes if target_node else {}).items() if v},
            limit,
        )

        out: list[Query] = []
        for name, why, raw in bridges:
            rendered = sanitise_hypothesis(raw)
            if not rendered:
                self.log.warn("bridge.rejected", f"unusable query for {name!r}", bridge=name)
                continue
            self.log("bridge.proposed", f"{node.display_name} -> {name}: {why}",
                     node_id=node.node_id, bridge=name, query=rendered)
            out.append(
                Query(
                    template=QueryTemplate.HYPOTHESIS,
                    rendered=rendered,
                    subject_name=node.display_name,
                    node_id=node.node_id,
                )
            )
        return out

    def _candidate_facts(self, node_id: str) -> Optional[dict[str, Any]]:
        node = self.store.get(node_id)
        if node is None:
            return None
        return {
            "id": node_id,
            "name": node.display_name,
            "attributes": {k: sorted(v)[:4] for k, v in node.attributes.items() if v},
            "sources": len({o.url for o in node.observations}),
        }

    def _display(self, node_id: str) -> str:
        node = self.store.get(node_id)
        return node.display_name if node else node_id

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
            queries += await self._bridge_queries(node, target)

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

        claimed = await self._ingest_assertions(targets, sem)
        await self._ingest(urls)
        self.log("providers.finished",
                 f"{len(urls)} documents discovered, {claimed} claims admitted",
                 documents=len(urls), claims=claimed)

    async def _ingest_assertions(
        self, targets: list[tuple[str, list[str]]], sem: asyncio.Semaphore
    ) -> int:
        """Admit relationships providers state outright, with no page to read.

        Runs before _ingest, not instead of it: the discovered documents are
        still fetched, and a hop grounded in someone's prose outranks one
        grounded in a record. This is what keeps a relationship that no page
        happens to phrase extractably — which is most registry filings, and many
        Wikidata claims — from being lost entirely.
        """
        asserting = [p for p in self.providers if hasattr(p, "assert_relations")]
        if not asserting:
            return 0

        async def ask(provider, person: str, orgs: list[str]):  # type: ignore[no-untyped-def]
            async with sem:
                return await provider.assert_relations(person=person, orgs=orgs)

        results = await asyncio.gather(
            *(
                ask(provider, person, orgs)
                for provider in asserting
                for person, orgs in targets
            ),
            return_exceptions=True,
        )

        admitted = 0
        for batch in results:
            if isinstance(batch, BaseException):
                self.log.warn("claim.error", f"{type(batch).__name__}: {batch}")
                continue
            for assertion in batch:
                if await self._admit_claim(assertion):
                    admitted += 1
        return admitted

    async def _admit_claim(self, assertion: Any) -> bool:
        extraction = _claim_extraction(assertion)
        if extraction is None:
            return False
        # An island is an island however well attested. A claim between two
        # strangers still cannot contribute a route, and admitting it would
        # undo the guard that keeps 67-of-83 discovered people out of the graph.
        if not self._connects_to_graph(extraction, assertion.subject):
            return False

        observation_url = assertion.source_url
        subject = await self.resolver.resolve(
            extraction.subject_name,
            Observation(
                url=observation_url, page_title=assertion.source_title,
                span_text=extraction.span_text, span_start=extraction.span_start,
                span_end=extraction.span_end, retrieved_at=utcnow(),
                attributes=bucket_attributes([]),
            ),
            # The provider's canonical id is a stronger identity statement than
            # any name match: it is the entity, not a string that looks like it.
            canonical_urls={observation_url} if assertion.subject_id else set(),
            page_url=observation_url,
            provenance_name=assertion.subject,
        )
        obj = await self.resolver.resolve(
            extraction.object_name,
            Observation(
                url=observation_url, page_title=assertion.source_title,
                span_text=extraction.span_text, span_start=extraction.span_start,
                span_end=extraction.span_end, retrieved_at=utcnow(),
                attributes=bucket_attributes([]),
            ),
            page_url=observation_url,
            provenance_name=assertion.object,
        )
        edge = self.store.add_edge(
            subject.node_id, obj.node_id, extraction,
            source_url=observation_url,
            source_title=assertion.source_title,
            retrieved_at=utcnow(),
        )
        if edge is None:
            return False
        self.log(
            "claim.admitted",
            f"{assertion.subject} {assertion.relation} {assertion.object}",
            provider=assertion.provider, source=observation_url,
        )
        return True

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

            # Checked here, after this page is in the graph, rather than before:
            # the page is already fetched and extracted by this point, so
            # stopping first would pay for it and then throw it away.
            every = max(1, self.s.route_check_every_pages)
            if ingested % every == 0 and await self._checkpoint():
                self._stop_mid_level = True
                break

        if self._stop_mid_level:
            # as_completed leaves the rest running. Cancel and await them, or
            # they surface later as "Task exception was never retrieved" — and
            # keep spending Claude calls on a crawl that is already finished.
            for pending in tasks:
                pending.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

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

        # Length first, then prose over records: at equal length a route whose
        # every hop is somebody's writing outranks one leaning on a provider's
        # claim, however well attested that claim is. Identity strength breaks
        # the remaining ties, as before.
        routes.sort(
            key=lambda r: (
                r.length,
                _claim_hops(r),
                -_basis_rank(r.weakest_identity_basis),
            )
        )
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
        for edge in edges:
            if edge.extraction.resolution_basis in CLAIM_BASES:
                warnings.append(
                    f"{WARN_STRUCTURED_CLAIM_HOP} — {edge.extraction.resolved_statement} "
                    f"({edge.source_url})"
                )
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


def _claim_hops(route: Route) -> int:
    return sum(1 for hop in route.hops if hop.resolution_basis in CLAIM_BASES)


def _claim_extraction(assertion: Any) -> Optional[Extraction]:
    """Render an assertion as an Extraction whose span is the claim itself.

    The span is not page text and is not pretending to be: STRUCTURED_CLAIM says
    so at every layer that reads it. It exists because the whole pipeline below
    this point — grounding, merging, route building — is built on Extraction, and
    a claim that cannot travel through it cannot become a hop.
    """
    subject = (getattr(assertion, "subject", "") or "").strip()
    obj = (getattr(assertion, "object", "") or "").strip()
    relation = (getattr(assertion, "relation", "") or "").strip()
    if not subject or not obj or subject.casefold() == obj.casefold():
        return None
    statement = f"{subject} {relation} {obj}".strip()
    try:
        return Extraction(
            subject_name=subject,
            object_name=obj,
            span_text=statement,
            span_start=0,
            span_end=len(statement),
            resolved_statement=statement,
            resolution_basis=ResolutionBasis.STRUCTURED_CLAIM,
        )
    except Exception:
        return None


def _basis_rank(basis: IdentityBasis) -> int:
    from artemis.models import IDENTITY_BASIS_STRENGTH

    return IDENTITY_BASIS_STRENGTH[basis]


__all__ = ["Connector", "SerperUnavailable", "traverse", "Traversal"]
