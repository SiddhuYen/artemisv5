"""Expansion policy interface + symmetric default.

The default policy treats both endpoints identically — no asymmetric or
notability logic in this build. Only the *depth budget* differs per side, and
that lives in config, not here. A different policy (notability-weighted,
employer-first, whatever) drops in without touching the search loop.
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence

from artemis.config import Settings
from artemis.models import Node
from artemis.search.base import Query
from artemis.search.templates import ANGLE_TEMPLATES, QueryTemplate, StrategyAngle, render


class ExpansionPolicy(Protocol):
    def plan_queries(
        self,
        node: Node,
        *,
        context: Optional[str] = None,
        other_endpoint_name: Optional[str] = None,
        is_seed: bool = False,
        angle: Optional[str] = None,
    ) -> list[Query]: ...

    def rank_frontier(
        self, scored: Sequence[tuple[str, int]], cap: int, *, toward_famous: bool = True
    ) -> list[str]: ...


class SymmetricPolicy:
    def __init__(self, settings: Settings) -> None:
        self.s = settings

    def plan_queries(
        self,
        node: Node,
        *,
        context: Optional[str] = None,
        other_endpoint_name: Optional[str] = None,
        is_seed: bool = False,
        angle: Optional[str] = None,
    ) -> list[Query]:
        name = node.display_name
        planned: list[tuple[QueryTemplate, dict[str, str]]] = []

        # The direct-bridge probe runs early for both endpoints: a lot of
        # two-hop paths are found by just asking Google about both people.
        if is_seed and other_endpoint_name:
            planned.append(
                (QueryTemplate.DIRECT_BRIDGE, {"other_endpoint_name": other_endpoint_name})
            )
        if context:
            planned.append((QueryTemplate.CONTEXT, {"context": context}))
            # A supplied disambiguator is almost always the person's
            # organisation, which is the one thing that reaches a roster page.
            # Without these two, co-listing can never fire on an endpoint: every
            # other query is person-shaped and person queries do not return team
            # pages.
            planned.append((QueryTemplate.EMPLOYER_TEAM, {"employer": context}))
            planned.append((QueryTemplate.ORG_ROSTER, {"employer": context}))

        # Broad search always runs; the angle only redirects what comes after it.
        planned.append((QueryTemplate.COLLEAGUES, {}))
        if is_seed:
            planned.append((QueryTemplate.PROFILE, {}))

        employers = sorted(node.attributes.get("employer", set()))
        chosen = ANGLE_TEMPLATES.get(
            StrategyAngle(angle) if angle in {a.value for a in StrategyAngle} else
            StrategyAngle.GENERIC,
            (),
        )
        for template in chosen:
            if template is QueryTemplate.EMPLOYER_TEAM:
                if employers:
                    planned.append((QueryTemplate.EMPLOYER_TEAM, {"employer": employers[0]}))
            else:
                planned.append((template, {}))

        queries: list[Query] = []
        for template, kwargs in planned:
            try:
                rendered = render(template, name, **kwargs)  # type: ignore[arg-type]
            except ValueError:
                continue  # missing input for this template: skip it, never fake it
            queries.append(
                Query(
                    template=template,
                    rendered=rendered,
                    subject_name=name,
                    node_id=node.node_id,
                )
            )
        return queries

    def rank_frontier(
        self, scored: Sequence[tuple[str, int]], cap: int, *, toward_famous: bool = True
    ) -> list[str]:
        """Order candidates by how many independent URLs they recur across.

        Recurrence is a proxy for public prominence, and `toward_famous` sets
        which way to travel. Each side of the search steers toward the fame
        level of the person it is trying to reach: climbing toward a famous
        target, descending toward an obscure one.

        Ranking purely by recurrence — the previous behaviour — is a fame bias
        in disguise. Expanding a YC podcast host surfaced Joe Rogan and Gay
        Talese, because prominent people recur across more pages by definition.
        That is the right instinct when hunting a president and exactly wrong
        when hunting a seed-stage VC.

        Recurrence remains an ordering heuristic only: never stored on an edge,
        never surfaced in output.
        """
        ordered = sorted(scored, key=lambda p: (-p[1] if toward_famous else p[1]))
        return [node_id for node_id, _ in ordered][:cap]
