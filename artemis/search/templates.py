"""The fixed query template enum.

Nothing generates freeform query strings — not Claude, not the expansion loop.
Every query issued by ARTEMIS is one of these six templates rendered with
values that came from a Node or the original request.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class QueryTemplate(str, Enum):
    CONTEXT = "context"
    COLLEAGUES = "colleagues"
    GOVERNANCE = "governance"
    EMPLOYER_TEAM = "employer_team"
    PROFILE = "profile"
    DIRECT_BRIDGE = "direct_bridge"
    PAST_EMPLOYERS = "past_employers"
    #: Organisation-shaped, not person-shaped. Person queries never return a
    #: team page, so co-listings had nothing to ground against — the roster is
    #: exactly the document where "who else is here" is written down.
    ORG_ROSTER = "org_roster"


class StrategyAngle(str, Enum):
    """Which angle of a person's network is most likely to bridge to the target.

    Ported from ArtemisV2's search_strategy: a model picks WHICH pre-written
    angle applies, it never writes query text. The query surface stays fully
    deterministic and inspectable, so a bad strategy call can only pick the
    wrong known option — it cannot invent an ungrounded direction.
    """

    CURRENT_EMPLOYER_LEADERSHIP = "current_employer_leadership"
    PAST_EMPLOYERS = "past_employers"
    BOARD_OR_ADVISORY = "board_or_advisory"
    INDUSTRY_PEERS = "industry_peers"
    GENERIC = "generic"


#: Angle -> extra templates. COLLEAGUES always fires, so an angle only ever
#: redirects the *second* query rather than replacing the broad search.
#:
#: INDUSTRY_PEERS maps to nothing on purpose. V2 shipped an industry-peers
#: query and found it dropped both endpoints and returned listicles; the angle
#: stays selectable because it is a real thing for the model to conclude, but
#: it buys no query.
ANGLE_TEMPLATES: dict[StrategyAngle, tuple[QueryTemplate, ...]] = {
    StrategyAngle.CURRENT_EMPLOYER_LEADERSHIP: (QueryTemplate.EMPLOYER_TEAM,),
    StrategyAngle.PAST_EMPLOYERS: (QueryTemplate.PAST_EMPLOYERS,),
    StrategyAngle.BOARD_OR_ADVISORY: (QueryTemplate.GOVERNANCE,),
    StrategyAngle.INDUSTRY_PEERS: (),
    StrategyAngle.GENERIC: (QueryTemplate.GOVERNANCE,),
}


#: Required render() keyword arguments per template, beyond ``name``.
REQUIRED_ARGS: dict[QueryTemplate, tuple[str, ...]] = {
    QueryTemplate.CONTEXT: ("context",),
    QueryTemplate.COLLEAGUES: (),
    QueryTemplate.GOVERNANCE: (),
    QueryTemplate.EMPLOYER_TEAM: ("employer",),
    QueryTemplate.PROFILE: (),
    QueryTemplate.DIRECT_BRIDGE: ("other_endpoint_name",),
    QueryTemplate.PAST_EMPLOYERS: (),
    QueryTemplate.ORG_ROSTER: ("employer",),
}


def render(
    template: QueryTemplate,
    name: str,
    *,
    context: Optional[str] = None,
    employer: Optional[str] = None,
    other_endpoint_name: Optional[str] = None,
) -> str:
    """Render one template. Raises ValueError if a required value is missing.

    Callers skip templates whose inputs they don't have (no employer known ->
    no EMPLOYER_TEAM query) rather than substituting placeholder text.
    """
    args = {"context": context, "employer": employer, "other_endpoint_name": other_endpoint_name}
    for required in REQUIRED_ARGS[template]:
        if not (args.get(required) or "").strip():
            raise ValueError(f"template {template.value} requires a non-empty {required}")

    if template is QueryTemplate.CONTEXT:
        return f'"{name}" "{context}"'
    if template is QueryTemplate.COLLEAGUES:
        return f'"{name}" cofounder OR colleague OR "worked with"'
    if template is QueryTemplate.GOVERNANCE:
        return f'"{name}" board OR advisor OR trustee'
    if template is QueryTemplate.EMPLOYER_TEAM:
        return f'"{name}" "{employer}" team OR staff OR leadership'
    if template is QueryTemplate.PROFILE:
        return f'"{name}" interview OR profile OR bio'
    if template is QueryTemplate.DIRECT_BRIDGE:
        return f'"{name}" "{other_endpoint_name}"'
    if template is QueryTemplate.PAST_EMPLOYERS:
        return f'"{name}" "previously at" OR "formerly at" OR "before joining"'
    if template is QueryTemplate.ORG_ROSTER:
        return f'"{employer}" "our team" OR partners OR people OR leadership'
    raise ValueError(f"unhandled template {template!r}")


def normalize_query(q: str) -> str:
    """Cache key normalisation: collapse whitespace, casefold."""
    return " ".join(q.split()).casefold()
