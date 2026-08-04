"""Per-job runtime: the budget ledger and the structured log sink.

Budgets are cross-cutting — search, fetch, and extraction all spend against the
same job. Rather than thread five counters through every call signature, one
ledger is created per job and passed down. Nothing spends without asking.

Spending returns False rather than raising, so a crawl that hits a ceiling
stops cleanly and reports which ceiling it was, instead of unwinding through
half-finished work.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from artemis.models import LogEntry, LogLevel, Stats


class BudgetExceeded(RuntimeError):
    """Raised only where a caller genuinely cannot continue (e.g. Serper hard stop)."""

    def __init__(self, limit: str) -> None:
        super().__init__(f"budget exhausted: {limit}")
        self.limit = limit


@dataclass
class RunControl:
    """Lets a crawl publish routes early and be stopped without being killed.

    The crawl finds a route well before it finishes: the frontiers meet, then it
    spends the rest of its depth budget looking for a shorter one. That is the
    right default — the first path found is not usually the best — but it means
    a run that dies in the meantime reports found:false, which reads exactly
    like "no path exists".

    So routes are published the moment they exist, and stopping is cooperative:
    `stop_requested` is polled at the same checkpoints, and a crawl that sees it
    returns the routes it has as a normal result. That is the difference between
    this and cancel(), which kills the task and reports the job FAILED.
    """

    #: Called with the ranked routes each time the set changes. Must not raise.
    on_routes: Callable[[Any], None] = lambda routes: None
    #: Called with the search's tiers as they fill. Same contract as on_routes:
    #: the console polls every 1.2s and needs to see the tier being worked on,
    #: not a structure that only appears once the run is over.
    on_tiers: Callable[[Any], None] = lambda tiers: None
    #: Polled at every route checkpoint. True means finish now and return.
    stop_requested: Callable[[], bool] = lambda: False

    def publish(self, routes: Any) -> None:
        try:
            self.on_routes(routes)
        except Exception:  # pragma: no cover - a console update is never fatal
            pass

    def publish_tiers(self, tiers: Any) -> None:
        try:
            self.on_tiers(tiers)
        except Exception:  # pragma: no cover - a console update is never fatal
            pass

    def should_stop(self) -> bool:
        try:
            return bool(self.stop_requested())
        except Exception:  # pragma: no cover
            return False


@dataclass
class BudgetLedger:
    max_serper_credits: int
    max_fetches: int
    max_claude_calls: int
    max_nodes_expanded: int
    wall_clock_s: float

    serper_queries: int = 0
    serper_credits_used: int = 0
    pages_fetched: int = 0
    claude_calls: int = 0
    nodes_expanded: int = 0

    started_at: float = field(default_factory=time.monotonic)
    hit: set[str] = field(default_factory=set)

    # -- clock --------------------------------------------------------------
    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    def out_of_time(self) -> bool:
        if self.elapsed_s >= self.wall_clock_s:
            self.hit.add("wall_clock_s")
            return True
        return False

    # -- spending -----------------------------------------------------------
    def try_spend_serper(self, credits: int = 1) -> bool:
        if self.out_of_time():
            return False
        if self.serper_credits_used + credits > self.max_serper_credits:
            self.hit.add("max_serper_credits")
            return False
        self.serper_queries += credits
        self.serper_credits_used += credits
        return True

    def try_spend_fetch(self) -> bool:
        if self.out_of_time():
            return False
        if self.pages_fetched + 1 > self.max_fetches:
            self.hit.add("max_fetches")
            return False
        self.pages_fetched += 1
        return True

    def try_spend_claude(self) -> bool:
        if self.out_of_time():
            return False
        if self.claude_calls + 1 > self.max_claude_calls:
            self.hit.add("max_claude_calls")
            return False
        self.claude_calls += 1
        return True

    def try_spend_node(self) -> bool:
        if self.out_of_time():
            return False
        if self.nodes_expanded + 1 > self.max_nodes_expanded:
            self.hit.add("max_nodes_expanded")
            return False
        self.nodes_expanded += 1
        return True

    # -- reporting ----------------------------------------------------------
    @property
    def limits_hit(self) -> list[str]:
        return sorted(self.hit)

    def snapshot(self, *, merges: int = 0, merges_blocked: int = 0) -> Stats:
        return Stats(
            serper_queries=self.serper_queries,
            serper_credits_used=self.serper_credits_used,
            pages_fetched=self.pages_fetched,
            claude_calls=self.claude_calls,
            nodes_expanded=self.nodes_expanded,
            merges=merges,
            merges_blocked=merges_blocked,
            elapsed_s=round(self.elapsed_s, 2),
        )


class JobLog:
    """Append-only structured log a poller can watch while the crawl runs."""

    def __init__(
        self,
        sink: Optional[Callable[[LogEntry], None]] = None,
        max_entries: int = 5000,
    ) -> None:
        self.entries: list[LogEntry] = []
        self._sink = sink
        self._max = max_entries
        self._dropped = 0

    def __call__(
        self,
        event: str,
        message: str = "",
        level: LogLevel = LogLevel.INFO,
        **data: Any,
    ) -> None:
        entry = LogEntry(event=event, message=message, level=level, data=data)
        if len(self.entries) < self._max:
            self.entries.append(entry)
        else:
            self._dropped += 1
        if self._sink is not None:
            self._sink(entry)

    def warn(self, event: str, message: str = "", **data: Any) -> None:
        self(event, message, level=LogLevel.WARNING, **data)

    def error(self, event: str, message: str = "", **data: Any) -> None:
        self(event, message, level=LogLevel.ERROR, **data)

    @property
    def dropped(self) -> int:
        return self._dropped
