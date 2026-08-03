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
