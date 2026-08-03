"""In-process job registry, status, and structured job log.

Builds take minutes of live crawling, so the API is job-based. The registry is a
dict plus asyncio tasks, behind an interface narrow enough to swap for
Redis/Celery later: submit, get, cancel, and a log that streams as it fills.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Optional, Protocol

from artemis.config import Settings
from artemis.extract.client import ClaudeClient
from artemis.graph.connect import Connector
from artemis.models import (
    ConnectRequest,
    JobState,
    JobStatus,
    LogEntry,
    Result,
    Stats,
    utcnow,
)
from artemis.runtime import BudgetLedger, JobLog
from artemis.scrape.cache import DiskCache
from artemis.scrape.fetcher import Fetcher
from artemis.search.serper import SerperProvider, SerperUnavailable


class JobStore(Protocol):
    async def submit(self, request: ConnectRequest) -> str: ...
    def get(self, job_id: str) -> Optional[JobState]: ...
    def list(self, limit: int = 0) -> list[JobState]: ...
    async def cancel(self, job_id: str) -> bool: ...


class InProcessJobRegistry:
    """Dict + asyncio tasks. One process, no durability — deliberately."""

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._jobs: dict[str, JobState] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self.cache = DiskCache(
            settings.cache_dir, enabled=settings.cache_enabled, ttl_s=settings.cache_ttl_s
        )

    # -- interface ----------------------------------------------------------
    async def submit(self, request: ConnectRequest) -> str:
        job_id = uuid.uuid4().hex[:16]
        self._jobs[job_id] = JobState(id=job_id, request=request)
        self._tasks[job_id] = asyncio.create_task(self._run(job_id, request))
        return job_id

    def get(self, job_id: str) -> Optional[JobState]:
        state = self._jobs.get(job_id)
        return self._with_live_elapsed(state) if state else None

    def list(self, limit: int = 0) -> list[JobState]:
        """Newest first. The UI's job grid reads this; nothing else does."""
        ordered = sorted(self._jobs.values(), key=lambda s: s.created_at, reverse=True)
        chosen = ordered[:limit] if limit > 0 else ordered
        return [self._with_live_elapsed(s) for s in chosen]

    @staticmethod
    def _with_live_elapsed(state: JobState) -> JobState:
        """Report elapsed time as measured, not as last recorded.

        `stats.elapsed_s` is written by the log sink, so it only advances when
        the job has something to say. A crawl waiting on a batch of extractions
        emits nothing for minutes and its timer sits still — which reads as
        "stalled" when it is merely quiet, and reads as "26 seconds" when the
        job has genuinely been running for ten minutes.

        Every other stat is a count and is correct as recorded. Only elapsed is
        a clock, so only elapsed is recomputed on read.
        """
        if state.status is JobStatus.RUNNING:
            state.stats.elapsed_s = round((utcnow() - state.created_at).total_seconds(), 2)
        return state

    async def cancel(self, job_id: str) -> bool:
        task = self._tasks.get(job_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    async def shutdown(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    # -- execution ----------------------------------------------------------
    async def _run(self, job_id: str, request: ConnectRequest) -> None:
        state = self._jobs[job_id]
        state.status = JobStatus.RUNNING
        state.updated_at = utcnow()

        # The resolver only exists once the connector is built, but the sink is
        # needed before that; this holder lets live stats include merge counts
        # as soon as they start happening.
        holder: dict[str, object] = {"resolver": None}

        def sink(entry: LogEntry) -> None:
            state.log.append(entry)
            resolver = holder["resolver"]
            state.stats = ledger.snapshot(
                merges=getattr(resolver, "merges", 0),
                merges_blocked=getattr(resolver, "merges_blocked", 0),
            )
            state.updated_at = utcnow()

        log = JobLog(sink=sink, max_entries=self.s.max_log_entries_per_job)
        budget = request.budget
        ledger = BudgetLedger(
            max_serper_credits=(budget.max_serper_credits if budget else None)
            or self.s.max_serper_credits,
            max_fetches=(budget.max_fetches if budget else None) or self.s.max_fetches,
            max_claude_calls=(budget.max_claude_calls if budget else None)
            or self.s.max_claude_calls,
            max_nodes_expanded=(budget.max_nodes_expanded if budget else None)
            or self.s.max_nodes_expanded,
            wall_clock_s=(budget.wall_clock_s if budget else None) or self.s.wall_clock_s,
        )

        try:
            claude = ClaudeClient(self.s, self.cache, ledger, log)
            async with SerperProvider(self.s, self.cache, ledger, log) as provider, Fetcher(
                self.s, self.cache, ledger, log
            ) as fetcher:
                connector = Connector(
                    request, self.s, provider, fetcher, claude, ledger, log
                )
                holder["resolver"] = connector.resolver
                result: Result = await asyncio.wait_for(
                    connector.run(), timeout=ledger.wall_clock_s + 60
                )

            state.result = result
            state.stats = result.stats
            state.warnings = result.warnings
            state.status = JobStatus.DONE
            log("job.finished", "done", found=result.found, routes=len(result.routes))

        except SerperUnavailable as exc:
            # Serper is the only door to the web. There is no fallback.
            state.status = JobStatus.FAILED
            state.error = str(exc)
            state.warnings.append(str(exc))
            state.stats = ledger.snapshot()
            log.error("job.failed", str(exc))
        except asyncio.CancelledError:
            state.status = JobStatus.FAILED
            state.error = "cancelled"
            state.stats = ledger.snapshot()
            log.error("job.cancelled", "cancelled")
            raise
        except (asyncio.TimeoutError, Exception) as exc:  # noqa: B014 - explicit for clarity
            state.status = JobStatus.FAILED
            state.error = f"{type(exc).__name__}: {exc}"
            state.stats = ledger.snapshot()
            log.error("job.failed", state.error)
        finally:
            state.log = log.entries
            state.updated_at = utcnow()
            if state.stats == Stats():
                state.stats = ledger.snapshot()
