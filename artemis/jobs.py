"""In-process job registry, status, and structured job log.

Builds take minutes of live crawling, so the API is job-based. The registry is a
dict plus asyncio tasks, behind an interface narrow enough to swap for
Redis/Celery later: submit, get, cancel, and a log that streams as it fills.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import timedelta
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
    Route,
    SearchTier,
    Stats,
    utcnow,
)
from artemis.runtime import BudgetLedger, JobLog, RunControl
from artemis.scrape.cache import DiskCache
from artemis.scrape.fetcher import Fetcher
from artemis.search.serper import SerperProvider, SerperUnavailable


#: Terminal for retention purposes. The reaper must never touch QUEUED or
#: RUNNING: _run captures `state` as a local and the sink closure holds it, so
#: evicting a live job orphans a task that keeps crawling — still spending
#: Serper credits and Claude calls — where cancel() can no longer reach it.
_TERMINAL = frozenset({JobStatus.DONE, JobStatus.FAILED})


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
        self._reaper: Optional[asyncio.Task[None]] = None
        #: Jobs asked to finish with what they have. Distinct from cancel():
        #: the crawl stops at its next checkpoint and returns a normal result,
        #: rather than being killed and reported FAILED.
        self._finish_requested: set[str] = set()
        # Swept at a twentieth of the retention window, floored so tests with a
        # tiny retention still sweep and ceilinged at five minutes so a six-hour
        # retention does not mean a six-hour lag before anything is freed.
        self._reap_every_s = max(1.0, min(300.0, settings.job_retention_s / 20))
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

    def request_finish(self, job_id: str) -> bool:
        """Ask a running job to stop at its next checkpoint and return.

        Unlike cancel(), the job finishes normally: it keeps the routes it has
        already found and reports DONE. Cancelling instead would mark it FAILED
        and throw the routes away, which is the opposite of what someone
        pressing "use this route" is asking for.
        """
        state = self._jobs.get(job_id)
        if state is None or state.status in _TERMINAL:
            return False
        self._finish_requested.add(job_id)
        return True

    def start_reaper(self) -> None:
        """Begin enforcing job_retention_s. Idempotent; needs a running loop."""
        if self._reaper is None or self._reaper.done():
            self._reaper = asyncio.create_task(self._reap_forever())

    async def shutdown(self) -> None:
        if self._reaper is not None:
            self._reaper.cancel()
            try:
                await self._reaper
            except asyncio.CancelledError:
                pass
            self._reaper = None
        # Snapshot: cancelling can complete a task synchronously, and mutating
        # _tasks between the cancel pass and the gather would leave a task
        # cancelled but never awaited.
        pending = list(self._tasks.values())
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    # -- retention ----------------------------------------------------------
    def reap(self) -> int:
        """Drop finished jobs past their retention. Returns how many went.

        Both dicts, deliberately. A JobState is ~4.3 MB once its log is full,
        and a *cancelled* task pins its whole coroutine frame — connector,
        graph, fetcher — for as long as the Task object is reachable.
        """
        cutoff = utcnow() - timedelta(seconds=self.s.job_retention_s)
        stale = [
            job_id
            for job_id, state in self._jobs.items()
            if state.status in _TERMINAL and state.updated_at <= cutoff
        ]
        for job_id in stale:
            self._jobs.pop(job_id, None)
            self._finish_requested.discard(job_id)
            task = self._tasks.pop(job_id, None)
            # Retrieve the exception so dropping the Task does not trip
            # asyncio's "Task exception was never retrieved" on GC. A cancelled
            # task re-raises CancelledError from .exception(), so skip those.
            if task is not None and task.done() and not task.cancelled():
                task.exception()
        return len(stale)

    async def _reap_forever(self) -> None:
        while True:
            await asyncio.sleep(self._reap_every_s)
            try:
                self.reap()
            except Exception:  # pragma: no cover - defensive
                # This loop runs for the life of the process. One bad sweep
                # must not silently end retention for every job after it.
                pass

    # -- execution ----------------------------------------------------------
    async def _run(self, job_id: str, request: ConnectRequest) -> None:
        state = self._jobs.get(job_id)
        if state is None:  # reaped or deleted before the task got its first step
            return
        state.status = JobStatus.RUNNING
        state.updated_at = utcnow()

        # The resolver only exists once the connector is built, but the sink is
        # needed before that; this holder lets live stats include merge counts
        # as soon as they start happening.
        holder: dict[str, object] = {"resolver": None}

        def sink(_entry: LogEntry) -> None:
            # No append here: state.log IS log.entries (bound below), so the
            # entry has already landed and is subject to max_entries. Appending
            # a second copy was what made the live log unbounded — JobLog capped
            # its own list but called the sink regardless, so a 12k-event crawl
            # carried 12k entries in memory and only shrank to 5k when it ended.
            resolver = holder["resolver"]
            state.stats = ledger.snapshot(
                merges=getattr(resolver, "merges", 0),
                merges_blocked=getattr(resolver, "merges_blocked", 0),
            )
            state.updated_at = utcnow()

        def publish_routes(routes: list[Route]) -> None:
            # Mid-crawl, so the console can offer the route now instead of after
            # the remaining levels. `result` stays None until the run finishes.
            state.preview_routes = list(routes)
            state.updated_at = utcnow()

        def publish_tiers(tiers: list[SearchTier]) -> None:
            # Copied, not aliased: the connector keeps mutating its own list as
            # the next tier fills, and a poll must never serialise a tier while
            # it is half-rebuilt.
            state.tiers = [t.model_copy(deep=True) for t in tiers]
            state.updated_at = utcnow()

        log = JobLog(sink=sink, max_entries=self.s.max_log_entries_per_job)
        # One list, not two. Bound up front rather than only in the finally, so
        # a job that is still running is already capped instead of being trimmed
        # to size the moment it ends.
        state.log = log.entries
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

        claude = connector = result = None
        try:
            claude = ClaudeClient(self.s, self.cache, ledger, log)
            async with SerperProvider(self.s, self.cache, ledger, log) as provider, Fetcher(
                self.s, self.cache, ledger, log
            ) as fetcher:
                connector = Connector(
                    request, self.s, provider, fetcher, claude, ledger, log,
                    control=RunControl(
                        on_routes=publish_routes,
                        on_tiers=publish_tiers,
                        stop_requested=lambda: job_id in self._finish_requested,
                    ),
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
            # The ranked result supersedes the preview. Leaving both would let
            # the console offer "stop here" against a run that already stopped.
            state.preview_routes = []
            self._finish_requested.discard(job_id)
            if state.stats == Stats():
                state.stats = ledger.snapshot()
            # Cancelling a task pins its coroutine frame — and everything local
            # to it — even after the Task object is dropped. Measured on CPython
            # 3.12: a normally-finished coroutine frees at once, one that raised
            # frees when the Task goes, but a cancelled one never does. The
            # connector owns the whole crawl graph, so every cancelled run
            # leaked a graph for the life of the process. Releasing the names
            # here is what actually frees them; clearing tracebacks does not.
            holder["resolver"] = None
            if connector is not None:
                # Releasing our own name is not enough: run()'s frame holds the
                # connector, so the connector must drop the graph itself.
                connector.release()
            claude = connector = result = None  # noqa: F841 - frees the frame
