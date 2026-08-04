"""Retention and memory-release behaviour of the in-process job registry.

These cover the three leaks that let a long-lived container grow until the host
killed it: jobs that were never evicted, a live log that ignored its own cap,
and cancelled crawls whose coroutine frame pinned the whole graph.
"""
from __future__ import annotations

import asyncio
import gc
import weakref
from datetime import timedelta

import pytest

from artemis.config import Settings
from artemis.jobs import InProcessJobRegistry
from artemis.models import ConnectRequest, JobState, JobStatus, Result, Stats, utcnow
from artemis.runtime import JobLog

pytestmark = pytest.mark.asyncio


def settings(**over) -> Settings:
    base = dict(
        serper_api_key=None,
        anthropic_api_key=None,
        cache_enabled=False,
        job_retention_s=60,
    )
    base.update(over)
    return Settings(**base)


def seed(reg: InProcessJobRegistry, job_id: str, status: JobStatus, age_s: float) -> JobState:
    st = JobState(
        id=job_id,
        request=ConnectRequest(person_a="A", person_b="B"),
        status=status,
    )
    st.updated_at = utcnow() - timedelta(seconds=age_s)
    reg._jobs[job_id] = st
    return st


# ---------------------------------------------------------------- retention
async def test_finished_job_past_retention_is_evicted() -> None:
    reg = InProcessJobRegistry(settings(job_retention_s=60))
    seed(reg, "old", JobStatus.DONE, age_s=120)

    async def noop() -> None:
        return None

    task = asyncio.create_task(noop())
    await task
    reg._tasks["old"] = task

    assert reg.reap() == 1
    assert "old" not in reg._jobs, "JobState still retained"
    assert "old" not in reg._tasks, "Task still retained — it pins the coroutine frame"
    assert reg.get("old") is None


async def test_finished_job_within_retention_is_kept() -> None:
    reg = InProcessJobRegistry(settings(job_retention_s=600))
    seed(reg, "recent", JobStatus.DONE, age_s=30)
    assert reg.reap() == 0
    assert "recent" in reg._jobs


@pytest.mark.parametrize("status", [JobStatus.RUNNING, JobStatus.QUEUED])
async def test_live_job_is_never_evicted_however_old(status: JobStatus) -> None:
    """A quiet crawl looks stale: updated_at only advances when it logs."""
    reg = InProcessJobRegistry(settings(job_retention_s=1))
    seed(reg, "live", status, age_s=86_400)
    assert reg.reap() == 0
    assert "live" in reg._jobs


async def test_failed_jobs_are_reaped_too() -> None:
    reg = InProcessJobRegistry(settings(job_retention_s=10))
    seed(reg, "boom", JobStatus.FAILED, age_s=99)
    assert reg.reap() == 1


async def test_reap_retrieves_task_exception_without_raising() -> None:
    """Dropping a failed Task unretrieved makes asyncio warn on GC."""
    reg = InProcessJobRegistry(settings(job_retention_s=1))
    seed(reg, "err", JobStatus.FAILED, age_s=60)

    async def boom() -> None:
        raise RuntimeError("crawl died")

    task = asyncio.create_task(boom())
    await asyncio.gather(task, return_exceptions=True)
    reg._tasks["err"] = task

    assert reg.reap() == 1  # must not re-raise RuntimeError


async def test_reap_of_cancelled_task_does_not_raise() -> None:
    reg = InProcessJobRegistry(settings(job_retention_s=1))
    seed(reg, "cancelled", JobStatus.FAILED, age_s=60)

    async def forever() -> None:
        await asyncio.sleep(3600)

    task = asyncio.create_task(forever())
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    reg._tasks["cancelled"] = task

    assert reg.reap() == 1  # .exception() on a cancelled task would raise


# ---------------------------------------------------------------- the loop
async def test_reaper_loop_runs_and_survives_a_bad_sweep(monkeypatch) -> None:
    reg = InProcessJobRegistry(settings(job_retention_s=20))
    reg._reap_every_s = 0.01

    calls = {"n": 0}
    real = reg.reap

    def flaky() -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("one bad sweep")
        return real()

    monkeypatch.setattr(reg, "reap", flaky)
    reg.start_reaper()
    await asyncio.sleep(0.08)
    assert calls["n"] > 2, f"loop stopped after the exception (calls={calls['n']})"
    await reg.shutdown()


async def test_start_reaper_is_idempotent() -> None:
    reg = InProcessJobRegistry(settings())
    reg.start_reaper()
    first = reg._reaper
    reg.start_reaper()
    assert reg._reaper is first
    await reg.shutdown()


async def test_shutdown_cancels_the_reaper() -> None:
    reg = InProcessJobRegistry(settings())
    reg.start_reaper()
    task = reg._reaper
    assert task is not None
    await reg.shutdown()
    assert task.done()
    assert reg._reaper is None


async def test_reap_interval_is_bounded() -> None:
    assert InProcessJobRegistry(settings(job_retention_s=6 * 3600))._reap_every_s == 300.0
    assert InProcessJobRegistry(settings(job_retention_s=10))._reap_every_s == 1.0


# ---------------------------------------------------------------- log bound
async def test_live_log_respects_max_entries() -> None:
    """The sink used to append a second copy, so the live log was unbounded."""
    s = settings(max_log_entries_per_job=50)
    state = JobState(id="j", request=ConnectRequest(person_a="A", person_b="B"))

    def sink(_entry) -> None:
        return None

    log = JobLog(sink=sink, max_entries=s.max_log_entries_per_job)
    state.log = log.entries
    for i in range(500):
        log("fetch.ok", f"page {i}")

    assert len(state.log) == 50, f"live log grew to {len(state.log)}"


# ------------------------------------------------------- cancellation leak
async def test_cancelled_run_releases_its_crawl_graph(monkeypatch) -> None:
    """The leak that mattered most: a cancelled crawl pinned its whole graph.

    Cancellation keeps the coroutine frame alive even after the Task is dropped,
    so _run has to release its own locals in the finally.
    """
    import artemis.jobs as jobs_mod

    class BigGraph:
        def __init__(self) -> None:
            self.nodes = [{"i": i} for i in range(20_000)]

    box: dict[str, weakref.ref] = {}

    class FakeCtx:
        def __init__(self, *a, **k) -> None: ...
        async def __aenter__(self): return self
        async def __aexit__(self, *exc): return False

    class FakeConnector:
        """Mirrors the real Connector: run() holds self, release() drops state."""

        def __init__(self, *a, **k) -> None:
            self.resolver = object()
            self.graph = BigGraph()
            box["ref"] = weakref.ref(self.graph)

        def release(self) -> None:
            self.graph = None

        async def run(self) -> Result:
            await asyncio.sleep(3600)
            return Result(found=False)

    monkeypatch.setattr(jobs_mod, "SerperProvider", FakeCtx)
    monkeypatch.setattr(jobs_mod, "Fetcher", FakeCtx)
    monkeypatch.setattr(jobs_mod, "ClaudeClient", lambda *a, **k: object())
    monkeypatch.setattr(jobs_mod, "Connector", FakeConnector)

    reg = InProcessJobRegistry(settings(job_retention_s=0))
    job_id = await reg.submit(ConnectRequest(person_a="A", person_b="B"))
    await asyncio.sleep(0.05)
    assert box["ref"]() is not None, "fake connector never built"

    assert await reg.cancel(job_id)
    await asyncio.gather(reg._tasks[job_id], return_exceptions=True)

    reg.reap()
    gc.collect()
    assert box["ref"]() is None, "cancelled crawl still pinning its graph"


async def test_real_connector_release_drops_the_graph() -> None:
    """Keeps the fake above honest: the real release() must free the store."""
    from artemis.graph.connect import Connector

    conn = Connector.__new__(Connector)  # bypass the heavy __init__
    from artemis.graph.store import GraphStore

    conn.store = GraphStore()
    conn.store.nodes.update({f"n{i}": object() for i in range(1000)})
    conn.resolver = object()
    conn.pivots = object()
    conn.providers = [object()]
    conn._expanded = {"a"}
    conn._urls_seen = {"u"}
    ref = weakref.ref(conn.store)

    conn.release()
    gc.collect()

    assert ref() is None, "release() left the old GraphStore reachable"
    assert conn.store.nodes == {}
    assert conn.resolver is None
    assert conn.pivots is None


async def test_reaped_job_state_is_actually_collectable(monkeypatch) -> None:
    """Eviction has to free the JobState, not merely hide it from get()."""
    reg = InProcessJobRegistry(settings(job_retention_s=0))
    st = seed(reg, "gone", JobStatus.DONE, age_s=10)
    st.stats = Stats()
    ref = weakref.ref(st)
    del st

    assert reg.reap() == 1
    gc.collect()
    assert ref() is None, "JobState survived eviction — something still holds it"
