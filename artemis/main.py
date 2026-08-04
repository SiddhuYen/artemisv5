"""FastAPI app and routes.

    GET  /                -> the console (static SPA)
    POST /connect         -> 202 {"job_id": ...}
    GET  /jobs            -> summaries, newest first
    GET  /jobs/{id}       -> status + live log + stats (+ result when done)
    GET  /jobs/{id}/result-> Result, or 404 if not ready

The network routes below are the console's own surface and touch nothing the
crawl reads. See ``artemis.network`` for why a LinkedIn roster is kept strictly
outside the graph.

    GET  /operator            -> who the roster belongs to, or 404
    PUT  /operator            -> set them
    POST /network/upload      -> a Connections.csv; requires an operator
    GET  /network/contacts    -> the roster
    POST /network/match       -> which of these names are in the roster
    DELETE /network/contacts  -> drop it
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from artemis.config import get_settings
from artemis.graph.relations import RelationCache
from artemis.jobs import InProcessJobRegistry
from artemis.models import (
    ConnectAccepted,
    ConnectRequest,
    ContactView,
    JobStatus,
    JobSummary,
    JobView,
    NetworkMatchRequest,
    NetworkUploadResult,
    OperatorView,
    Result,
    SetOperatorRequest,
    iso_z,
)
from artemis.network import CsvFormatError, NetworkStore, parse_linkedin_csv

STATIC_DIR = Path(__file__).parent / "static"

#: A Connections.csv for a very well-connected account is a few MB of text. The
#: cap is on the read, not on Content-Length, so a lying header buys nothing.
MAX_UPLOAD_BYTES = 32 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.settings = settings
    app.state.jobs = InProcessJobRegistry(settings)
    app.state.network = NetworkStore(settings.cache_dir)
    app.state.jobs.start_reaper()
    try:
        yield
    finally:
        # Without the finally an exception through the yield skips shutdown
        # entirely, and the reaper is still pending when the loop tears down.
        await app.state.jobs.shutdown()


app = FastAPI(
    title="ARTEMIS",
    version="0.1.0",
    description=(
        "Evidence-grounded introduction paths between two people, from public web "
        "data only. Every hop carries a verbatim span from a fetched page and the "
        "URL it came from."
    ),
    lifespan=lifespan,
)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def console() -> Response:
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="console not built")
    # The SPA is edited in place during development; caching it defeats that.
    return FileResponse(index, headers={"Cache-Control": "no-store"})


@app.get("/health")
async def health(relations: bool = False) -> dict[str, object]:
    """Liveness, plus enough to tell whether the cache disk is really in play.

    `relations=true` adds the banked counts. Off by default and deliberately so:
    the Dockerfile probes this every 30s with a 5s timeout, and the people count
    is a UNION over the whole relations table — cheap at ten thousand rows, not
    obviously cheap at a million. A health check that gets slower as the cache
    grows would restart the container for the same reason it did before.
    """
    s = app.state.settings
    # Whether the cache survives a restart is invisible from outside, and both
    # ways of getting it wrong are quiet: point ARTEMIS_CACHE_DIR at no disk and
    # every run is cold forever, or mount one the container cannot write to and
    # DiskCache._write swallows the OSError and caches nothing. Report the path,
    # whether it is actually writable, and how much is banked in it.
    cache_dir = Path(s.cache_dir)
    probe = cache_dir / ".write-probe"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        writable = True
        cache_error = None
    except OSError as exc:
        writable = False
        cache_error = f"{type(exc).__name__}: {exc}"

    payload: dict[str, object] = {
        "ok": True,
        "serper_configured": bool(s.serper_api_key),
        "claude_configured": s.claude_enabled,
        "extraction_model": s.extraction_model,
        "verification_model": s.verification_model,
        "degraded": not s.claude_enabled,
        "cache_dir": str(cache_dir),
        "cache_writable": writable,
        "cache_error": cache_error,
    }
    if relations:
        counts = await RelationCache(
            cache_dir,
            enabled=s.relation_cache_enabled and writable,
            ttl_s=s.relation_cache_ttl_s,
        ).stats()
        payload["relations_cached"] = counts["relations"]
        payload["people_cached"] = counts["people"]
    return payload


# ---------------------------------------------------------------------------
# Connect
# ---------------------------------------------------------------------------


@app.post("/connect", status_code=status.HTTP_202_ACCEPTED, response_model=ConnectAccepted)
async def connect(request: ConnectRequest) -> ConnectAccepted:
    settings = app.state.settings
    if not settings.serper_api_key:
        raise HTTPException(
            status_code=503,
            detail="Serper is not configured; it is the only route to the web.",
        )
    job_id = await app.state.jobs.submit(request)
    return ConnectAccepted(job_id=job_id)


@app.get("/jobs", response_model=list[JobSummary])
async def list_jobs(limit: int = 50) -> list[JobSummary]:
    return [_summarise(state) for state in app.state.jobs.list(limit=max(0, limit))]


@app.get("/jobs/{job_id}", response_model=JobView)
async def get_job(job_id: str) -> JobView:
    state = app.state.jobs.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return JobView(
        id=state.id,
        status=state.status,
        log=state.log,
        result=state.result,
        warnings=state.warnings,
        stats=state.stats,
    )


@app.get("/jobs/{job_id}/result", response_model=Result)
async def get_result(job_id: str) -> Result:
    state = app.state.jobs.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="unknown job")
    if state.status is JobStatus.FAILED:
        raise HTTPException(status_code=409, detail=state.error or "job failed")
    if state.result is None:
        raise HTTPException(status_code=404, detail=f"not ready (status: {state.status.value})")
    return state.result


@app.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_job(job_id: str) -> Response:
    if app.state.jobs.get(job_id) is None:
        raise HTTPException(status_code=404, detail="unknown job")
    await app.state.jobs.cancel(job_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _summarise(state) -> JobSummary:  # type: ignore[no-untyped-def]
    return JobSummary(
        id=state.id,
        status=state.status,
        person_a=state.request.person_a,
        person_b=state.request.person_b,
        context_a=state.request.context_a,
        context_b=state.request.context_b,
        found=state.result.found if state.result else None,
        routes=len(state.result.routes) if state.result else 0,
        created_at=iso_z(state.created_at),
        updated_at=iso_z(state.updated_at),
        stats=state.stats,
        error=state.error,
    )


# ---------------------------------------------------------------------------
# The operator and their roster
# ---------------------------------------------------------------------------


@app.get("/operator", response_model=OperatorView)
async def get_operator() -> OperatorView:
    record = app.state.network.get_operator()
    if record is None:
        raise HTTPException(status_code=404, detail="no operator set")
    return OperatorView(**record)


@app.put("/operator", response_model=OperatorView)
async def put_operator(request: SetOperatorRequest) -> OperatorView:
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="operator name cannot be blank")
    record = await app.state.network.set_operator(name, request.context.strip())
    return OperatorView(**record)


@app.post("/network/upload", response_model=NetworkUploadResult)
async def upload_network(file: UploadFile = File(...)) -> NetworkUploadResult:
    """Ingest a LinkedIn Connections.csv into the roster.

    Refused without an operator. The roster is *somebody's* connections, and an
    unowned one cannot be used as an origin or matched against a route — so the
    console makes setting an operator the gate on this call rather than letting
    contacts pile up under nobody.
    """
    store: NetworkStore = app.state.network
    if store.get_operator() is None:
        raise HTTPException(
            status_code=409,
            detail="set an operator first — the roster is whose connections these are",
        )

    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )
    try:
        contacts = parse_linkedin_csv(raw)
    except CsvFormatError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    counts = await store.add_many(contacts)
    return NetworkUploadResult(**counts, total=store.count(), parsed=len(contacts))


@app.get("/network/contacts", response_model=list[ContactView])
async def list_contacts(search: str = "", limit: int = 0) -> list[ContactView]:
    rows = app.state.network.contacts(search=search, limit=max(0, limit))
    return [ContactView(**c.as_dict()) for c in rows]  # type: ignore[arg-type]


@app.post("/network/match")
async def match_network(request: NetworkMatchRequest) -> dict[str, dict[str, str]]:
    """Which of these names are also in the roster, and on what basis.

    Name matching only — see ``NetworkStore.match``. Nothing here asserts that a
    route's Jane Smith is your Jane Smith, and no answer from this route ever
    reaches the graph or a hop.
    """
    return app.state.network.match(request.names)


@app.delete("/network/contacts")
async def clear_contacts() -> dict[str, int]:
    return {"deleted": await app.state.network.clear()}
