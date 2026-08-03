"""FastAPI app and routes.

    POST /connect         -> 202 {"job_id": ...}
    GET  /jobs/{id}       -> status + live log + stats (+ result when done)
    GET  /jobs/{id}/result-> Result, or 404 if not ready
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Response, status

from artemis.config import get_settings
from artemis.jobs import InProcessJobRegistry
from artemis.models import ConnectAccepted, ConnectRequest, JobStatus, JobView, Result


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.settings = settings
    app.state.jobs = InProcessJobRegistry(settings)
    yield
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


@app.get("/health")
async def health() -> dict[str, object]:
    s = app.state.settings
    return {
        "ok": True,
        "serper_configured": bool(s.serper_api_key),
        "claude_configured": s.claude_enabled,
        "extraction_model": s.extraction_model,
        "verification_model": s.verification_model,
        "degraded": not s.claude_enabled,
    }


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
