"""Multi-tenant GPU job fair-queuing API.

Implements weighted deficit round-robin (WDRR) with wait-time aging on top
of FastAPI + SQLite. See README.md for the full API and algorithm reference.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse

from .config import Settings
from .database import Database
from .scheduler import Job, SchedulingConfig, TenantState, plan_batch
from . import schemas

settings = Settings.from_env()
db = Database(settings.db_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()
    # One lock serialises all mutating endpoints: capacity checks, planning
    # and state application stay atomic even under concurrent requests.
    app.state.lock = asyncio.Lock()
    app.state.settings = settings
    yield
    await db.close()


app = FastAPI(
    title="Multi-tenant GPU Fair Queue",
    version="1.0.0",
    description=(
        "Submit GPU jobs per tenant and dispatch them with deterministic "
        "weighted deficit round-robin plus wait-time aging. Capacity is never "
        "exceeded."
    ),
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _job_out(row) -> schemas.JobOut:
    return schemas.JobOut(
        job_id=row["job_id"],
        tenant_id=row["tenant_id"],
        gpus=row["gpus"],
        duration_seconds=row["duration_seconds"],
        weight_snapshot=row["weight_snapshot"],
        status=row["status"],
        submitted_at=row["submitted_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        cost=float(row["gpus"] * row["duration_seconds"]),
    )


async def _balances():
    rows = await db.tenant_overview()
    return [
        schemas.BalanceOut(
            tenant_id=r["tenant_id"],
            weight=r["weight"],
            deficit=r["deficit"],
            queued_jobs=r["queued_jobs"],
            running_jobs=r["running_jobs"],
            queued_gpus=r["queued_gpus"],
            running_gpus=r["running_gpus"],
        )
        for r in rows
    ]


async def _capacity(running: int | None = None) -> schemas.CapacityOut:
    if running is None:
        running = await db.running_gpus()
    return schemas.CapacityOut(
        total_gpus=settings.total_gpus,
        running_gpus=running,
        free_gpus=settings.total_gpus - running,
    )


# --------------------------------------------------------------------------- #
# Meta endpoints
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=schemas.HealthResponse, tags=["meta"])
async def health():
    running = await db.running_gpus()
    return schemas.HealthResponse(
        status="ok",
        total_gpus=settings.total_gpus,
        running_gpus=running,
        free_gpus=settings.total_gpus - running,
    )


# --------------------------------------------------------------------------- #
# Tenants & submission
# --------------------------------------------------------------------------- #
@app.post("/tenants", response_model=schemas.TenantOut, tags="tenants",
          status_code=201)
async def create_tenant(body: schemas.TenantCreate, request: Request):
    async with request.app.state.lock:
        now = time.time()
        await db.upsert_tenant(body.tenant_id, body.weight, now)
        for r in await db.tenant_states():
            if r["tenant_id"] == body.tenant_id:
                return schemas.TenantOut(
                    tenant_id=r["tenant_id"],
                    weight=r["weight"],
                    deficit=r["deficit"],
                    created_at=r["created_at"],
                )
    raise HTTPException(500, "tenant lookup failed")


@app.post("/jobs", response_model=schemas.JobOut, tags="jobs",
          status_code=201)
async def submit_job(body: schemas.JobSubmit, request: Request):
    if body.gpus > settings.total_gpus:
        raise HTTPException(
            422,
            f"job requires {body.gpus} GPUs but the cluster only has "
            f"{settings.total_gpus}",
        )
    async with request.app.state.lock:
        now = time.time()
        weights = {
            r["tenant_id"]: r["weight"] for r in await db.tenant_states()
        }
        if body.tenant_id not in weights:
            if body.weight is None:
                raise HTTPException(
                    400,
                    f"unknown tenant {body.tenant_id!r}; provide 'weight' on "
                    f"first submission or create it via POST /tenants",
                )
            await db.upsert_tenant(body.tenant_id, body.weight, now)
            weight = body.weight
        elif body.weight is not None:
            await db.upsert_tenant(body.tenant_id, body.weight, now)
            weight = body.weight
        else:
            weight = weights[body.tenant_id]

        job_id = f"job-{uuid.uuid4().hex[:12]}"
        await db.insert_job(
            job_id, body.tenant_id, body.gpus, body.duration_seconds,
            weight, now,
        )
        row = await db.get_job(job_id)
        return _job_out(row)


# --------------------------------------------------------------------------- #
# Scheduling
# --------------------------------------------------------------------------- #
@app.post("/schedule", response_model=schemas.ScheduleResponse,
          tags=["scheduler"])
async def schedule_next_batch(request: Request):
    async with request.app.state.lock:
        now = time.time()

        queued_rows = await db.queued_jobs()
        state_rows = await db.tenant_states()
        running = await db.running_gpus()
        free_gpus = settings.total_gpus - running

        jobs = [
            Job(
                job_id=r["job_id"],
                tenant_id=r["tenant_id"],
                gpus=r["gpus"],
                duration_seconds=r["duration_seconds"],
                submitted_at=r["submitted_at"],
                seq=r["seq"],
            )
            for r in queued_rows
        ]
        states = [
            TenantState(
                tenant_id=r["tenant_id"],
                weight=r["weight"],
                deficit=r["deficit"],
            )
            for r in state_rows
        ]
        cfg = SchedulingConfig(
            total_gpus=settings.total_gpus,
            base_quantum=settings.base_quantum,
            aging_rate=settings.aging_rate,
            aging_cap_factor=settings.aging_cap_factor,
        )
        cursor = await db.get_cursor()
        plan = plan_batch(jobs, states, cursor, now, cfg, free_gpus)

        if plan.selected:
            selected_payload = json.dumps(
                [
                    {"order": p.order, "job_id": p.job_id,
                     "tenant_id": p.tenant_id}
                    for p in plan.selected
                ]
            )
            blocked_payload = json.dumps(plan.blocked)
        else:
            selected_payload = None
            blocked_payload = None
        # Persist even when nothing ran: aging accrued and the round-robin
        # cursor rotated, and both must survive restarts for determinism.
        await db.apply_batch(
            [p.job_id for p in plan.selected],
            plan.deficits,
            plan.next_cursor,
            now,
            selected_payload,
            blocked_payload,
        )

        scheduled_out = [
            schemas.ScheduledJobOut(
                order=p.order,
                job_id=p.job_id,
                tenant_id=p.tenant_id,
                gpus=p.gpus,
                duration_seconds=p.duration_seconds,
                cost=p.cost,
                wait_seconds=round(p.wait_seconds, 6),
                quantum=round(p.quantum, 6),
                deficit_after=round(p.deficit_after, 6),
                passes_before_run=p.passes_before_run,
            )
            for p in plan.selected
        ]
        new_running = running + sum(p.gpus for p in plan.selected)
        return schemas.ScheduleResponse(
            scheduled=scheduled_out,
            selection_order=[p.job_id for p in plan.selected],
            blocked=plan.blocked,
            passes=plan.passes,
            next_cursor=plan.next_cursor,
            capacity=await _capacity(new_running),
            balances=await _balances(),
        )


# --------------------------------------------------------------------------- #
# Lifecycle: complete / cancel
# --------------------------------------------------------------------------- #
@app.post("/jobs/{job_id}/complete", response_model=schemas.JobOut, tags="jobs")
async def complete_job(job_id: str, request: Request):
    async with request.app.state.lock:
        prior = await db.set_job_terminal(job_id, "completed", time.time())
        if prior is None:
            raise HTTPException(404, f"job {job_id!r} not found")
        if prior not in ("running", "completed"):
            raise HTTPException(
                409,
                f"job {job_id!r} is {prior!r}, only a running job can complete",
            )
        return _job_out(await db.get_job(job_id))


@app.post("/jobs/{job_id}/cancel", response_model=schemas.JobOut, tags="jobs")
async def cancel_job(job_id: str, request: Request):
    async with request.app.state.lock:
        prior = await db.set_job_terminal(job_id, "cancelled", time.time())
        if prior is None:
            raise HTTPException(404, f"job {job_id!r} not found")
        if prior in ("completed", "cancelled"):
            if prior == "cancelled":
                return _job_out(await db.get_job(job_id))  # idempotent
            raise HTTPException(
                409, f"job {job_id!r} already completed and cannot be cancelled"
            )
        return _job_out(await db.get_job(job_id))


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
@app.get("/queue", response_model=schemas.QueueResponse, tags="queries")
async def get_queue(request: Request):
    async with request.app.state.lock:
        running = await db.running_gpus()
        return schemas.QueueResponse(
            capacity=await _capacity(running),
            queued=[_job_out(r) for r in await db.queued_jobs()],
            balances=await _balances(),
            next_cursor=await db.get_cursor(),
        )


@app.get("/jobs", response_model=list[schemas.JobOut], tags="queries")
async def list_jobs(tenant_id: str | None = None, status: str | None = None):
    valid = {"queued", "running", "completed", "cancelled"}
    if status is not None and status not in valid:
        raise HTTPException(422, f"status must be one of {sorted(valid)}")
    if tenant_id is not None:
        statuses = (status,) if status else None
        rows = await db.jobs_of_tenant(tenant_id, statuses)
        if not rows:
            # Distinguish "unknown tenant" from "tenant with no jobs".
            known = {r["tenant_id"] for r in await db.tenant_states()}
            if tenant_id not in known:
                raise HTTPException(404, f"tenant {tenant_id!r} not found")
    elif status:
        rows = [r for r in await db.all_jobs() if r["status"] == status]
    else:
        rows = await db.all_jobs()
    return [_job_out(r) for r in rows]


@app.get("/jobs/{job_id}", response_model=schemas.JobOut, tags="queries")
async def get_job(job_id: str):
    row = await db.get_job(job_id)
    if row is None:
        raise HTTPException(404, f"job {job_id!r} not found")
    return _job_out(row)


@app.get("/balances", response_model=list[schemas.BalanceOut], tags="queries")
async def get_balances():
    return await _balances()


@app.get("/schedule/history", response_model=list[schemas.HistoryEntry],
         tags="queries")
async def schedule_history(limit: int = 20):
    limit = max(1, min(limit, 200))
    out = []
    for r in await db.list_history(limit):
        out.append(
            schemas.HistoryEntry(
                id=r["id"],
                scheduled_at=r["scheduled_at"],
                selected=json.loads(r["selected"]),
                blocked=json.loads(r["blocked"]),
            )
        )
    return out
