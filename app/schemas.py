"""Pydantic request/response models."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

JobStatus = Literal["queued", "running", "completed", "cancelled"]


class TenantCreate(BaseModel):
    tenant_id: str = Field(..., min_length=1, max_length=64, examples=["acme"])
    weight: float = Field(
        ..., gt=0, le=1000, description="Positive scheduling weight"
    )

    @field_validator("tenant_id")
    @classmethod
    def valid_id(cls, v: str) -> str:
        if not all(c.isalnum() or c in "_-" for c in v):
            raise ValueError("tenant_id may only contain letters, digits, _ and -")
        return v


class JobSubmit(BaseModel):
    tenant_id: str = Field(..., min_length=1, max_length=64)
    gpus: int = Field(..., gt=0, description="Number of GPUs required")
    duration_seconds: int = Field(
        ..., gt=0, description="Estimated runtime in seconds"
    )
    weight: float | None = Field(
        None, gt=0, le=1000,
        description="Tenant weight; required on first submission or to update",
    )

    @field_validator("tenant_id")
    @classmethod
    def valid_id(cls, v: str) -> str:
        if not all(c.isalnum() or c in "_-" for c in v):
            raise ValueError("tenant_id may only contain letters, digits, _ and -")
        return v


class JobOut(BaseModel):
    job_id: str
    tenant_id: str
    gpus: int
    duration_seconds: int
    weight_snapshot: float
    status: JobStatus
    submitted_at: float
    started_at: float | None
    finished_at: float | None
    cost: float


class TenantOut(BaseModel):
    tenant_id: str
    weight: float
    deficit: float
    created_at: float


class BalanceOut(BaseModel):
    tenant_id: str
    weight: float
    deficit: float
    queued_jobs: int
    running_jobs: int
    queued_gpus: int
    running_gpus: int


class ScheduledJobOut(BaseModel):
    order: int
    job_id: str
    tenant_id: str
    gpus: int
    duration_seconds: int
    cost: float
    wait_seconds: float
    quantum: float
    deficit_after: float
    passes_before_run: int


class CapacityOut(BaseModel):
    total_gpus: int
    running_gpus: int
    free_gpus: int


class ScheduleResponse(BaseModel):
    scheduled: list[ScheduledJobOut]
    selection_order: list[str]
    blocked: list[str]
    passes: int
    next_cursor: str
    capacity: CapacityOut
    balances: list[BalanceOut]


class QueueResponse(BaseModel):
    capacity: CapacityOut
    queued: list[JobOut]
    balances: list[BalanceOut]
    next_cursor: str


class ScheduledRef(BaseModel):
    order: int
    job_id: str
    tenant_id: str


class HistoryEntry(BaseModel):
    id: int
    scheduled_at: float
    selected: list[ScheduledRef]
    blocked: list[str]


class HealthResponse(BaseModel):
    status: str
    total_gpus: int
    running_gpus: int
    free_gpus: int
