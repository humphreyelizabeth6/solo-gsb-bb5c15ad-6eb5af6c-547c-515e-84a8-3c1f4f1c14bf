"""Application configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    value = float(raw)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {raw!r}")
    return value


@dataclass(frozen=True)
class Settings:
    # Total number of GPUs shared by all tenants in the cluster.
    total_gpus: int
    # SQLite database file path. ":memory:" is only useful for tests.
    db_path: str
    # Base quantum, in GPU-seconds per weight unit, that a tenant is entitled
    # to during one round-robin pass: base_quantum = BASE_QUANTUM * weight.
    base_quantum: float
    # Extra quantum per weight unit and per second the tenant's head job has
    # been waiting: aging_quantum = AGING_RATE * weight * wait_seconds.
    aging_rate: float
    # Maximum aging bonus as a fraction of the base quantum.
    aging_cap_factor: float

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            total_gpus=_env_int("TOTAL_GPUS", 8),
            db_path=os.getenv("DB_PATH", "/data/scheduler.db"),
            base_quantum=_env_float("BASE_QUANTUM", 60.0),
            aging_rate=_env_float("AGING_RATE", 1.0),
            aging_cap_factor=_env_float("AGING_CAP_FACTOR", 10.0),
        )
