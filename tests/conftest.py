import os
import sqlite3
import tempfile

import pytest
from fastapi.testclient import TestClient

_tmpdir = tempfile.mkdtemp(prefix="gpu-scheduler-test-")
DB_PATH = os.path.join(_tmpdir, "test.db")
os.environ["DB_PATH"] = DB_PATH
os.environ["TOTAL_GPUS"] = "8"
os.environ["BASE_QUANTUM"] = "60"
os.environ["AGING_RATE"] = "1"
os.environ["AGING_CAP_FACTOR"] = "10"

from app.main import app  # noqa: E402


def _execute(sql: str, params: tuple = ()) -> None:
    """Direct stdlib access to the same SQLite file (WAL allows this)."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def wipe() -> None:
    for table in ("jobs", "tenants", "kv", "batch_history"):
        _execute(f"DELETE FROM {table}")


def set_submitted_at(job_id: str, ts: float) -> None:
    """Backdate a job's submission time to control wait-based aging."""
    _execute("UPDATE jobs SET submitted_at = ? WHERE job_id = ?", (ts, job_id))


@pytest.fixture()
def client():
    # `with` triggers the lifespan (DB connect/close) for every test.
    with TestClient(app) as c:
        wipe()
        yield c
