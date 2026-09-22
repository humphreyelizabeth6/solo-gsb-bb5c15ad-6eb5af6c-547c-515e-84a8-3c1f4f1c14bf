"""Asynchronous SQLite persistence layer.

A single connection guarded by one asyncio.Lock serialises every mutating
operation, which is exactly the serialization a scheduler needs: batch
planning and state application must be atomic with respect to GPU capacity.
"""
from __future__ import annotations

import aiosqlite

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS tenants (
    tenant_id  TEXT PRIMARY KEY,
    weight     REAL NOT NULL CHECK (weight > 0),
    -- Persisted WDRR deficit ("balance") carried between schedule batches.
    deficit    REAL NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id           TEXT PRIMARY KEY,
    tenant_id        TEXT NOT NULL REFERENCES tenants(tenant_id),
    gpus             INTEGER NOT NULL CHECK (gpus > 0),
    duration_seconds INTEGER NOT NULL CHECK (duration_seconds > 0),
    weight_snapshot  REAL NOT NULL,
    status           TEXT NOT NULL CHECK (status IN ('queued','running','completed','cancelled')),
    submitted_at     REAL NOT NULL,
    started_at       REAL,
    finished_at      REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_tenant ON jobs(tenant_id);

CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS batch_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    scheduled_at REAL NOT NULL,
    selected     TEXT NOT NULL,
    blocked      TEXT NOT NULL
);
"""


class Database:
    def __init__(self, path: str):
        self.path = path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        # isolation_level=None => sqlite autocommit; transactions are opened
        # explicitly (apply_batch uses BEGIN IMMEDIATE for atomicity).
        self._db = aiosqlite.connect(self.path, isolation_level=None)
        await self._db.__aenter__()
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.commit()
            await self._db.__aexit__(None, None, None)
            self._db = None

    @property
    def conn(self) -> aiosqlite.Connection:
        assert self._db is not None, "Database.connect() must be awaited first"
        return self._db

    async def upsert_tenant(self, tenant_id: str, weight: float, now: float) -> bool:
        """Create a tenant or update its weight. Returns True if newly created."""
        async with self.conn.execute(
            "SELECT 1 FROM tenants WHERE tenant_id = ?", (tenant_id,)
        ) as cur:
            existed = await cur.fetchone()
        if existed:
            await self.conn.execute(
                "UPDATE tenants SET weight = ? WHERE tenant_id = ?",
                (weight, tenant_id),
            )
        else:
            await self.conn.execute(
                "INSERT INTO tenants (tenant_id, weight, created_at) "
                "VALUES (?, ?, ?)",
                (tenant_id, weight, now),
            )
        await self.conn.commit()
        return not existed

    async def tenant_states(self) -> list[aiosqlite.Row]:
        async with self.conn.execute(
            "SELECT tenant_id, weight, deficit, created_at FROM tenants "
            "ORDER BY tenant_id"
        ) as cur:
            return list(await cur.fetchall())

    async def tenant_overview(self) -> list[aiosqlite.Row]:
        """Per-tenant deficit plus queued/running job and GPU aggregates."""
        async with self.conn.execute(
            """
            SELECT t.tenant_id,
                   t.weight,
                   t.deficit,
                   COUNT(CASE WHEN j.status = 'queued'  THEN 1 END) AS queued_jobs,
                   COUNT(CASE WHEN j.status = 'running' THEN 1 END) AS running_jobs,
                   COALESCE(SUM(CASE WHEN j.status = 'queued'
                                     THEN j.gpus END), 0) AS queued_gpus,
                   COALESCE(SUM(CASE WHEN j.status = 'running'
                                     THEN j.gpus END), 0) AS running_gpus
            FROM tenants t
            LEFT JOIN jobs j ON j.tenant_id = t.tenant_id
            GROUP BY t.tenant_id
            ORDER BY t.tenant_id
            """
        ) as cur:
            return list(await cur.fetchall())

    async def insert_job(
        self,
        job_id: str,
        tenant_id: str,
        gpus: int,
        duration_seconds: int,
        weight: float,
        now: float,
    ) -> None:
        await self.conn.execute(
            """
            INSERT INTO jobs (job_id, tenant_id, gpus, duration_seconds,
                              weight_snapshot, status, submitted_at)
            VALUES (?, ?, ?, ?, ?, 'queued', ?)
            """,
            (job_id, tenant_id, gpus, duration_seconds, weight, now),
        )
        await self.conn.commit()

    async def get_job(self, job_id: str) -> aiosqlite.Row | None:
        async with self.conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ) as cur:
            return await cur.fetchone()

    async def queued_jobs(self) -> list[aiosqlite.Row]:
        async with self.conn.execute(
            "SELECT *, rowid AS seq FROM jobs WHERE status = 'queued' "
            "ORDER BY rowid"
        ) as cur:
            return list(await cur.fetchall())

    async def running_gpus(self) -> int:
        async with self.conn.execute(
            "SELECT COALESCE(SUM(gpus), 0) AS n FROM jobs WHERE status = 'running'"
        ) as cur:
            row = await cur.fetchone()
            return int(row["n"])

    async def all_jobs(self) -> list[aiosqlite.Row]:
        async with self.conn.execute(
            "SELECT * FROM jobs ORDER BY submitted_at DESC, job_id"
        ) as cur:
            return list(await cur.fetchall())

    async def jobs_of_tenant(
        self, tenant_id: str, statuses: tuple[str, ...] | None = None
    ) -> list[aiosqlite.Row]:
        if statuses is None:
            async with self.conn.execute(
                "SELECT * FROM jobs WHERE tenant_id = ? "
                "ORDER BY submitted_at DESC, job_id",
                (tenant_id,),
            ) as cur:
                return list(await cur.fetchall())
        placeholders = ",".join("?" for _ in statuses)
        async with self.conn.execute(
            f"SELECT * FROM jobs WHERE tenant_id = ? AND status IN ({placeholders}) "
            f"ORDER BY submitted_at DESC, job_id",
            (tenant_id, *statuses),
        ) as cur:
            return list(await cur.fetchall())

    async def queued_count(self, tenant_id: str) -> int:
        async with self.conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE tenant_id = ? "
            "AND status = 'queued'",
            (tenant_id,),
        ) as cur:
            row = await cur.fetchone()
            return int(row["n"])

    async def set_job_terminal(
        self, job_id: str, status: str, now: float
    ) -> str | None:
        """Mark a job completed/cancelled. Returns its prior status, or None."""
        async with self.conn.execute(
            "SELECT status, tenant_id FROM jobs WHERE job_id = ?", (job_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        prior = row["status"]
        if prior == status:
            return prior  # idempotent: nothing to change
        await self.conn.execute(
            "UPDATE jobs SET status = ?, finished_at = ? WHERE job_id = ?",
            (status, now, job_id),
        )
        # Standard DRR: once a tenant's queue empties, leftover credit is lost.
        if await self.queued_count(row["tenant_id"]) == 0:
            await self.conn.execute(
                "UPDATE tenants SET deficit = 0 WHERE tenant_id = ?",
                (row["tenant_id"],),
            )
        await self.conn.commit()
        return prior

    async def get_cursor(self) -> str:
        async with self.conn.execute(
            "SELECT value FROM kv WHERE key = 'rr_cursor'"
        ) as cur:
            row = await cur.fetchone()
        return row["value"] if row else ""

    async def apply_batch(
        self,
        selected_ids: list[str],
        deficits: dict[str, float],
        cursor: str,
        now: float,
        selected_json: str | None = None,
        blocked_json: str | None = None,
    ) -> None:
        """Persist one scheduling plan atomically (one IMMEDIATE transaction).

        When nothing was selected, deficit/cursor still advanced (aging
        accrued, round rotated) but no history row is written.
        """
        db = self.conn
        await db.execute("BEGIN IMMEDIATE")
        try:
            for jid in selected_ids:
                await db.execute(
                    "UPDATE jobs SET status = 'running', started_at = ? "
                    "WHERE job_id = ? AND status = 'queued'",
                    (now, jid),
                )
            for tid, deficit in deficits.items():
                await db.execute(
                    "UPDATE tenants SET deficit = ? WHERE tenant_id = ?",
                    (deficit, tid),
                )
            await db.execute(
                "INSERT INTO kv(key, value) VALUES('rr_cursor', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (cursor,),
            )
            if selected_json is not None:
                await db.execute(
                    "INSERT INTO batch_history(scheduled_at, selected, blocked) "
                    "VALUES (?, ?, ?)",
                    (now, selected_json, blocked_json or "[]"),
                )
            await db.commit()
        except Exception:
            await db.rollback()
            raise

    async def list_history(self, limit: int) -> list[aiosqlite.Row]:
        async with self.conn.execute(
            "SELECT * FROM batch_history ORDER BY id DESC LIMIT ?", (limit,)
        ) as cur:
            return list(await cur.fetchall())
