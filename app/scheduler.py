"""Deterministic weighted deficit round-robin (WDRR) batch planner.

This module is pure Python: it takes plain dataclasses describing the current
queue state and returns the selection plan without touching the database or
reading the clock. That makes the scheduling rule easy to unit test and
guarantees the result is reproducible for the same inputs (no randomness).

Cost model
----------
A job's cost is ``gpus * duration_seconds`` (GPU-seconds). Every round-robin
pass a tenant receives a deficit allowance ("quantum") of::

    quantum = BASE_QUANTUM * weight + aging_bonus

where the aging bonus is::

    min(AGING_RATE * weight * head_wait_seconds,
        AGING_CAP_FACTOR * BASE_QUANTUM * weight)

so the longer a tenant's head job has waited, the more deficit it accumulates
each pass; aging is the compensation that prevents small jobs from starving
behind long, heavy jobs. The cap keeps one very old job from monopolising the
cluster.

Capacity safety
---------------
A job is only selected when ``job.gpus <= free_gpus``. If the head job does
not fit, the tenant keeps its deficit and is retried in later passes. The
scheduler never exceeds the cluster's GPU capacity; when only oversize head
jobs remain the batch ends and those jobs are reported as ``blocked``.

Bounded passes (fast-forward)
-----------------------------
Instead of spinning through possibly thousands of no-selection passes, when a
full pass selects nothing but some head jobs still fit the remaining capacity,
the minimum number of passes before any head becomes affordable is computed
analytically (deficits grow linearly while no head moves, so quantum and aging
stay constant), credits are accrued in one jump and scheduling continues.

Determinism
-----------
Tenants are visited in lexicographic tenant_id order; the round-robin start
position (``cursor``) rotates one tenant per pass and is persisted between
batches; jobs run in job_id (FIFO) order. Identical input always yields the
identical plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Floating point slack so values like 119.99999999999999 compare equal to 120.
EPS = 1e-9


@dataclass(frozen=True)
class SchedulingConfig:
    total_gpus: int
    base_quantum: float = 60.0
    aging_rate: float = 1.0
    aging_cap_factor: float = 10.0

    def quantum(self, weight: float, wait_seconds: float) -> float:
        base = self.base_quantum * weight
        aging = self.aging_rate * weight * max(0.0, wait_seconds)
        cap = self.aging_cap_factor * self.base_quantum * weight
        return base + min(aging, cap)


@dataclass
class Job:
    job_id: str
    tenant_id: str
    gpus: int
    duration_seconds: int
    submitted_at: float  # epoch seconds
    seq: int = 0  # submission sequence (FIFO key), ties broken by job_id

    @property
    def cost(self) -> float:
        return float(self.gpus * self.duration_seconds)


@dataclass
class TenantState:
    tenant_id: str
    weight: float
    deficit: float = 0.0


@dataclass
class PlannedJob:
    job_id: str
    tenant_id: str
    gpus: int
    duration_seconds: int
    cost: float
    order: int
    deficit_after: float  # tenant balance right after this job was charged
    wait_seconds: float
    quantum: float  # quantum granted in the pass that selected this job
    passes_before_run: int


@dataclass
class BatchPlan:
    selected: list[PlannedJob] = field(default_factory=list)
    # Head jobs that did not fit the remaining capacity when the batch ended.
    blocked: list[str] = field(default_factory=list)
    # Final deficit ("balance") per tenant after this batch.
    deficits: dict[str, float] = field(default_factory=dict)
    free_gpus: int = 0
    next_cursor: str = ""
    passes: int = 0


def _rotate(active: list[str], cursor: str) -> list[str]:
    """Canonically sorted tenants cycled so that ``cursor`` comes first."""
    if not active:
        return []
    if cursor not in active:
        return list(active)
    i = active.index(cursor)
    return active[i:] + active[:i]


def _next_cursor(active: list[str], cursor: str, shift: int = 1) -> str:
    """Advance ``shift`` canonical positions among currently active tenants.

    A missing cursor behaves as "start of the canonical list", i.e. active[0],
    so shifting by one yields active[1] as expected for the first rotation.
    """
    if not active:
        return cursor
    base = active.index(cursor) if cursor in active else 0
    return active[(base + shift) % len(active)]


def plan_batch(
    jobs: list[Job],
    states: list[TenantState],
    cursor: str,
    now: float,
    cfg: SchedulingConfig,
    free_gpus: int,
) -> BatchPlan:
    """Compute the jobs that can start given ``free_gpus`` idle GPUs.

    ``jobs``  - the tenants' *queued* jobs (FIFO by submission sequence).
    ``states`` - one entry per tenant with persisted weight and deficit.
    ``cursor`` - tenant at which the round starts (persisted between batches).
    """
    tenants: dict[str, TenantState] = {
        s.tenant_id: TenantState(s.tenant_id, s.weight, s.deficit)
        for s in states
    }
    queues: dict[str, list[Job]] = {}
    for j in jobs:
        queues.setdefault(j.tenant_id, []).append(j)
    for q in queues.values():
        q.sort(key=lambda j: (j.seq, j.job_id))

    canonical = sorted(tenants)
    selected: list[PlannedJob] = []
    remaining = free_gpus
    passes = 0  # total round-robin passes (including fast-forwarded ones)

    def head_quantum(tid: str) -> float:
        head = queues[tid][0]
        wait = max(0.0, now - head.submitted_at)
        return cfg.quantum(tenants[tid].weight, wait)

    def visit(tid: str, quantum: float) -> None:
        """Grant one quantum, then run every affordable job that still fits."""
        nonlocal remaining
        st = tenants[tid]
        st.deficit += quantum
        wait = max(0.0, now - queues[tid][0].submitted_at)
        while queues[tid]:
            job = queues[tid][0]
            if remaining < job.gpus:
                return  # does not fit: keep job and deficit, retry later
            if job.cost > st.deficit + EPS:
                return  # cannot afford yet
            st.deficit -= job.cost
            remaining -= job.gpus
            queues[tid].pop(0)
            selected.append(
                PlannedJob(
                    job_id=job.job_id,
                    tenant_id=tid,
                    gpus=job.gpus,
                    duration_seconds=job.duration_seconds,
                    cost=job.cost,
                    order=len(selected),
                    deficit_after=st.deficit,
                    wait_seconds=wait,
                    quantum=quantum,
                    passes_before_run=passes,
                )
            )
            if not queues[tid]:
                # Standard DRR: a tenant whose queue empties loses leftover
                # credit, preventing balance hoarding across idle periods.
                st.deficit = 0.0

    while True:
        active = [tid for tid in canonical if queues.get(tid)]
        if not active:
            break

        # --- one round-robin pass starting at `cursor` ---------------------
        any_ran = False
        for tid in _rotate(active, cursor):
            qq = head_quantum(tid)
            before = len(selected)
            visit(tid, qq)
            if len(selected) != before:
                any_ran = True
        passes += 1

        active_after = [tid for tid in canonical if queues.get(tid)]
        if any_ran:
            cursor = _next_cursor(active_after, cursor, shift=1)
            continue

        # --- pass selected nothing: fast-forward or finish -----------------
        if not active_after:
            break
        fitting = [
            tid for tid in active_after if queues[tid][0].gpus <= remaining
        ]
        if not fitting:
            # Every remaining head needs more GPUs than are idle: the batch
            # can make no further progress until running jobs finish.
            cursor = _next_cursor(active_after, cursor, shift=1)
            break

        # Minimum extra passes (beyond the one just done) until some fitting
        # head becomes affordable. Heads do not move during those passes,
        # hence wait times, quanta and the active set all stay constant.
        # k = first subsequent pass in which that head can run; passes
        # 1..k-1 are empty and get skipped here, pass k is executed by the
        # normal visit loop below (which grants that pass's quantum itself).
        k_min: int | None = None
        qq_map: dict[str, float] = {}
        for tid in active_after:
            qq = head_quantum(tid)
            qq_map[tid] = qq
            if tid in fitting:
                need = queues[tid][0].cost - tenants[tid].deficit
                if need > EPS:
                    k = int(need / qq)
                    if k * qq < need - EPS:
                        k += 1
                    k = max(1, k)
                    if k_min is None or k < k_min:
                        k_min = k

        if k_min is None:
            # Defensive: all fitting heads already affordable yet nothing ran
            # in the pass above should be impossible. Avoid spinning.
            for tid in fitting:
                tenants[tid].deficit = max(
                    tenants[tid].deficit, queues[tid][0].cost
                )
            k_min = 1

        skip = k_min - 1
        for tid in active_after:
            tenants[tid].deficit += skip * qq_map[tid]
        # The completed zero pass rotates once, plus one per skipped pass.
        cursor = _next_cursor(active_after, cursor, shift=1 + skip)
        passes += skip

    blocked = [
        queues[tid][0].job_id
        for tid in canonical
        if queues.get(tid) and remaining < queues[tid][0].gpus
    ]
    deficits = {
        tid: (round(st.deficit, 9) if abs(st.deficit) > EPS else 0.0)
        for tid, st in tenants.items()
    }

    return BatchPlan(
        selected=selected,
        blocked=blocked,
        deficits=deficits,
        free_gpus=remaining,
        next_cursor=cursor,
        passes=passes,
    )
