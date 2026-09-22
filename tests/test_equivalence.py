"""Property-style test: the fast-forward planner must be equivalent to a
naive pass-by-pass WDRR loop over a wide range of random inputs.
"""
from __future__ import annotations

import random
from dataclasses import replace

from app.scheduler import (
    EPS,
    Job,
    SchedulingConfig,
    TenantState,
    plan_batch,
)

NOW = 5_000_000.0


def naive_plan(jobs, states, cursor, now, cfg, free_gpus):
    """Reference implementation: literal round-robin passes, no fast-forward."""
    tenants = {s.tenant_id: replace(s) for s in states}
    queues: dict[str, list[Job]] = {}
    for j in jobs:
        queues.setdefault(j.tenant_id, []).append(replace(j))
    for q in queues.values():
        q.sort(key=lambda j: (j.seq, j.job_id))

    canonical = sorted(tenants)
    selected: list[str] = []
    remaining = free_gpus

    def rotate(active, cur):
        if cur not in active:
            return list(active)
        i = active.index(cur)
        return active[i:] + active[:i]

    for _ in range(20_000):
        active = [t for t in canonical if queues.get(t)]
        if not active:
            break
        any_ran = False
        for tid in rotate(active, cursor):
            st = tenants[tid]
            head = queues[tid][0]
            wait = max(0.0, now - head.submitted_at)
            st.deficit += cfg.quantum(st.weight, wait)
            while queues[tid]:
                job = queues[tid][0]
                if remaining < job.gpus:
                    break
                if job.cost > st.deficit + EPS:
                    break
                st.deficit -= job.cost
                remaining -= job.gpus
                queues[tid].pop(0)
                selected.append(job.job_id)
                any_ran = True
                if not queues[tid]:
                    st.deficit = 0.0
        active_after = [t for t in canonical if queues.get(t)]
        if active_after:
            base = active_after.index(cursor) if cursor in active_after else 0
            cursor = active_after[(base + 1) % len(active_after)]
        if not any_ran:
            fitting = [t for t in active_after
                       if queues[t][0].gpus <= remaining]
            if not fitting:
                break
            # naive: simply loop more passes (handled by outer for)
    blocked = [
        queues[t][0].job_id
        for t in canonical
        if queues.get(t) and remaining < queues[t][0].gpus
    ]
    deficits = {t: round(st.deficit, 9) for t, st in tenants.items()}
    return selected, blocked, deficits, remaining, cursor


def _random_case(rng):
    n_tenants = rng.randint(1, 6)
    tenants = [f"t{i}" for i in range(n_tenants)]
    states = [TenantState(t, rng.choice([0.5, 1.0, 2.0, 3.5]),
                          rng.choice([0.0, 0.0, 30.0, 120.5]))
              for t in tenants]
    jobs = []
    seq = 0
    for t in tenants:
        for _ in range(rng.randint(0, 4)):
            gpus = rng.randint(1, 6)
            jobs.append(Job(
                job_id=f"{t}-j{seq}",
                tenant_id=t,
                gpus=gpus,
                duration_seconds=rng.choice([10, 60, 100, 300, 900]),
                submitted_at=NOW - rng.choice([0, 0, 5, 60, 600, 5000]),
                seq=seq,
            ))
            seq += 1
    cfg = SchedulingConfig(
        total_gpus=8,
        base_quantum=rng.choice([30.0, 60.0]),
        aging_rate=rng.choice([0.0, 0.5, 1.0]),
        aging_cap_factor=rng.choice([2.0, 10.0]),
    )
    free = rng.randint(0, 8)
    cursor = rng.choice(["", tenants[0], tenants[-1]])
    return jobs, states, cursor, cfg, free


def test_matches_naive_implementation_on_random_cases():
    rng = random.Random(20260922)  # fixed seed => reproducible test
    for case_idx in range(120):
        jobs, states, cursor, cfg, free = _random_case(rng)
        got = plan_batch(
            [replace(j) for j in jobs], [replace(s) for s in states],
            cursor, NOW, cfg, free,
        )
        exp = naive_plan(jobs, states, cursor, NOW, cfg, free)
        got_tuple = (
            [p.job_id for p in got.selected],
            got.blocked,
            {k: (0.0 if abs(v) < EPS else v)
             for k, v in got.deficits.items()},
            got.free_gpus,
            got.next_cursor,
        )
        assert got_tuple == exp, (
            f"case {case_idx} mismatch:\n got={got_tuple}\n exp={exp}\n"
            f" jobs={[(j.job_id, j.gpus, j.duration_seconds) for j in jobs]}\n"
            f" free={free} cfg={cfg}"
        )
