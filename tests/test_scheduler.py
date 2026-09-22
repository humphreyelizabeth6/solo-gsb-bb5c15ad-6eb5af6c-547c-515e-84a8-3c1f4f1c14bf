"""Unit tests for the pure WDRR planner.

Expected orders below were worked out by hand for
BASE_QUANTUM=60, AGING_RATE=1, AGING_CAP_FACTOR=10, cost = gpus*seconds.
"""
from __future__ import annotations

from dataclasses import replace

from app.scheduler import Job, SchedulingConfig, TenantState, plan_batch

CFG = SchedulingConfig(total_gpus=8)
NOW = 1_000_000.0


def j(job_id, tenant, gpus=1, duration=100, submitted_at=NOW):
    return Job(job_id, tenant, gpus, duration, submitted_at)


def s(tenant, weight=1.0, deficit=0.0):
    return TenantState(tenant, weight, deficit)


def ids(plan):
    return [p.job_id for p in plan.selected]


def test_weighted_order_three_to_one():
    # a weight 1, b weight 3; two cost-100 jobs each, all freshly submitted.
    jobs = [j("a1", "a"), j("a2", "a"), j("b1", "b"), j("b2", "b")]
    states = [s("a", 1), s("b", 3)]
    plan = plan_batch(jobs, states, "", NOW, CFG, free_gpus=8)
    # b earns 180/pass vs a's 60/pass: b runs two jobs before a's first.
    assert ids(plan) == ["b1", "b2", "a1", "a2"]
    assert plan.free_gpus == 4
    # queues emptied -> standard DRR drops leftover credit
    assert plan.deficits == {"a": 0.0, "b": 0.0}
    assert plan.selected[0].passes_before_run == 0  # b1 in the very first pass


def test_without_aging_small_job_goes_first():
    # Equal weights; a's job costs 300, b's costs 100; nobody waited.
    jobs = [j("a1", "a", duration=300), j("b1", "b", duration=100)]
    plan = plan_batch(jobs, [s("a"), s("b")], "", NOW, CFG, free_gpus=8)
    # b's small job becomes affordable first (fairness: it is not starved).
    assert ids(plan)[0] == "b1"
    assert ids(plan) == ["b1", "a1"]


def test_aging_reverses_order_for_old_head_job():
    # Same scenario but a's big job has been waiting 600 seconds.
    # aging bonus = min(1*1*600, 10*60*1) = 600 -> a's quantum is 660,
    # enough to run its cost-300 job in the first pass before b.
    jobs = [
        j("a1", "a", duration=300, submitted_at=NOW - 600),
        j("b1", "b", duration=100, submitted_at=NOW),
    ]
    plan = plan_batch(jobs, [s("a"), s("b")], "", NOW, CFG, free_gpus=8)
    assert ids(plan) == ["a1", "b1"]
    a1 = plan.selected[0]
    assert a1.passes_before_run == 0
    assert a1.wait_seconds == 600
    assert a1.quantum == 660.0


def test_aging_bonus_is_capped():
    # Even after a very long wait the bonus cannot exceed cap factor * base.
    jobs = [j("a1", "a", duration=100, submitted_at=NOW - 100_000)]
    plan = plan_batch(jobs, [s("a")], "", NOW, CFG, free_gpus=8)
    assert ids(plan) == ["a1"]
    assert plan.selected[0].quantum == 60.0 + 600.0  # cap = 10 * 60


def test_capacity_is_never_exceeded_and_oversize_heads_block():
    # Two 5-GPU jobs on an 8-GPU cluster: only one fits; the other is blocked.
    jobs = [j("a1", "a", gpus=5, duration=10),
            j("b1", "b", gpus=5, duration=10)]
    plan = plan_batch(jobs, [s("a"), s("b")], "", NOW, CFG, free_gpus=8)
    assert ids(plan) == ["a1"]
    assert plan.free_gpus == 3
    assert plan.blocked == ["b1"]

    # Second batch once 5 GPUs are freed: b1 now runs.
    states2 = [
        s("a", 1, plan.deficits["a"]),
        s("b", 1, plan.deficits["b"]),
    ]
    plan2 = plan_batch([j("b1", "b", gpus=5, duration=10)], states2,
                       plan.next_cursor, NOW + 60, CFG, free_gpus=5)
    assert ids(plan2) == ["b1"]
    assert plan2.free_gpus == 0


def test_blocked_head_keeps_accruing_across_batches():
    # a1 needs 6 GPUs but only 3 are free: it must keep its accumulated
    # quantum so that once capacity frees up it has aged credit.
    jobs = [j("a1", "a", gpus=6, duration=100)]
    plan = plan_batch(jobs, [s("a")], "", NOW, CFG, free_gpus=3)
    assert ids(plan) == []
    assert plan.blocked == ["a1"]
    # one zero pass granted the base quantum even though nothing ran
    assert plan.deficits["a"] == 60.0

    # A second blocked batch accrues another quantum.
    states = [s("a", 1, plan.deficits["a"])]
    plan2 = plan_batch([j("a1", "a", gpus=6, duration=100)], states,
                       plan.next_cursor, NOW + 30, CFG, free_gpus=3)
    assert plan2.deficits["a"] > plan.deficits["a"]


def test_reproducible_same_input_same_plan():
    jobs = [
        j("a1", "a", gpus=2, duration=123),
        j("a2", "a", gpus=1, duration=77),
        j("b1", "b", gpus=3, duration=50, submitted_at=NOW - 42),
        j("c1", "c", gpus=1, duration=500),
    ]
    states = [s("a", 2.5, 12.0), s("b", 1.0), s("c", 4.0, 30.0)]
    p1 = plan_batch(jobs, states, "b", NOW, CFG, free_gpus=7)
    p2 = plan_batch(
        [replace(x) for x in jobs],
        [replace(x) for x in states],
        "b", NOW, CFG, 7,
    )
    assert ids(p1) == ids(p2)
    assert p1.deficits == p2.deficits
    assert p1.next_cursor == p2.next_cursor
    assert p1.passes == p2.passes
    used = sum(x.gpus for x in p1.selected)
    assert used <= 7


def test_empty_plan_when_queue_empty():
    plan = plan_batch([], [], "", NOW, CFG, free_gpus=8)
    assert plan.selected == []
    assert plan.blocked == []
    assert plan.free_gpus == 8
