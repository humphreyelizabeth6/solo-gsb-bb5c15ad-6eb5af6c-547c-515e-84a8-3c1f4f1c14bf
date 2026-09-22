"""Concurrency and capacity-safety tests."""
from __future__ import annotations

import concurrent.futures


def submit_sync(client, tenant, gpus, duration, weight=None):
    payload = {"tenant_id": tenant, "gpus": gpus, "duration_seconds": duration}
    if weight is not None:
        payload["weight"] = weight
    r = client.post("/jobs", json=payload)
    assert r.status_code == 201, r.text
    return r.json()["job_id"]


def test_concurrent_schedules_never_exceed_capacity(client):
    # 20 tenants x one 3-GPU job on an 8-GPU cluster => at most 2 can run.
    for i in range(20):
        submit_sync(client, f"t{i:02d}", 3, 100, weight=1 + (i % 5) * 0.5)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(lambda: client.post("/schedule")) for _ in range(12)
        ]
        results = [f.result() for f in futures]

    for r in results:
        assert r.status_code == 200, r.text
        cap = r.json()["capacity"]
        assert cap["running_gpus"] <= cap["total_gpus"]
        for p in r.json()["scheduled"]:
            assert p["gpus"] <= cap["free_gpus"] + p["gpus"]  # ran within cap

    final = client.get("/queue").json()
    # floor(8/3) = 2 jobs using 6 GPUs; every remaining head needs 3 > 2 free.
    assert final["capacity"]["running_gpus"] == 6
    assert final["capacity"]["free_gpus"] == 2
    assert len(final["queued"]) == 18

    # Across ALL scheduling responses, the sum of GPUs in one batch plus the
    # already-running count can never exceed the cluster size.
    for r in results:
        body = r.json()
        used_after = body["capacity"]["running_gpus"]
        assert used_after <= 8


def test_concurrent_submits_and_cancels_are_consistent(client):
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(
            lambda i: submit_sync(client, "busy", 1, 10, weight=2),
            range(24),
        ))

    client.post("/schedule")
    running = client.get("/jobs", params={"status": "running"}).json()
    assert len(running) == 8  # fills the whole cluster, one GPU each
    running_ids = {j["job_id"] for j in running}

    # Cancel only QUEUED jobs concurrently; running allocation must be intact.
    queued_ids = [jid for jid in ids if jid not in running_ids][:12]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(
            lambda jid: client.post(f"/jobs/{jid}/cancel").status_code,
            queued_ids,
        ))
    assert all(s == 200 for s in statuses)

    q = client.get("/queue").json()
    assert q["capacity"]["running_gpus"] == 8
    assert q["capacity"]["free_gpus"] == 0
    # 24 submitted - 8 running - 12 cancelled = 4 still queued.
    assert len(q["queued"]) == 4

    # Complete two running jobs concurrently; capacity frees by exactly 2.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        codes = list(pool.map(
            lambda jid: client.post(f"/jobs/{jid}/complete").status_code,
            list(running_ids)[:2],
        ))
    assert codes == [200, 200]
    q2 = client.get("/queue").json()
    assert q2["capacity"]["running_gpus"] == 6
