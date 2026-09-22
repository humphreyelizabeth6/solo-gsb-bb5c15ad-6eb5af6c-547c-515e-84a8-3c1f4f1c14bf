"""End-to-end API tests using FastAPI's TestClient (8-GPU cluster)."""
from __future__ import annotations

from conftest import set_submitted_at


def submit(client, tenant, gpus, duration, weight=None):
    payload = {"tenant_id": tenant, "gpus": gpus, "duration_seconds": duration}
    if weight is not None:
        payload["weight"] = weight
    r = client.post("/jobs", json=payload)
    assert r.status_code == 201, r.text
    return r.json()["job_id"]


def order(resp):
    return resp.json()["selection_order"]


# --------------------------------------------------------------------------- #
def test_health_and_empty_queue(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["total_gpus"] == 8
    assert body["free_gpus"] == 8

    r = client.post("/schedule")
    assert r.status_code == 200
    assert r.json()["selection_order"] == []


def test_submit_requires_weight_for_unknown_tenant(client):
    r = client.post("/jobs", json={
        "tenant_id": "newbie", "gpus": 1, "duration_seconds": 10,
    })
    assert r.status_code == 400

    r = client.post("/jobs", json={
        "tenant_id": "newbie", "gpus": 0, "duration_seconds": 10, "weight": 1,
    })
    assert r.status_code == 422

    r = client.post("/jobs", json={
        "tenant_id": "newbie", "gpus": 9, "duration_seconds": 10, "weight": 1,
    })
    assert r.status_code == 422
    assert "8" in r.json()["detail"]

    r = client.post("/jobs", json={
        "tenant_id": "bad/id", "gpus": 1, "duration_seconds": 10, "weight": 1,
    })
    assert r.status_code == 422


def test_weighted_dispatch_order(client):
    a1 = submit(client, "a", 1, 100, weight=1)
    a2 = submit(client, "a", 1, 100)
    b1 = submit(client, "b", 1, 100, weight=3)
    b2 = submit(client, "b", 1, 100)

    r = client.post("/schedule")
    assert order(r) == [b1, b2, a1, a2]
    body = r.json()
    assert body["capacity"]["running_gpus"] == 4
    assert body["capacity"]["free_gpus"] == 4
    bal = {b["tenant_id"]: b for b in body["balances"]}
    assert bal["a"]["deficit"] == 0
    assert bal["b"]["deficit"] == 0

    q = client.get("/queue").json()
    assert q["queued"] == []  # all dispatched


def test_capacity_blocked_then_released_by_completion(client):
    import time

    a1 = submit(client, "a", 6, 60, weight=1)
    b1 = submit(client, "b", 6, 60, weight=1)
    # Break the lexicographic tie deterministically via aging: a waited longer.
    set_submitted_at(a1, time.time() - 60)

    r = client.post("/schedule")
    assert order(r) == [a1]
    assert r.json()["blocked"] == [b1]
    assert r.json()["capacity"]["free_gpus"] == 2

    # b1 cannot run while a1 holds 6 GPUs.
    r2 = client.post("/schedule")
    assert order(r2) == []
    assert r2.json()["blocked"] == [b1]
    # deficit accrues while waiting for capacity.
    bal_b = next(b for b in r2.json()["balances"] if b["tenant_id"] == "b")
    assert bal_b["deficit"] > 0

    # completing a1 frees exactly 6 GPUs; b1 now fits and runs.
    assert client.post(f"/jobs/{a1}/complete").status_code == 200
    r3 = client.post("/schedule")
    assert order(r3) == [b1]
    assert r3.json()["capacity"]["free_gpus"] == 2

    assert client.post(f"/jobs/{b1}/complete").status_code == 200


def test_cancel_queued_job_frees_and_is_idempotent(client):
    a1 = submit(client, "a", 2, 100, weight=1)
    b1 = submit(client, "b", 2, 100, weight=1)
    assert client.post(f"/jobs/{a1}/cancel").status_code == 200
    # idempotent
    assert client.post(f"/jobs/{a1}/cancel").status_code == 200
    r = client.post("/schedule")
    assert order(r) == [b1]
    assert client.get(f"/jobs/{a1}").json()["status"] == "cancelled"

    # completing a queued (already cancelled) job is a 409
    assert client.post(f"/jobs/{a1}/complete").status_code == 409
    # unknown job -> 404
    assert client.post("/jobs/nope/complete").status_code == 404
    assert client.post("/jobs/nope/cancel").status_code == 404


def test_balance_carries_between_batches_and_resets_on_empty(client):
    # a1 (1 GPU, cost 100) runs; a2 needs all 8 GPUs and is blocked after a1.
    a1 = submit(client, "a", 1, 100, weight=1)
    a2 = submit(client, "a", 8, 100)

    r1 = client.post("/schedule")
    assert order(r1) == [a1]
    assert r1.json()["blocked"] == [a2]
    bal_a = next(b for b in r1.json()["balances"] if b["tenant_id"] == "a")
    # Deficit kept accumulating across the internal passes even though a2
    # could not fit; the positive balance is persisted for the next batch.
    assert bal_a["deficit"] > 0
    carried = bal_a["deficit"]

    # Finish a1: all 8 GPUs free. a2 immediately benefits from the carried
    # balance: 140 carried means it needs 11 more passes instead of 14.
    client.post(f"/jobs/{a1}/complete")
    r2 = client.post("/schedule")
    assert order(r2) == [a2]
    scheduled = r2.json()["scheduled"][0]
    assert scheduled["passes_before_run"] == 11
    # queue emptied => leftover balance resets to zero (standard DRR).
    bal_a2 = next(b for b in r2.json()["balances"] if b["tenant_id"] == "a")
    assert bal_a2["deficit"] == 0
    assert carried > 0


def test_aging_lets_old_job_overtake(client):
    # a's big job submitted long ago, b's small job now; equal weights.
    import time

    a1 = submit(client, "a", 1, 300, weight=1)
    b1 = submit(client, "b", 1, 100, weight=1)
    set_submitted_at(a1, time.time() - 600)

    r = client.post("/schedule")
    assert order(r) == [a1, b1]


def test_tenant_weight_update_and_listing(client):
    r = client.post("/tenants", json={"tenant_id": "z", "weight": 2})
    assert r.status_code == 201
    r = client.post("/tenants", json={"tenant_id": "z", "weight": 5})
    assert r.status_code == 201
    assert r.json()["weight"] == 5

    submit(client, "z", 1, 10)
    client.post("/jobs", json={"tenant_id": "z", "gpus": 1,
                               "duration_seconds": 10, "weight": 7})
    assert client.get("/jobs", params={"tenant_id": "z"}).status_code == 200
    assert client.get("/jobs", params={"status": "queued"}).status_code == 200
    assert client.get("/jobs", params={"status": "bogus"}).status_code == 422
    assert client.get("/jobs", params={"tenant_id": "ghost"}).status_code == 404
    assert client.get("/balances").status_code == 200
    assert client.get("/schedule/history").status_code == 200


def test_schedule_history_records_batches(client):
    a1 = submit(client, "a", 1, 10, weight=1)
    client.post("/schedule")
    hist = client.get("/schedule/history").json()
    assert len(hist) == 1
    assert hist[0]["selected"][0]["job_id"] == a1
    # empty scheduling rounds are not recorded as batches
    client.post("/schedule")
    assert len(client.get("/schedule/history").json()) == 1
