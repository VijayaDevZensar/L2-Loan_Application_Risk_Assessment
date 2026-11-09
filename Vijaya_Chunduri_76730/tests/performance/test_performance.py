"""
PERFORMANCE TESTS — Day 2, Exercise 5
FinTrust Bank AI QE Capstone

Captures latency (avg, p95) and throughput at 1, 5, 10, 20 concurrent users.
Uses threading to simulate concurrent load against the /score endpoint.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import pytest

SETTINGS = json.loads((ROOT / "config" / "settings.json").read_text())
THRESHOLDS = SETTINGS["thresholds"]

SAMPLE_APP = {
    "application_id": "APP20001",
    "age": 38,
    "annual_income": 72000.0,
    "employment_years": 10.0,
    "existing_debt": 12000.0,
    "credit_score": 710,
    "loan_amount": 35000.0,
    "loan_term_months": 48,
    "previous_defaults": 0,
    "region": "Central",
    "employment_type": "Salaried",
    "repayment_status": "Good",
}


def _make_score_request(client, results: list, idx: int):
    app = {**SAMPLE_APP, "application_id": f"APP{20001 + idx:05d}"}
    start = time.perf_counter()
    try:
        r = client.post("/score", json=app)
        elapsed = time.perf_counter() - start
        results.append({"status": r.status_code, "latency": elapsed, "error": None})
    except Exception as e:
        elapsed = time.perf_counter() - start
        results.append({"status": 0, "latency": elapsed, "error": str(e)})


def _run_load(client, n_users: int, n_requests_per_user: int = 1) -> dict:
    results = []
    threads = []
    global_start = time.perf_counter()

    for i in range(n_users * n_requests_per_user):
        t = threading.Thread(target=_make_score_request, args=(client, results, i))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed_total = time.perf_counter() - global_start
    latencies = sorted([r["latency"] for r in results])
    errors = [r for r in results if r["status"] != 200]

    n = len(latencies)
    avg = sum(latencies) / n if n > 0 else 0
    p95_idx = max(0, int(n * 0.95) - 1)
    p95 = latencies[p95_idx] if latencies else 0
    throughput = n / elapsed_total if elapsed_total > 0 else 0
    error_rate = len(errors) / n if n > 0 else 0

    return {
        "n_users": n_users,
        "n_requests": n,
        "avg_latency_s": round(avg, 4),
        "p95_latency_s": round(p95, 4),
        "throughput_rps": round(throughput, 2),
        "error_count": len(errors),
        "error_rate": round(error_rate, 4),
    }


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.api.app import app
    return TestClient(app)


@pytest.mark.parametrize("n_users", [1, 5, 10, 20])
def test_perf_score_endpoint(client, n_users):
    """PERF: /score latency and error rate at 1, 5, 10, 20 concurrent requests."""
    stats = _run_load(client, n_users)

    p95 = stats["p95_latency_s"]
    error_rate = stats["error_rate"]
    threshold_p95 = THRESHOLDS["score_p95_seconds"]
    threshold_err = THRESHOLDS["error_rate_max"]

    print(f"\nPerformance [{n_users} users]: avg={stats['avg_latency_s']:.3f}s  "
          f"p95={p95:.3f}s  rps={stats['throughput_rps']}  errors={stats['error_count']}")

    # Save results
    report_path = ROOT / "reports" / "day2" / f"perf_score_{n_users}users.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(stats, indent=2))

    assert error_rate <= threshold_err, (
        f"PERF FAIL [{n_users} users]: error_rate {error_rate:.1%} exceeds {threshold_err:.1%}"
    )
    assert p95 <= threshold_p95, (
        f"PERF FAIL [{n_users} users]: p95 {p95:.3f}s exceeds threshold {threshold_p95}s — "
        "Review scalability. This would be a RELEASE BLOCK in production."
    )


def test_perf_write_summary(client):
    """Write consolidated performance summary to reports/day2/."""
    summary = []
    for n_users in [1, 5, 10, 20]:
        stats = _run_load(client, n_users)
        summary.append(stats)
        print(f"  [{n_users:2d} users] p95={stats['p95_latency_s']:.3f}s  rps={stats['throughput_rps']:.1f}")

    report_path = ROOT / "reports" / "day2" / "performance_summary.json"
    report_path.write_text(json.dumps(summary, indent=2))
    assert report_path.exists()
