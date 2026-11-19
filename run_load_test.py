import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

import httpx

REPORT_PATH = Path("traces/load_test.json")
FINAL_STATUSES = {"DONE", "FAILED"}
QUESTIONS = [
    "Quiero el paquete Bariloche Clásico para 2 personas. ¿Cuánto sale y cuánto es la seña?",
    "Quiero Cataratas del Iguazú para 3 personas. ¿Cuánto sale en total y cuánto dejo de seña?",
    "Quiero Salta y Jujuy para 2 personas con upgrade de hotel. ¿Cuánto sale y qué hacemos cada día?",
    "Quiero Río de Janeiro para 2 personas. ¿Cuánto es la seña para reservar?",
    "¿Qué incluye el paquete Punta Cana y qué seguro trae?",
]


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


async def run_job(client: httpx.AsyncClient, question: str, approve: bool, poll_seconds: float) -> dict[str, Any]:
    start = time.perf_counter()
    response = await client.post("/tasks", json={"question": question})
    response.raise_for_status()
    submit_ms = (time.perf_counter() - start) * 1000
    job_id = response.json()["job_id"]
    statuses = ["PENDING"]

    while statuses[-1] not in FINAL_STATUSES:
        await asyncio.sleep(poll_seconds)
        job = (await client.get(f"/tasks/{job_id}")).json()
        if job["status"] != statuses[-1]:
            statuses.append(job["status"])
            print(f"{time.perf_counter() - start:7.1f}s  {job_id[:8]}  {job['status']}")
        if job["status"] == "WAITING_APPROVAL" and approve:
            await client.post(f"/tasks/{job_id}/approve", json={"approved": True})

    result = job.get("result") or {}
    return {
        "job_id": job_id,
        "question": question,
        "submit_ms": round(submit_ms, 1),
        "total_s": round(time.perf_counter() - start, 1),
        "statuses": statuses,
        "final_status": job["status"],
        "booking_status": result.get("booking_status"),
        "cost_estimate": result.get("cost_estimate"),
        "error": job.get("error"),
    }


async def run_load_test(
    client: httpx.AsyncClient, questions: list[str], approve: bool = True, poll_seconds: float = 3
) -> dict[str, Any]:
    jobs = await asyncio.gather(*(run_job(client, q, approve, poll_seconds) for q in questions))
    submit = [j["submit_ms"] for j in jobs]
    total = [j["total_s"] for j in jobs]
    return {
        "requests": len(jobs),
        "done": sum(j["final_status"] == "DONE" for j in jobs),
        "failed": sum(j["final_status"] == "FAILED" for j in jobs),
        "submit_ms_p50": percentile(submit, 50),
        "submit_ms_p95": percentile(submit, 95),
        "total_s_p50": percentile(total, 50),
        "total_s_p95": percentile(total, 95),
        "jobs": jobs,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Sends 5 concurrent tasks to the API and follows them until they end.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--no-approve", action="store_true", help="leave the jobs in WAITING_APPROVAL")
    args = parser.parse_args()

    async with httpx.AsyncClient(base_url=args.base_url, timeout=30) as client:
        report = await run_load_test(client, QUESTIONS, approve=not args.no_approve)

    print(f"\n{report['done']} DONE, {report['failed']} FAILED de {report['requests']} pedidos")
    print(f"POST /tasks: p50 {report['submit_ms_p50']} ms, p95 {report['submit_ms_p95']} ms")
    print(f"Job completo: p50 {report['total_s_p50']} s, p95 {report['total_s_p95']} s")
    for job in report["jobs"]:
        print(f"  {job['job_id'][:8]}  {' -> '.join(job['statuses'])}  {job['total_s']} s  {job['booking_status']}")
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Reporte: {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
