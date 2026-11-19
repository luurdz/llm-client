import asyncio
from types import SimpleNamespace

import httpx
from fakeredis import FakeAsyncRedis
from langchain_core.messages import AIMessage
from langgraph.types import Command

from app.main import app, get_redis
from app.worker import worker_loop
from run_load_test import QUESTIONS, percentile, run_load_test


class FakeJobGraph:
    """Pauses when the question asks for a price and confirms on resume, like the real job graph."""

    async def ainvoke(self, graph_input, config):
        await asyncio.sleep(0.05)
        if isinstance(graph_input, Command):
            return {"messages": [AIMessage("Reserva confirmada")], "booking_status": "confirmed"}
        if "Cuánto" in graph_input["messages"][0].text:
            return {"__interrupt__": [SimpleNamespace(value={"deposit": 1317.0})]}
        return {"messages": [AIMessage("Incluye todo")]}


def test_percentile_uses_nearest_rank():
    values = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert (percentile(values, 50), percentile(values, 95)) == (3.0, 5.0)


def test_five_concurrent_tasks_end_done_through_api_and_worker():
    redis = FakeAsyncRedis(decode_responses=True)
    app.dependency_overrides[get_redis] = lambda: redis

    async def run():
        worker = asyncio.create_task(worker_loop(FakeJobGraph(), redis, max_jobs=5))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            report = await asyncio.wait_for(run_load_test(client, QUESTIONS, poll_seconds=0.05), timeout=30)
        worker.cancel()
        return report

    try:
        report = asyncio.run(run())
    finally:
        app.dependency_overrides.clear()

    assert (report["requests"], report["done"], report["failed"]) == (5, 5, 0)
    statuses = [job["statuses"][-1] for job in report["jobs"]]
    assert statuses == ["DONE"] * 5
    approved = [job for job in report["jobs"] if "WAITING_APPROVAL" in job["statuses"]]
    assert len(approved) == 4
    assert all(job["booking_status"] == "confirmed" for job in approved)
