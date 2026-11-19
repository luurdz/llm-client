import asyncio

import pytest
from fakeredis import FakeAsyncRedis
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import build_job_graph
from app.jobs import JobStatus, create_job, enqueue_resume, get_job, job_key, next_message
from app.worker import handle_message, worker_loop
from tests.test_multiagent_graph import ITINERARY, NOTES, QUESTION, FakeTravelModel, cost_call


@pytest.fixture
def redis() -> FakeAsyncRedis:
    return FakeAsyncRedis(decode_responses=True)


def travel_graph(with_cost: bool = True):
    if with_cost:
        model = FakeTravelModel(
            messages=iter([AIMessage(NOTES), cost_call(), AIMessage("Total 4390, seña 1317"), AIMessage(ITINERARY)]),
            decisions=["researcher", "analyst", "planner", "FINISH"],
        )
    else:
        model = FakeTravelModel(messages=iter([AIMessage(NOTES)]), decisions=["researcher", "FINISH"])
    return build_job_graph(model, rag=None, checkpointer=InMemorySaver())


class BrokenGraph:
    async def ainvoke(self, graph_input, config):
        raise RuntimeError("429 quota exceeded")


async def process_next(graph, redis) -> None:
    await handle_message(graph, redis, await next_message(redis, timeout=1))


def test_job_waits_for_approval_then_finishes_confirmed(redis):
    graph = travel_graph()

    async def run():
        job_id = await create_job(redis, QUESTION)
        await process_next(graph, redis)
        waiting = await get_job(redis, job_id)
        await enqueue_resume(redis, job_id, approved=True)
        await process_next(graph, redis)
        return waiting, await get_job(redis, job_id)

    waiting, done = asyncio.run(run())
    assert waiting["status"] == JobStatus.WAITING_APPROVAL
    assert waiting["approval_request"]["deposit"] == 1317.0
    assert done["status"] == JobStatus.DONE
    assert done["result"]["booking_status"] == "confirmed"
    assert done["result"]["cost_estimate"] == {"people": 2, "total": 4390.0, "deposit": 1317.0}
    assert done["result"]["answer"] == "Reserva confirmada: se cobra la seña de 1317.0 dólares."


def test_job_without_cost_is_done_without_approval(redis):
    async def run():
        job_id = await create_job(redis, "¿Qué incluye Cusco?")
        await process_next(travel_graph(with_cost=False), redis)
        return await get_job(redis, job_id)

    job = asyncio.run(run())
    assert job["status"] == JobStatus.DONE
    assert job["result"]["research_notes"] == NOTES
    assert job["result"]["booking_status"] is None


def test_exception_marks_the_job_failed(redis):
    async def run():
        job_id = await create_job(redis, QUESTION)
        await process_next(BrokenGraph(), redis)
        return await get_job(redis, job_id)

    job = asyncio.run(run())
    assert job["status"] == JobStatus.FAILED
    assert job["error"] == "RuntimeError: 429 quota exceeded"


def test_error_after_the_graph_also_marks_the_job_failed(redis):
    class BadStateGraph:
        async def ainvoke(self, graph_input, config):
            return {}  # no "messages": summarize_result raises KeyError after the graph ran

    async def run():
        job_id = await create_job(redis, QUESTION)
        await process_next(BadStateGraph(), redis)
        return await get_job(redis, job_id)

    job = asyncio.run(run())
    assert job["status"] == JobStatus.FAILED


def test_unknown_job_is_ignored(redis):
    async def run():
        await handle_message(BrokenGraph(), redis, {"job_id": "no-existe", "action": "run"})
        return await redis.exists(job_key("no-existe"))

    assert asyncio.run(run()) == 0


def test_worker_loop_runs_jobs_concurrently(redis):
    class SlowGraph:
        running = 0
        peak = 0

        async def ainvoke(self, graph_input, config):
            SlowGraph.running += 1
            SlowGraph.peak = max(SlowGraph.peak, SlowGraph.running)
            await asyncio.sleep(0.2)
            SlowGraph.running -= 1
            return {"messages": [AIMessage("ok")]}

    async def run():
        ids = [await create_job(redis, f"consulta {i}") for i in range(5)]
        loop = asyncio.create_task(worker_loop(SlowGraph(), redis, max_jobs=5))
        await asyncio.sleep(0.6)
        loop.cancel()
        return [await get_job(redis, i) for i in ids]

    jobs = asyncio.run(run())
    assert [j["status"] for j in jobs] == [JobStatus.DONE] * 5
    assert SlowGraph.peak == 5
