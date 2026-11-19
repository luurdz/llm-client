import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from redis.asyncio import Redis

from agent.graph import build_llm
from app.graph import build_job_graph, open_checkpointer
from app.jobs import JobStatus, connect_redis, get_job, next_message, update_job
from app.observability import build_run_config, log_tracing_status
from cloud_rag.connection import connect_pinecone
from cloud_rag.ingest import load_chunk_metas
from cloud_rag.rag_system import RAGSystem

DATA_DIR = Path("data")
REQUESTS_PER_MINUTE = 10  # Gemini free tier allows 15
MAX_CONCURRENT_JOBS = 5

logger = logging.getLogger(__name__)


def summarize_result(state: dict[str, Any]) -> dict[str, Any]:
    answers = [m.text for m in state["messages"] if isinstance(m, AIMessage) and m.text]
    cost = state.get("cost_estimate")
    return {
        "answer": answers[-1] if answers else None,
        "research_notes": state.get("research_notes"),
        "cost_estimate": cost.model_dump() if cost else None,
        "itinerary": state.get("itinerary"),
        "booking_status": state.get("booking_status"),
        "supervisor_steps": state.get("step_count"),
    }


async def handle_message(graph: CompiledStateGraph, redis: Redis, message: dict[str, Any]) -> None:
    job_id = message["job_id"]
    job = await get_job(redis, job_id)
    if job is None:
        logger.warning("Job %s not found, message discarded", job_id)
        return

    try:
        await update_job(redis, job_id, JobStatus.RUNNING)
        if message["action"] == "resume":
            graph_input = Command(resume={"approved": message["approved"]})
        else:
            graph_input = {"messages": [HumanMessage(job["question"])]}

        state = await graph.ainvoke(graph_input, build_run_config(job_id))

        if "__interrupt__" in state:
            await update_job(redis, job_id, JobStatus.WAITING_APPROVAL, approval_request=state["__interrupt__"][0].value)
            logger.info("Job %s waiting for approval", job_id)
        else:
            await update_job(redis, job_id, JobStatus.DONE, result=summarize_result(state))
            logger.info("Job %s done", job_id)
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        await update_job(redis, job_id, JobStatus.FAILED, error=f"{type(e).__name__}: {e}")


async def worker_loop(graph: CompiledStateGraph, redis: Redis, max_jobs: int = MAX_CONCURRENT_JOBS) -> None:
    slots = asyncio.Semaphore(max_jobs)
    running: set[asyncio.Task] = set()

    async def process(message: dict[str, Any]) -> None:
        try:
            await handle_message(graph, redis, message)
        finally:
            slots.release()

    while True:
        await slots.acquire()
        message = await next_message(redis)
        if message is None:
            slots.release()
            continue
        task = asyncio.create_task(process(message))
        running.add(task)
        task.add_done_callback(running.discard)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("google_genai", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    load_dotenv()
    log_tracing_status()
    redis_url = os.environ["REDIS_URL"]

    index, embeddings, namespace = connect_pinecone()
    llm = build_llm()
    llm.rate_limiter = InMemoryRateLimiter(requests_per_second=REQUESTS_PER_MINUTE / 60, max_bucket_size=1)
    rag = RAGSystem(index=index, embeddings=embeddings, namespace=namespace, metas=load_chunk_metas(DATA_DIR))

    redis = connect_redis(redis_url)
    try:
        async with open_checkpointer(redis_url) as checkpointer:
            graph = build_job_graph(llm, rag, checkpointer)
            logger.info("Worker ready, up to %d jobs at a time", MAX_CONCURRENT_JOBS)
            await worker_loop(graph, redis)
    finally:
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
