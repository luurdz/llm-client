import asyncio

import pytest
from fakeredis import FakeAsyncRedis

from app.jobs import JobStatus, create_job, enqueue_resume, get_job, next_message, update_job


@pytest.fixture
def redis() -> FakeAsyncRedis:
    return FakeAsyncRedis(decode_responses=True)


def test_create_job_writes_pending_and_enqueues(redis):
    async def run():
        job_id = await create_job(redis, "Cusco para 2 personas")
        job = await get_job(redis, job_id)
        message = await next_message(redis, timeout=1)
        return job_id, job, message

    job_id, job, message = asyncio.run(run())
    assert job["status"] == JobStatus.PENDING
    assert job["question"] == "Cusco para 2 personas"
    assert message == {"job_id": job_id, "action": "run"}


def test_get_job_returns_none_for_unknown_id(redis):
    assert asyncio.run(get_job(redis, "no-existe")) is None


def test_update_job_stores_json_fields(redis):
    async def run():
        job_id = await create_job(redis, "Cusco")
        await update_job(redis, job_id, JobStatus.WAITING_APPROVAL, approval_request={"deposit": 1317.0})
        await update_job(redis, job_id, JobStatus.DONE, result={"answer": "ok"})
        return await get_job(redis, job_id)

    job = asyncio.run(run())
    assert job["status"] == JobStatus.DONE
    assert job["approval_request"] == {"deposit": 1317.0}
    assert job["result"] == {"answer": "ok"}


def test_enqueue_resume_queues_the_decision(redis):
    async def run():
        job_id = await create_job(redis, "Cusco")
        await next_message(redis, timeout=1)
        await enqueue_resume(redis, job_id, approved=True)
        return job_id, await get_job(redis, job_id), await next_message(redis, timeout=1)

    job_id, job, message = asyncio.run(run())
    assert job["status"] == JobStatus.PENDING
    assert message == {"job_id": job_id, "action": "resume", "approved": True}


def test_next_message_returns_none_when_queue_is_empty(redis):
    assert asyncio.run(next_message(redis, timeout=1)) is None
