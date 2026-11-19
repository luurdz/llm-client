import asyncio

import pytest
from fakeredis import FakeAsyncRedis

from app.jobs import QUEUE_KEY, JobStatus, create_job, enqueue_resume, get_job, job_key, next_message, update_job


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


async def waiting_job(redis) -> str:
    job_id = await create_job(redis, "Cusco")
    await next_message(redis, timeout=1)
    await update_job(redis, job_id, JobStatus.WAITING_APPROVAL, approval_request={"deposit": 1317.0})
    return job_id


def test_enqueue_resume_queues_the_decision(redis):
    async def run():
        job_id = await waiting_job(redis)
        accepted = await enqueue_resume(redis, job_id, approved=True)
        return job_id, accepted, await get_job(redis, job_id), await next_message(redis, timeout=1)

    job_id, accepted, job, message = asyncio.run(run())
    assert accepted is True
    assert job["status"] == JobStatus.PENDING
    assert job["approved"] == "True"
    assert message == {"job_id": job_id, "action": "resume", "approved": True}


def test_second_approval_is_rejected(redis):
    async def run():
        job_id = await waiting_job(redis)
        first = await enqueue_resume(redis, job_id, approved=True)
        await update_job(redis, job_id, JobStatus.WAITING_APPROVAL)  # second request read the status before the first wrote
        second = await enqueue_resume(redis, job_id, approved=False)
        return first, second, await redis.llen(QUEUE_KEY)

    first, second, queued = asyncio.run(run())
    assert (first, second) == (True, False)
    assert queued == 1


def test_approval_is_rejected_when_job_is_not_waiting(redis):
    async def run():
        job_id = await create_job(redis, "Cusco")
        return await enqueue_resume(redis, job_id, approved=True)

    assert asyncio.run(run()) is False


def test_approval_of_unknown_job_does_not_create_it(redis):
    async def run():
        accepted = await enqueue_resume(redis, "no-existe", approved=True)
        return accepted, await redis.exists(job_key("no-existe"))

    assert asyncio.run(run()) == (False, 0)


def test_every_write_keeps_the_ttl(redis):
    async def run():
        job_id = await create_job(redis, "Cusco")
        await update_job(redis, job_id, JobStatus.RUNNING)
        return await redis.ttl(job_key(job_id))

    assert asyncio.run(run()) > 0


def test_next_message_returns_none_when_queue_is_empty(redis):
    assert asyncio.run(next_message(redis, timeout=1)) is None
