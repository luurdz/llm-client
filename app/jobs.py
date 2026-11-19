import json
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from redis.asyncio import Redis

QUEUE_KEY = "travel:queue"
JOB_KEY_PREFIX = "travel:job:"
JOB_TTL_SECONDS = 7 * 24 * 3600
JSON_FIELDS = ("result", "approval_request")
QUEUE_WAIT_SECONDS = 5
SOCKET_TIMEOUT_SECONDS = 30  # must be longer than QUEUE_WAIT_SECONDS: redis-py closes reads after 5 s by default


class JobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    DONE = "DONE"
    FAILED = "FAILED"


def connect_redis(redis_url: str) -> Redis:
    return Redis.from_url(redis_url, decode_responses=True, socket_timeout=SOCKET_TIMEOUT_SECONDS)


def job_key(job_id: str) -> str:
    return f"{JOB_KEY_PREFIX}{job_id}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def create_job(redis: Redis, question: str) -> str:
    job_id = str(uuid.uuid4())
    key = job_key(job_id)
    now = _now()
    async with redis.pipeline(transaction=True) as pipe:
        pipe.hset(
            key,
            mapping={"job_id": job_id, "status": JobStatus.PENDING, "question": question, "created_at": now, "updated_at": now},
        )
        pipe.expire(key, JOB_TTL_SECONDS)
        pipe.lpush(QUEUE_KEY, json.dumps({"job_id": job_id, "action": "run"}))
        await pipe.execute()
    return job_id


async def get_job(redis: Redis, job_id: str) -> dict[str, Any] | None:
    data = await redis.hgetall(job_key(job_id))
    if not data:
        return None
    for field in JSON_FIELDS:
        if field in data:
            data[field] = json.loads(data[field])
    return data


async def update_job(redis: Redis, job_id: str, status: JobStatus, **fields: Any) -> None:
    mapping = {"status": status, "updated_at": _now()}
    for name, value in fields.items():
        mapping[name] = json.dumps(value, ensure_ascii=False) if name in JSON_FIELDS else str(value)
    key = job_key(job_id)
    async with redis.pipeline(transaction=True) as pipe:
        pipe.hset(key, mapping=mapping)
        pipe.expire(key, JOB_TTL_SECONDS)
        await pipe.execute()


async def enqueue_resume(redis: Redis, job_id: str, approved: bool) -> bool:
    key = job_key(job_id)
    if await redis.hget(key, "status") != JobStatus.WAITING_APPROVAL:
        return False
    if not await redis.hsetnx(key, "approved", str(approved)):
        return False
    async with redis.pipeline(transaction=True) as pipe:
        pipe.hset(key, mapping={"status": JobStatus.PENDING, "updated_at": _now()})
        pipe.lpush(QUEUE_KEY, json.dumps({"job_id": job_id, "action": "resume", "approved": approved}))
        await pipe.execute()
    return True


async def next_message(redis: Redis, timeout: int = QUEUE_WAIT_SECONDS) -> dict[str, Any] | None:
    item = await redis.brpop([QUEUE_KEY], timeout=timeout)
    if item is None:
        return None
    _, payload = item
    return json.loads(payload)
