import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from app.jobs import JobStatus, connect_redis, create_job, enqueue_resume, get_job


class TaskRequest(BaseModel):
    question: str = Field(min_length=1, description="Consulta del cliente para el asistente de viajes")


class ApprovalRequest(BaseModel):
    approved: bool = Field(description="true cobra la seña y confirma la reserva, false la rechaza")


class TaskAccepted(BaseModel):
    job_id: str
    status: JobStatus


class TaskState(BaseModel):
    job_id: str
    status: JobStatus
    question: str
    created_at: str
    updated_at: str
    approval_request: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    load_dotenv()
    app.state.redis = connect_redis(os.environ["REDIS_URL"])
    yield
    await app.state.redis.aclose()


app = FastAPI(title="Asistente de viajes multi-agente", lifespan=lifespan)


def get_redis(request: Request) -> Redis:
    return request.app.state.redis


@app.post("/tasks", status_code=status.HTTP_202_ACCEPTED, response_model=TaskAccepted)
async def submit_task(task: TaskRequest, redis: Redis = Depends(get_redis)) -> TaskAccepted:
    job_id = await create_job(redis, task.question)
    return TaskAccepted(job_id=job_id, status=JobStatus.PENDING)


@app.get("/tasks/{job_id}", response_model=TaskState)
async def get_task(job_id: str, redis: Redis = Depends(get_redis)) -> TaskState:
    job = await get_job(redis, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Job {job_id} not found")
    return TaskState(**job)


@app.post("/tasks/{job_id}/approve", status_code=status.HTTP_202_ACCEPTED, response_model=TaskAccepted)
async def approve_task(job_id: str, decision: ApprovalRequest, redis: Redis = Depends(get_redis)) -> TaskAccepted:
    job = await get_job(redis, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Job {job_id} not found")
    if not await enqueue_resume(redis, job_id, decision.approved):
        raise HTTPException(status.HTTP_409_CONFLICT, f"Job {job_id} is {job['status']}, not waiting for approval")
    return TaskAccepted(job_id=job_id, status=JobStatus.PENDING)
