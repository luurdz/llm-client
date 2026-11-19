import asyncio

import pytest
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient

from app.jobs import QUEUE_KEY, JobStatus, next_message, update_job
from app.main import app, get_redis


@pytest.fixture
def redis() -> FakeAsyncRedis:
    return FakeAsyncRedis(decode_responses=True)


@pytest.fixture
def client(redis):
    app.dependency_overrides[get_redis] = lambda: redis
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_post_task_returns_job_id_and_enqueues(client, redis):
    response = client.post("/tasks", json={"question": "Cusco para 2 personas"})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "PENDING"
    assert asyncio.run(next_message(redis, timeout=1)) == {"job_id": body["job_id"], "action": "run"}


def test_post_task_rejects_empty_question(client):
    assert client.post("/tasks", json={"question": ""}).status_code == 422


def test_get_task_returns_its_state(client):
    job_id = client.post("/tasks", json={"question": "Cusco"}).json()["job_id"]

    body = client.get(f"/tasks/{job_id}").json()

    assert body["job_id"] == job_id
    assert body["status"] == "PENDING"
    assert body["question"] == "Cusco"
    assert body["result"] is None


def test_get_unknown_task_is_404(client):
    assert client.get("/tasks/no-existe").status_code == 404


def test_approve_waiting_task_enqueues_the_decision(client, redis):
    job_id = client.post("/tasks", json={"question": "Cusco"}).json()["job_id"]
    asyncio.run(next_message(redis, timeout=1))
    asyncio.run(update_job(redis, job_id, JobStatus.WAITING_APPROVAL, approval_request={"deposit": 1317.0}))

    assert client.get(f"/tasks/{job_id}").json()["approval_request"] == {"deposit": 1317.0}
    response = client.post(f"/tasks/{job_id}/approve", json={"approved": True})

    assert response.status_code == 202
    assert asyncio.run(next_message(redis, timeout=1)) == {"job_id": job_id, "action": "resume", "approved": True}


def test_approve_twice_is_409(client, redis):
    job_id = client.post("/tasks", json={"question": "Cusco"}).json()["job_id"]
    asyncio.run(update_job(redis, job_id, JobStatus.WAITING_APPROVAL))

    assert client.post(f"/tasks/{job_id}/approve", json={"approved": True}).status_code == 202
    assert client.post(f"/tasks/{job_id}/approve", json={"approved": False}).status_code == 409


def test_approve_task_not_waiting_is_409(client, redis):
    job_id = client.post("/tasks", json={"question": "Cusco"}).json()["job_id"]

    response = client.post(f"/tasks/{job_id}/approve", json={"approved": True})

    assert response.status_code == 409
    assert response.json()["detail"] == f"Job {job_id} is PENDING, not waiting for approval"
    assert asyncio.run(redis.llen(QUEUE_KEY)) == 1


def test_approve_unknown_task_is_404(client):
    assert client.post("/tasks/no-existe/approve", json={"approved": True}).status_code == 404
