import asyncio

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer
from langgraph.types import Command

from app.graph import build_job_graph, build_serializer
from multiagent.state import CostEstimate
from tests.test_multiagent_graph import ITINERARY, NOTES, QUESTION, FakeTravelModel, cost_call

CONFIG = {"configurable": {"thread_id": "job-test"}}


def full_model() -> FakeTravelModel:
    return FakeTravelModel(
        messages=iter([AIMessage(NOTES), cost_call(), AIMessage("Total 4390, seña 1317"), AIMessage(ITINERARY)]),
        decisions=["researcher", "analyst", "planner", "FINISH"],
    )


def run_until_approval(model: FakeTravelModel):
    app = build_job_graph(model, rag=None, checkpointer=InMemorySaver())
    result = asyncio.run(app.ainvoke({"messages": [HumanMessage(QUESTION)]}, CONFIG))
    return app, result


def test_graph_pauses_before_confirming_the_deposit():
    app, result = run_until_approval(full_model())

    [pause] = result["__interrupt__"]
    assert pause.value == {
        "action": "confirm_deposit",
        "question": QUESTION,
        "people": 2,
        "total": 4390.0,
        "deposit": 1317.0,
    }
    assert app.get_state(CONFIG).next == ("approval",)


def test_resume_with_approval_confirms_the_booking():
    app, _ = run_until_approval(full_model())
    result = asyncio.run(app.ainvoke(Command(resume={"approved": True}), CONFIG))

    assert result["booking_status"] == "confirmed"
    assert result["messages"][-1].text == "Reserva confirmada: se cobra la seña de 1317.0 dólares."


def test_resume_with_rejection_does_not_confirm():
    app, _ = run_until_approval(full_model())
    result = asyncio.run(app.ainvoke(Command(resume={"approved": False}), CONFIG))

    assert result["booking_status"] == "rejected"


def test_graph_ends_without_pause_when_there_is_no_cost():
    model = FakeTravelModel(messages=iter([AIMessage(NOTES)]), decisions=["researcher", "FINISH"])
    _, result = run_until_approval(model)

    assert "__interrupt__" not in result
    assert result.get("booking_status") is None


def test_redis_serializer_restores_cost_estimate():
    cost = CostEstimate(people=2, total=4390.0, deposit=1317.0)
    serde = build_serializer()

    restored = serde.loads_typed(serde.dumps_typed({"cost_estimate": cost}))

    assert restored["cost_estimate"] == cost


def test_redis_serializer_without_allowlist_returns_a_dict():
    cost = CostEstimate(people=2, total=4390.0, deposit=1317.0)
    serde = JsonPlusRedisSerializer()

    restored = serde.loads_typed(serde.dumps_typed({"cost_estimate": cost}))

    assert not isinstance(restored["cost_estimate"], CostEstimate)
