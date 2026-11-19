from typing import Any, Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END
from langgraph.types import interrupt

from multiagent.state import TravelState

BookingStatus = Literal["confirmed", "rejected"]


class JobState(TravelState):
    booking_status: BookingStatus | None


def needs_approval(state: JobState) -> Literal["approval", "__end__"]:
    return "approval" if state.get("cost_estimate") is not None else END


def build_approval_request(state: JobState) -> dict[str, Any]:
    cost = state["cost_estimate"]
    return {
        "action": "confirm_deposit",
        "question": state["messages"][0].text,
        "people": cost.people,
        "total": cost.total,
        "deposit": cost.deposit,
    }


async def approval_node(state: JobState) -> dict:
    request = build_approval_request(state)
    decision = interrupt(request)
    if decision.get("approved"):
        status, text = "confirmed", f"Reserva confirmada: se cobra la seña de {request['deposit']} dólares."
    else:
        status, text = "rejected", "Reserva no confirmada: el operador rechazó el cobro de la seña."
    return {"booking_status": status, "messages": [AIMessage(content=text, name="approval")]}
