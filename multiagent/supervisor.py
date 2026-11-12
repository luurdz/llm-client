import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from multiagent.prompts import SUPERVISOR_PROMPT
from multiagent.state import NextAgent, SupervisorDecision, TravelState

logger = logging.getLogger(__name__)

MAX_STEPS = 8
PENDING = "pendiente"


def describe_state(state: TravelState) -> str:
    """Arma el texto que lee el supervisor: la consulta y lo que ya produjo cada agente."""
    cost = state.get("cost_estimate")
    cost_text = f"total {cost.total} dólares, seña {cost.deposit} dólares, {cost.people} pasajeros" if cost else PENDING
    analyst_answer = next(
        (m.text for m in reversed(state["messages"]) if isinstance(m, AIMessage) and m.name == "analyst"), PENDING
    )
    return (
        f"Consulta del cliente:\n{state['messages'][0].text}\n\n"
        f"Notas del investigador:\n{state.get('research_notes') or PENDING}\n\n"
        f"Cálculo de costos (tool):\n{cost_text}\n\n"
        f"Respuesta del analista:\n{analyst_answer}\n\n"
        f"Itinerario:\n{state.get('itinerary') or PENDING}"
    )


def fallback_route(state: TravelState) -> NextAgent:
    """Ruta fija para cuando el modelo no devuelve una decisión válida: el primer dato que falta, o FINISH."""
    if not state.get("research_notes"):
        return "researcher"
    if state.get("cost_estimate") is None:
        return "analyst"
    if not state.get("itinerary"):
        return "planner"
    return "FINISH"


def make_supervisor_node(llm: BaseChatModel, max_steps: int = MAX_STEPS):
    router = llm.with_structured_output(SupervisorDecision)

    async def supervisor_node(state: TravelState) -> dict:
        step = state.get("step_count", 0) + 1
        if step > max_steps:
            decision = SupervisorDecision(next_agent="FINISH", reason=f"Se alcanzó el tope de {max_steps} pasos.")
        else:
            try:
                decision = await router.ainvoke(
                    [SystemMessage(content=SUPERVISOR_PROMPT), HumanMessage(content=describe_state(state))]
                )
            except Exception as e:
                logger.warning("Supervisor sin decisión válida, uso la ruta fija: %s", e)
                decision = SupervisorDecision(next_agent=fallback_route(state), reason="Ruta fija por error del modelo.")
            if decision.next_agent in ("analyst", "planner") and not state.get("research_notes"):
                decision = SupervisorDecision(next_agent="researcher", reason="Faltan las notas del investigador.")
        logger.info("Paso %d -> %s (%s)", step, decision.next_agent, decision.reason)
        return {
            "next_agent": decision.next_agent,
            "step_count": step,
            "supervisor_feedback": decision.feedback or None,
            "messages": [AIMessage(content=f"-> {decision.next_agent}: {decision.reason}", name="supervisor")],
        }

    return supervisor_node
