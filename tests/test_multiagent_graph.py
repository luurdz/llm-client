import asyncio

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from multiagent.agents.analyst_agent import make_analyst_node
from multiagent.agents.planner_agent import make_planner_node
from multiagent.graph import build_travel_graph
from multiagent.state import CostEstimate, SupervisorDecision

QUESTION = "Quiero el paquete Cusco y Machu Picchu para 2 personas con upgrade de hotel. ¿Cuánto sale y qué hacemos cada día?"
NOTES = (
    "Cusco y Machu Picchu — 7 noches, 1.950 dólares por persona. Incluye tren a Aguas Calientes, "
    "Machu Picchu con guía y excursión al Valle Sagrado. No planificar actividades exigentes el primer día. "
    "Mejora de hotel: 35 dólares por noche y por persona. Seña 30%."
)
ITINERARY = "Día 1: llegada y descanso\nDía 2: Valle Sagrado\nDía 3: tren a Aguas Calientes y Machu Picchu con guía"


class FakeTravelModel(GenericFakeChatModel):
    """Modelo falso: los agentes leen `messages` en orden y el supervisor lee `decisions` en orden."""

    decisions: list = []
    prompts: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        def decide(messages):
            self.prompts.append(messages[-1].content)
            decision = self.decisions.pop(0)
            if isinstance(decision, Exception):
                raise decision
            agent, feedback = decision if isinstance(decision, tuple) else (decision, "")
            return SupervisorDecision(next_agent=agent, feedback=feedback, reason="test")

        return RunnableLambda(decide)


def cost_call() -> AIMessage:
    args = {"price_per_person": 1950, "people": 2, "nights": 7, "extra_per_night": 35}
    return AIMessage(content="", tool_calls=[{"name": "calculate_trip_cost", "args": args, "id": "1"}])


def run(model: FakeTravelModel, max_steps: int = 8) -> dict:
    app = build_travel_graph(model, rag=None, max_steps=max_steps).compile()
    return asyncio.run(app.ainvoke({"messages": [HumanMessage(QUESTION)]}))


def test_supervisor_delegates_to_each_agent_and_finishes():
    model = FakeTravelModel(
        messages=iter([AIMessage(NOTES), cost_call(), AIMessage("Total 4390, seña 1317"), AIMessage(ITINERARY)]),
        decisions=["researcher", "analyst", "planner", "FINISH"],
    )
    result = run(model)

    assert result["research_notes"] == NOTES
    assert result["cost_estimate"] == CostEstimate(people=2, total=4390.0, deposit=1317.0)
    assert result["itinerary"] == ITINERARY
    assert result["step_count"] == 4
    authors = [m.name for m in result["messages"] if isinstance(m, AIMessage)]
    assert authors == ["supervisor", "researcher", "supervisor", "analyst", "supervisor", "planner", "supervisor"]
    assert "total 4390.0 dólares, seña 1317.0" in model.prompts[-1]


def test_step_cap_stops_an_endless_correction_loop():
    model = FakeTravelModel(messages=iter([AIMessage(NOTES)] * 10), decisions=["researcher"] * 10)
    result = run(model, max_steps=3)

    assert result["step_count"] == 4
    assert result["next_agent"] == "FINISH"
    assert len(model.prompts) == 3


def test_invalid_decision_falls_back_to_the_first_missing_piece():
    model = FakeTravelModel(
        messages=iter([AIMessage(NOTES)]),
        decisions=[ValueError("respuesta sin formato"), "FINISH"],
    )
    result = run(model)

    assert result["research_notes"] == NOTES
    assert [m.name for m in result["messages"] if isinstance(m, AIMessage)][:2] == ["supervisor", "researcher"]


def test_planner_without_notes_is_redirected_to_researcher():
    model = FakeTravelModel(messages=iter([AIMessage(NOTES)]), decisions=["planner", "FINISH"])
    result = run(model)

    assert result["research_notes"] == NOTES
    assert result.get("itinerary") is None


class RecordingAgent:
    """Agente falso: guarda lo que recibe y responde siempre `reply`."""

    def __init__(self, reply: str = ITINERARY):
        self.reply = reply
        self.inputs = []

    async def ainvoke(self, payload):
        self.inputs.append(payload["messages"][0].content)
        return {"messages": [AIMessage(self.reply)]}


def test_supervisor_feedback_reaches_the_planner():
    agent = RecordingAgent()
    node = make_planner_node(agent)
    state = {"messages": [HumanMessage(QUESTION)], "research_notes": NOTES, "supervisor_feedback": "Pasá Machu Picchu al día 3."}

    out = asyncio.run(node(state))

    assert "Correcciones del supervisor:\nPasá Machu Picchu al día 3." in agent.inputs[0]
    assert out["itinerary"] == ITINERARY
    assert out["messages"][0].name == "planner"


def test_planner_missing_data_keeps_the_previous_itinerary():
    node = make_planner_node(RecordingAgent(reply="**Falta dato: noches**"))
    state = {"messages": [HumanMessage(QUESTION)], "research_notes": NOTES, "itinerary": ITINERARY}

    out = asyncio.run(node(state))

    assert "itinerary" not in out
    assert out["messages"][0].text == "**Falta dato: noches**"


def test_analyst_receives_feedback_and_keeps_estimate_without_tool_call():
    agent = RecordingAgent(reply="Falta dato: cantidad de pasajeros")
    node = make_analyst_node(agent)
    state = {"messages": [HumanMessage(QUESTION)], "research_notes": NOTES, "supervisor_feedback": "Recalculá con 2 pasajeros."}

    out = asyncio.run(node(state))

    assert "Correcciones del supervisor:\nRecalculá con 2 pasajeros." in agent.inputs[0]
    assert "cost_estimate" not in out
    assert out["messages"][0].name == "analyst"


def test_rejected_itinerary_goes_back_to_the_planner_with_feedback():
    model = FakeTravelModel(
        messages=iter([AIMessage(NOTES), AIMessage("Día 1: Machu Picchu"), AIMessage(ITINERARY)]),
        decisions=["researcher", "planner", ("planner", "El día 1 no puede tener Machu Picchu."), "FINISH"],
    )
    result = run(model)

    assert result["itinerary"] == ITINERARY
    assert result["supervisor_feedback"] is None
    supervisor_messages = [m.text for m in result["messages"] if m.name == "supervisor"]
    assert "feedback: El día 1 no puede tener Machu Picchu." in supervisor_messages[2]


def test_supervisor_sees_the_planner_missing_data_answer():
    model = FakeTravelModel(
        messages=iter([AIMessage(NOTES), AIMessage("Falta dato: noches")]),
        decisions=["researcher", "planner", "FINISH"],
    )
    result = run(model)

    assert result.get("itinerary") is None
    assert "Última respuesta del planificador:\nFalta dato: noches" in model.prompts[2]
