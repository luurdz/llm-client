from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from multiagent.prompts import PLANNER_PROMPT
from multiagent.state import TravelState


def build_planner_agent(llm: BaseChatModel) -> CompiledStateGraph:
    return create_react_agent(llm, [], prompt=PLANNER_PROMPT, name="planner")


def make_planner_node(agent: CompiledStateGraph):
    async def planner_node(state: TravelState) -> dict:
        text = f"{state['messages'][0].text}\n\nDatos:\n{state['research_notes']}"
        feedback = state.get("supervisor_feedback")
        previous = state.get("itinerary")
        if feedback and previous:
            text += f"\n\nItinerario anterior:\n{previous}\n\nCorrecciones del supervisor:\n{feedback}"
        elif feedback:
            text += f"\n\nCorrecciones del supervisor:\n{feedback}"
        result = await agent.ainvoke({"messages": [HumanMessage(content=text)]})
        answer = result["messages"][-1].text
        message = AIMessage(content=answer, name="planner")
        if answer.startswith("Falta dato"):
            return {"messages": [message]}
        return {"itinerary": answer, "messages": [message]}
    return planner_node
