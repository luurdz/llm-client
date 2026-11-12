from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from multiagent.prompts import ANALYST_PROMPT
from multiagent.state import TravelState
from multiagent.tools import calculate_trip_cost


def build_analyst_agent(llm:BaseChatModel) -> CompiledStateGraph:
    return create_react_agent(llm, [calculate_trip_cost], prompt=ANALYST_PROMPT, name= "analyst")

def make_analyst_node(agent: CompiledStateGraph):
    async def analyst_node(state: TravelState) -> dict:
        text = f"{state["messages"][0].text}\n\nDatos:\n{state["research_notes"]}"
        feedback = state.get("supervisor_feedback")
        if feedback:
            text += f"\n\nCorrecciones del supervisor:\n{feedback}"
        results = await agent.ainvoke({"messages": [HumanMessage(content=text)]})
        artifact = None
        for message in results["messages"]:
            if isinstance(message, ToolMessage) and message.name == "calculate_trip_cost":
                artifact = message.artifact
        answer = AIMessage(content=results["messages"][-1].text, name="analyst")
        if artifact is None:
            return {"messages": [answer]}
        return {"cost_estimate": artifact, "messages": [answer]}
    return analyst_node