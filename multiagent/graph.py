from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from cloud_rag.rag_system import RAGSystem
from multiagent.agents.analyst_agent import build_analyst_agent, make_analyst_node
from multiagent.agents.planner_agent import build_planner_agent, make_planner_node
from multiagent.agents.research_agent import build_research_agent, make_research_node
from multiagent.state import TravelState
from multiagent.supervisor import MAX_STEPS, make_supervisor_node

AGENTS = ("researcher", "analyst", "planner")


def route(state: TravelState) -> str:
    return state["next_agent"]


def build_travel_graph(llm: BaseChatModel, rag: RAGSystem | None, max_steps: int = MAX_STEPS) -> StateGraph:
    graph = StateGraph(TravelState)
    graph.add_node("supervisor", make_supervisor_node(llm, max_steps))
    graph.add_node("researcher", make_research_node(build_research_agent(llm, rag)))
    graph.add_node("analyst", make_analyst_node(build_analyst_agent(llm)))
    graph.add_node("planner", make_planner_node(build_planner_agent(llm)))
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor", route, {"researcher": "researcher", "analyst": "analyst", "planner": "planner", "FINISH": END}
    )
    for agent in AGENTS:
        graph.add_edge(agent, "supervisor")
    return graph
