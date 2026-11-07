from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from agent.tools import make_search_tool
from cloud_rag.rag_system import RAGSystem
from multiagent.prompts import RESEARCHER_PROMPT
from multiagent.state import TravelState


def build_research_agent(llm: BaseChatModel, rag: RAGSystem | None) -> CompiledStateGraph:
    return create_react_agent(llm, [make_search_tool(rag=rag)], prompt=RESEARCHER_PROMPT, name="researcher")


def make_research_node(agent: CompiledStateGraph):
    async def research_node(state: TravelState) -> dict:
        first_message= state["messages"][0].text
        result = await agent.ainvoke({"messages": [HumanMessage(content=first_message)]})
        response = result["messages"][-1].text
        return {"research_notes": response, "messages": [AIMessage(content=response, name="researcher")]}
    return research_node
