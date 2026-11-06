from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.prompts import SYSTEM_PROMPT
from llm_client.config import load_config
from llm_client.langchain_models import build_chat_model


def build_llm() -> BaseChatModel:
    config = load_config()
    config_t0 = config.model_copy(update={"temperature": 0})
    return build_chat_model(config_t0)

def build_agent_graph(llm: BaseChatModel, tools: list[BaseTool]) -> StateGraph:
    llm_with_tools = llm.bind_tools(tools)

    async def call_model(state: MessagesState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)
    graph.add_edge("tools", "model")
    return graph