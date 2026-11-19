from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.hitl import JobState, approval_node, needs_approval
from cloud_rag.rag_system import RAGSystem
from multiagent.graph import build_travel_graph

STATE_CLASSES = [("multiagent", "state", "CostEstimate")]


@asynccontextmanager
async def open_checkpointer(redis_url: str) -> AsyncIterator[AsyncRedisSaver]:
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        saver.serde = JsonPlusRedisSerializer(allowed_json_modules=STATE_CLASSES)
        await saver.asetup()
        yield saver


def build_job_graph(llm: BaseChatModel, rag: RAGSystem | None, checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    graph = StateGraph(JobState)
    graph.add_node("travel", build_travel_graph(llm, rag).compile())
    graph.add_node("approval", approval_node)
    graph.add_edge(START, "travel")
    graph.add_conditional_edges("travel", needs_approval, ["approval", END])
    graph.add_edge("approval", END)
    return graph.compile(checkpointer=checkpointer)
