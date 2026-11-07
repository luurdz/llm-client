import asyncio
import itertools

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError

from agent.graph import build_agent_graph
from agent.tools import make_tools


class FakeToolModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])

def test_retry_calls_the_tool_twice():
    model = FakeToolModel(
        messages=iter(
            [
                tool_call("get_booking", {"locator": "RSV3P8"}, "1"),
                tool_call("get_booking", {"locator": "RSV3P8"}, "2"),
                AIMessage(content="Confirmada"),
            ]
        )
    )
    app = build_agent_graph(model, make_tools(None)).compile()

    result = asyncio.run(app.ainvoke({"messages": [HumanMessage("¿Está confirmada RSV3P8?")]}))

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2
    assert "RS-3100" in tool_messages[0].content
    assert "confirmada" in tool_messages[1].content
    assert not result["messages"][-1].tool_calls


def test_recursion_limit_stops_endless_tool_calls():
    endless_calls = (tool_call("get_booking", {"locator": "RSV5M4"}, str(i)) for i in itertools.count())
    model = FakeToolModel(messages=endless_calls)
    app = build_agent_graph(model, make_tools(None)).compile()

    with pytest.raises(GraphRecursionError):
        asyncio.run(
            app.ainvoke(
                {"messages": [HumanMessage("¿Cómo está RSV5M4?")]},
                config={"recursion_limit": 5},
            )
        )


def test_memory_is_kept_per_thread(tmp_path):
    async def run():
        model = FakeToolModel(
            messages=iter(
                [
                    AIMessage(content="Hola"),
                    AIMessage(content="Seguimos"),
                    AIMessage(content="Otro"),
                ]
            )
        )
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "memory.sqlite")) as saver:
            app = build_agent_graph(model, make_tools(None)).compile(checkpointer=saver)
            same_thread = {"configurable": {"thread_id": "vendedor-1"}}
            other_thread = {"configurable": {"thread_id": "vendedor-2"}}

            await app.ainvoke({"messages": [HumanMessage("Hola")]}, config=same_thread)
            second = await app.ainvoke({"messages": [HumanMessage("¿Seguimos?")]}, config=same_thread)
            other = await app.ainvoke({"messages": [HumanMessage("Hola")]}, config=other_thread)

            assert len(second["messages"]) == 4
            assert len(other["messages"]) == 2

    asyncio.run(run())
