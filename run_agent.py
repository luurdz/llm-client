import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, messages_to_dict
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from agent.graph import build_agent_graph, build_llm
from agent.tools import make_tools
from cloud_rag.connection import connect_pinecone
from cloud_rag.ingest import load_chunk_metas
from cloud_rag.rag_system import RAGSystem

DATA_DIR = Path("data")
MEMORY_PATH = "agent_memory.sqlite"
TRACE_PATH = Path("traces/agent_trace.json")
RECURSION_LIMIT = 10

async def ask(app: CompiledStateGraph, thread_id: str, question: str, recursion_limit: int = RECURSION_LIMIT) -> list:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    messages_before = len((await app.aget_state(config)).values.get("messages", []))
    print(f"Question: {question}")
    result = await app.ainvoke({"messages": [HumanMessage(question)]}, config=config)
    tools_used = []
    for new_message in result["messages"][messages_before:]:
        for tool in getattr(new_message, "tool_calls", None) or []:
            tools_used.append(tool["name"])
    print(f"Tools: {tools_used}")
    print(result["messages"][-1].text)
    return result["messages"]


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    load_dotenv()
    try:
        index, embeddings, namespace = connect_pinecone()
        llm = build_llm()
    except (ValueError, ValidationError) as e:
        print(f"Configuration error: {e}")
        return
    rag = RAGSystem(index=index, embeddings=embeddings, namespace=namespace,metas=load_chunk_metas(DATA_DIR))
    tools = make_tools(rag)
    graph = build_agent_graph(llm=llm, tools=tools)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    async with AsyncSqliteSaver.from_conn_string(MEMORY_PATH) as saver:
        app = graph.compile(checkpointer=saver)
        print("Agent ready")
        multi_step = await ask(app, f"{run_id}-multipaso", "La reserva RSV7K2 no me deja emitir, ¿qué hago?")
        clarify = await ask(app, f"{run_id}-aclaracion", "¿En qué estado está la reserva RSV7I2?")
        retry = await ask(app, f"{run_id}-reintento", "¿Está confirmada la reserva RSV3P8?")
        memory = await ask(app, f"{run_id}-multipaso", "¿Y cómo se llama el pasajero?")
        other = await ask(app, f"{run_id}-otro-vendedor", "¿Y cómo se llama el pasajero?")
        memory_config = {"configurable": {"thread_id": f"{run_id}-multipaso"}}
        snapshot = await app.aget_state(memory_config)
        print(f"{run_id}-multipaso: {len(snapshot.values['messages'])} mensajes")
        try:
            await ask(app, f"{run_id}-limite", "La reserva RSV7K2 no me deja emitir, ¿qué hago?", recursion_limit=3)
        except GraphRecursionError:
            print("Cortado: superó los 3 pasos")
        trace = {"multi_step": messages_to_dict(multi_step),
                 "clarify": messages_to_dict(clarify),
                 "retry": messages_to_dict(retry),
                 "memory": messages_to_dict(memory),
                 "other": messages_to_dict(other),
                 }
        TRACE_PATH.parent.mkdir(exist_ok=True)
        TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Trace path: {TRACE_PATH}")


if __name__ == "__main__":
    asyncio.run(main())