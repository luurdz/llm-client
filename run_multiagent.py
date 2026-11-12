import asyncio
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, messages_to_dict
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import ValidationError

from agent.graph import build_llm
from cloud_rag.connection import connect_pinecone
from cloud_rag.ingest import load_chunk_metas
from cloud_rag.rag_system import RAGSystem
from multiagent.graph import build_travel_graph

DATA_DIR = Path("data")
TRACE_PATH = Path("traces/multiagent_trace.json")
REQUESTS_PER_MINUTE = 10  # Gemini free tier allows 15
QUESTION = (
    "Quiero el paquete Cusco y Machu Picchu para 2 personas con upgrade de hotel. "
    "¿Cuánto sale, cuánto es la seña y qué hacemos cada día?"
)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("google_genai", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    load_dotenv()
    try:
        index, embeddings, namespace = connect_pinecone()
        llm = build_llm()
        llm.rate_limiter = InMemoryRateLimiter(requests_per_second=REQUESTS_PER_MINUTE / 60, max_bucket_size=1)
    except (ValueError, ValidationError) as e:
        print(f"Configuration error: {e}")
        return
    rag = RAGSystem(index=index, embeddings=embeddings, namespace=namespace, metas=load_chunk_metas(DATA_DIR))
    app = build_travel_graph(llm, rag).compile()

    print(f"Consulta: {QUESTION}\n")
    result = await app.ainvoke({"messages": [HumanMessage(QUESTION)]})

    for message in result["messages"][1:]:
        print(f"[{message.name}] {message.text}\n")
    print(f"cost_estimate: {result.get('cost_estimate')!r}")
    print(f"pasos del supervisor: {result['step_count']}")

    TRACE_PATH.parent.mkdir(exist_ok=True)
    trace = {
        "question": QUESTION,
        "research_notes": result.get("research_notes"),
        "cost_estimate": result["cost_estimate"].model_dump() if result.get("cost_estimate") else None,
        "itinerary": result.get("itinerary"),
        "step_count": result["step_count"],
        "messages": messages_to_dict(result["messages"]),
    }
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Trace: {TRACE_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
