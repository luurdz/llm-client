import asyncio
import logging

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from rag.chain import get_rag_response

QUESTIONS = {
    "in_context": [
        "¿Cuánto me devuelven si cancelo un paquete de 2.000 dólares 20 días antes?",
        "¿Puedo llevar un power bank en la valija despachada?",
    ],
    "trap": [
        "¿Qué vacunas necesito para viajar a Kenia?",
        "¿Tienen paquetes a Tokio?",
    ],
}


async def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    load_dotenv()

    for kind, questions in QUESTIONS.items():
        print(f"\n=== Kind: {kind} ===")
        for question in questions:
            try:
                print(f"Question: {question}")
                result = await get_rag_response(question)
                print(f"Answer: {result.answer}")
                print(f"Sources: {result.sources}")
            except (ValueError, OutputParserException, ValidationError) as e:
                print(f"Rag failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
