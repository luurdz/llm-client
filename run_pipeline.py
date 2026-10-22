import asyncio
import logging

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from pipeline.chain import IncompleteOutputError, process_text, process_texts

EXAMPLES = {
    "normal": (
        "2026-10-19 14:32:10 ERROR [order-service] Timeout de 30s al llamar a payment-service. "
        "El servicio de pedidos está hecho en FastAPI, guarda los pedidos en PostgreSQL y "
        "publica eventos en Kafka. La tasa de aciertos del caché en Redis bajó al 5% y el pool "
        "de conexiones de PostgreSQL está agotado, así que los clientes no pueden terminar sus compras."
    ),
    "ambiguous": (
        "El sistema anda más lento desde ayer. Puede ser la base de datos o el caché, "
        "no estamos seguros. Algunos usuarios se quejaron."
    ),
    "non_technical": "Hoy hace mucho calor y mañana va a llover.",
}


async def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("google_genai").setLevel(logging.WARNING)

    load_dotenv()

    try:
        print("\n=== Single text (ainvoke) ===")
        result = await process_text(EXAMPLES["normal"])
        print(result.model_dump_json(by_alias=True, indent=2))
    except (IncompleteOutputError, OutputParserException, ValidationError) as e:
        print(f"Extraction failed: {e}")

    print("\n=== Batch (abatch) ===")

    texts = list(EXAMPLES.values())
    results = await process_texts(texts)
    for name, result in zip(EXAMPLES.keys(), results):
        print(f"\n--- {name} ---")
        if isinstance(result, Exception):
            print(f"Extraction failed: {str(result).splitlines()[0]}")
        else:
            print(result.model_dump_json(by_alias=True, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
