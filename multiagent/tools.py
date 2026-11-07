from langchain_core.tools import tool

from multiagent.state import CostEstimate


@tool(response_format="content_and_artifact")
def calculate_trip_cost(price_per_person: float, people: int, nights: int, extra_per_night: float = 0, discount_rate: float = 0) -> tuple[str, CostEstimate]:
    """Calcula el costo total de un viaje y la seña del 30% para confirmar la reserva.

    Usala siempre que haya que dar un precio total. No hagas la cuenta vos.

    Args:
        price_per_person: precio del paquete por persona en dólares, por ejemplo 1950.
        people: cantidad de pasajeros, por ejemplo 2.
        nights: noches del paquete, por ejemplo 7.
        extra_per_night: extras por noche y por persona en dólares (por ejemplo, upgrade de hotel 35). 0 si no hay extras.
        discount_rate: descuento sobre el total, como fracción (0.05 = 5% por pago total en efectivo o transferencia). 0 si no hay descuento.
    """
    total = (price_per_person + extra_per_night * nights) * people
    total = round(total - (total * discount_rate), 2)
    deposit = round(total * 0.3, 2)
    cost_estimate = CostEstimate(people=people, total=total, deposit=deposit)
    return cost_estimate.model_dump_json(), cost_estimate