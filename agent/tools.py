import asyncio
import re

from langchain_core.tools import BaseTool, tool

from cloud_rag.rag_system import RAGSystem

# 6 characters, digits and uppercase letters except I and O (manual_sistema_de_reservas.md, line 39).
LOCATOR_PATTERN = re.compile(r"^[A-HJ-NP-Z0-9]{6}$")

BOOKINGS: dict[str, dict] = {
    "RSV7K2": {"passenger": "Lucía Fernández", "product": "Vuelo Buenos Aires - Madrid", "status": "bloqueada", "error": "RS-2210"},
    "RSV3P8": {"passenger": "Martín Gómez", "product": "Paquete Bariloche 7 noches", "status": "confirmada", "error": None, "transient_error": "RS-3100"},
    "RSV5M4": {"passenger": "Sofía Ruiz", "product": "Hotel en Salta 3 noches", "status": "confirmada", "error": None},
}

def make_tools(rag: RAGSystem | None) -> list[BaseTool]:
    """Crea las tools del agente.

    Cada llamada devuelve tools nuevas con su propio contador de intentos por localizador,
    que se usa para simular errores transitorios del proveedor.

    Args:
        rag: sistema RAG sobre la documentación de Rumbo Sur. Puede ser None si no se usa la búsqueda.

    Returns:
        Lista de tools para pasar a llm.bind_tools y a ToolNode.
    """
    attempts={}

    @tool
    def get_booking(locator: str) -> str:
        """Consulta una reserva en el sistema de reservas a partir de su localizador.

        Usala cuando el usuario pregunte por el estado o los datos de una reserva.
        El localizador tiene 6 caracteres, letras mayúsculas y dígitos, y nunca usa las letras I ni O (ejemplo: RSV7K2).

        Args:
            locator: localizador de la reserva.

        Returns:
            Los datos de la reserva (pasajero, producto, estado y código de error si lo tiene),
            o un texto que empieza con "ERROR RS-xxxx":
            - RS-1001: el localizador tiene un formato inválido o no existe. Pedile al usuario que lo verifique.
            - RS-3100: el proveedor no respondió. Es transitorio, volvé a llamar a la tool.
        """
        clean_locator = locator.strip().upper()
        if not LOCATOR_PATTERN.match(clean_locator):
            return f"ERROR RS-1001: el localizador {locator} tiene un formato inválido. Tiene 6 caracteres y nunca usa las letras I ni O. Pedile al vendedor que lo verifique."

        if clean_locator not in BOOKINGS:
            return f"ERROR RS-1001. La reserva {locator} no existe."

        attempts[clean_locator] = attempts.get(clean_locator, 0) + 1

        booking = BOOKINGS[clean_locator]

        if booking.get("transient_error") and attempts[clean_locator] == 1:
            return f"ERROR {booking['transient_error']}: el proveedor no respondió. Intentá de nuevo."

        response = f"Reserva {clean_locator}: pasajero {booking["passenger"]}, producto {booking["product"]}, estado {booking["status"]}"

        if booking.get("error"):
            response += f", error {booking['error']}"
        return response
    return [get_booking]