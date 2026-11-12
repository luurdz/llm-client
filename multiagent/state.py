from typing import Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class CostEstimate(BaseModel):
    people: int = Field(gt=0, description="Cantidad de pasajeros del viaje")
    total: float = Field(ge=0, description="Costo total del viaje en dólares, con extras incluidos")
    deposit: float = Field(ge=0, description="Seña en dólares para confirmar la reserva (30% del total)")

NextAgent = Literal["researcher", "analyst", "planner", "FINISH"]


class SupervisorDecision(BaseModel):
    next_agent: NextAgent = Field(description="Agente que trabaja ahora, o FINISH si la respuesta está completa y cumple la rúbrica")
    feedback: str = Field(default="", description="Qué tiene que corregir el agente elegido. Vacío si no hay nada que corregir")
    reason: str = Field(description="Por qué se eligió ese agente, en una oración")


class TravelState(MessagesState):
    research_notes: str | None
    cost_estimate: CostEstimate | None
    itinerary: str | None
    next_agent: NextAgent
    step_count: int
    supervisor_feedback: str | None
