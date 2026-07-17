"""Estado del grafo LangGraph."""

from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Estado que fluye por el grafo.

    - messages: historial de la conversación. `add_messages` es el reducer que
      appendea (y mergea por id) en vez de sobreescribir — así la memoria de
      conversación se acumula turno a turno (persistida por el checkpointer).
    - user_role: rol/persona con el que el usuario quiere que le hablen
      ("dev backend en fintech", "hincha de Boca"). El agente adapta el tono.
    - tool_call_counts: contador por tool para el rate limiting por conversación.
    - blocked: si un guardrail de entrada cortó el flujo (ruteo a END).
    - thinking_budget: tope de razonamiento del LLM para este turno. None usa el
      default de chat; el briefing lo setea más alto porque sí necesita razonar.
    """

    messages: Annotated[list, add_messages]
    user_role: Optional[str]
    tool_call_counts: dict[str, int]
    blocked: bool
    thinking_budget: Optional[int]
