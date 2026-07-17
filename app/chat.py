"""Capa de chat para el front (React).

Envuelve el grafo de LangGraph y devuelve un JSON simple, sin exponer el estado
interno del grafo. Tres formas de respuesta:

- {"type": "message",  "text": ..., "thread_id": ...}   respuesta normal del bot
- {"type": "approval", "prompt": ..., "detail": {...}, "thread_id": ...}
      el bot quiere enviar el email → el front muestra botones Aprobar/Rechazar
      y luego llama a resume_chat(thread_id, "approve"|"reject")
- {"type": "error",    "text": ..., "thread_id": ...}    error atrapado (cuota, red)

El HITL vive en el grafo (interrupt dentro de send_email_report). Acá solo lo
traducimos a un contrato limpio para el front.
"""

from __future__ import annotations

import logging
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from app.graph import graph

logger = logging.getLogger("tech_news_agent.chat")


def _new_thread_id() -> str:
    return f"chat-{uuid.uuid4().hex[:12]}"


def _last_ai_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content:
            return msg.content
    return ""


def _shape_result(result: dict, thread_id: str) -> dict:
    """Convierte el resultado del grafo en el contrato de chat."""
    # ¿El grafo se pausó esperando confirmación humana (HITL del email)?
    interrupts = result.get("__interrupt__")
    if interrupts:
        payload = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
        payload = payload if isinstance(payload, dict) else {"message": str(payload)}
        return {
            "type": "approval",
            "prompt": payload.get("message", "¿Confirmás la acción?"),
            "detail": {
                "action": payload.get("action"),
                "to": payload.get("to"),
                "subject": payload.get("subject"),
                "body": payload.get("body"),
            },
            "thread_id": thread_id,
        }

    return {
        "type": "message",
        "text": _last_ai_text(result.get("messages", [])),
        "thread_id": thread_id,
    }


def _error_response(exc: Exception, thread_id: str) -> dict:
    """Traduce errores técnicos a un mensaje amable para el usuario."""
    name = type(exc).__name__
    text = "Uf, algo salió mal procesando tu mensaje. Probá de nuevo en un momento."
    if "ResourceExhausted" in name or "429" in str(exc):
        text = (
            "Estoy saturado por límite de uso del modelo 😅. "
            "Probá de nuevo en un ratito."
        )
    elif "Timeout" in name or "Connection" in name:
        text = "No pude conectarme al servicio ahora mismo. Reintentá en unos segundos."
    logger.exception("Error en chat (thread=%s)", thread_id)
    return {"type": "error", "text": text, "thread_id": thread_id}


def send_chat(
    message: str, thread_id: str | None = None, user_role: str | None = None
) -> dict:
    """Procesa un turno de chat del usuario."""
    thread_id = thread_id or _new_thread_id()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=message)],
                "user_role": user_role,
                "tool_call_counts": {},
                "blocked": False,
            },
            config=config,
        )
        return _shape_result(result, thread_id)
    except Exception as exc:  # noqa: BLE001 — queremos degradar cualquier fallo
        return _error_response(exc, thread_id)


def resume_chat(thread_id: str, decision: str) -> dict:
    """Reanuda un grafo pausado en el HITL del email con la decisión humana.

    decision: "approve" | "reject".
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = graph.invoke(Command(resume=decision), config=config)
        return _shape_result(result, thread_id)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc, thread_id)
