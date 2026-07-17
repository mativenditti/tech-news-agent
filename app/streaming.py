"""Helpers de streaming SSE.

Traducen los resultados del grafo (chunks de texto, interrupt del HITL, errores)
a eventos SSE semánticos: dicts {"event": str, "data": <json string>} que
EventSourceResponse serializa al wire. Compartido por chat.py y briefing.py para
no duplicar el formato de approval/error.
"""

from __future__ import annotations

import json

from app.chat import _error_response


def token_event(text: str) -> dict:
    """Evento de un fragmento de texto de la respuesta del agente."""
    return {"event": "token", "data": json.dumps({"text": text})}


def done_event(thread_id: str) -> dict:
    """Evento de cierre normal (éxito)."""
    return {"event": "done", "data": json.dumps({"thread_id": thread_id})}


def approval_event(interrupt, thread_id: str) -> dict:
    """Evento de HITL: el grafo se pausó pidiendo confirmar el envío del email.

    `interrupt` es el objeto Interrupt de LangGraph; su .value es el payload que
    send_email_report pasó a interrupt(...) (action/to/subject/body/message).
    """
    payload = interrupt.value if isinstance(interrupt.value, dict) else {}
    data = {
        "prompt": payload.get("message", "¿Confirmás la acción?"),
        "action": payload.get("action"),
        "to": payload.get("to"),
        "subject": payload.get("subject"),
        "body": payload.get("body"),
        "thread_id": thread_id,
    }
    return {"event": "approval", "data": json.dumps(data)}


def error_event(exc: Exception, thread_id: str) -> dict:
    """Evento de error, con el mensaje amable de _error_response."""
    friendly = _error_response(exc, thread_id)  # {"type","text","thread_id"}
    return {
        "event": "error",
        "data": json.dumps({"text": friendly["text"], "thread_id": thread_id}),
    }
