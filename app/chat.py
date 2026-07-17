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
    # Ojo: Gemini 3 ("thinking") devuelve content como lista de bloques
    # ([{"type":"text","text":...}]), no como string, para poder adjuntar el
    # thought_signature. Por eso NO chequeamos isinstance(content, str) (daría ""):
    # usamos AIMessage.text, que extrae el texto tanto de content string como lista.
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            text = msg.text
            if text:
                return text
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


def build_initial_state(message: str, user_role: str | None) -> dict:
    """Estado inicial del grafo para un turno nuevo (AgentState).

    Único lugar donde se arma este dict, compartido por los caminos sync
    (send_chat) y streaming (stream_chat, stream_briefing). Si se agrega un campo
    a AgentState, se actualiza acá y no en cuatro lugares.
    """
    return {
        "messages": [HumanMessage(content=message)],
        "user_role": user_role,
        "tool_call_counts": {},
        "blocked": False,
    }


def send_chat(
    message: str, thread_id: str | None = None, user_role: str | None = None
) -> dict:
    """Procesa un turno de chat del usuario."""
    thread_id = thread_id or _new_thread_id()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = graph.invoke(build_initial_state(message, user_role), config=config)
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


def _stream_graph_events(inputs: dict, thread_id: str):
    """Corre graph.stream y emite eventos SSE (dicts {"event","data"}).

    Común a stream_chat y stream_briefing: streamea los fragmentos de texto del
    nodo agent como `token`; al terminar emite `approval` si el grafo quedó en el
    HITL del email, o `done`. Cualquier excepción se degrada a un evento `error`.

    Nota para tests: el loop corre sobre el `graph` de ESTE módulo (app.chat),
    incluso cuando se lo invoca desde stream_briefing. Para mockearlo, parcheá
    `app.chat.graph`, no el `graph` del módulo del caller.
    """
    from app import streaming  # import local para evitar ciclo con streaming.py

    config = {"configurable": {"thread_id": thread_id}}
    try:
        stream = graph.stream(inputs, config, stream_mode="messages")
        for chunk, metadata in stream:
            if metadata.get("langgraph_node") != "agent":
                continue
            text = chunk.text
            if text:
                yield streaming.token_event(text)

        interrupts = graph.get_state(config).interrupts
        if interrupts:
            yield streaming.approval_event(interrupts[0], thread_id)
        else:
            yield streaming.done_event(thread_id)
    except Exception as exc:  # noqa: BLE001 — degradar cualquier fallo a evento error
        yield streaming.error_event(exc, thread_id)


def stream_chat(message: str, thread_id: str | None = None, user_role: str | None = None):
    """Streamea un turno de chat como eventos SSE (dicts {"event","data"}).

    Emite `token` por cada fragmento de texto del nodo agent; al terminar, emite
    `approval` si el grafo se pausó en el HITL del email, o `done` en caso normal.
    Cualquier excepción se traduce a un evento `error` y cierra el stream.
    """
    thread_id = thread_id or _new_thread_id()
    yield from _stream_graph_events(build_initial_state(message, user_role), thread_id)
