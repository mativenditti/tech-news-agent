"""Tests del streaming SSE: helpers, generadoras y endpoints. Offline
(monkeypatch del grafo, sin llamar a Gemini/Tavily)."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessageChunk

from app import streaming


def test_approval_event_from_interrupt():
    class _Interrupt:
        value = {
            "action": "send_email_report",
            "to": "me@example.com",
            "subject": "Resumen",
            "body": "Top 3...",
            "message": "¿Confirmás el envío?",
        }

    evt = streaming.approval_event(_Interrupt(), "t1")
    assert evt["event"] == "approval"
    data = json.loads(evt["data"])
    assert data["action"] == "send_email_report"
    assert data["to"] == "me@example.com"
    assert data["subject"] == "Resumen"
    assert data["body"] == "Top 3..."
    assert data["prompt"] == "¿Confirmás el envío?"
    assert data["thread_id"] == "t1"


def test_error_event_is_friendly():
    evt = streaming.error_event(RuntimeError("429 ResourceExhausted: quota"), "t2")
    assert evt["event"] == "error"
    data = json.loads(evt["data"])
    assert "saturado" in data["text"].lower()
    assert "Traceback" not in data["text"]
    assert data["thread_id"] == "t2"


def test_token_and_done_events():
    tok = streaming.token_event("Hola")
    assert tok["event"] == "token"
    assert json.loads(tok["data"]) == {"text": "Hola"}

    done = streaming.done_event("t3")
    assert done["event"] == "done"
    assert json.loads(done["data"]) == {"thread_id": "t3"}
