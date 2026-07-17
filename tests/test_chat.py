"""Tests de la capa de chat (endpoints /chat y /chat/resume), offline.

Monkeypatcheamos el grafo que usa app.chat para no llamar a Gemini/Tavily.
Cubren: respuesta normal, approval (interrupt del email), resume, y error.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

import app.chat as chat_mod


class _FakeInterrupt:
    def __init__(self, value):
        self.value = value


class _FakeGraph:
    """Grafo falso configurable para cada escenario."""

    def __init__(self, mode):
        self.mode = mode
        self.resumed_with = None

    def invoke(self, payload, config=None):
        from langgraph.types import Command

        if isinstance(payload, Command):
            self.resumed_with = payload.resume
            # Tras aprobar/rechazar, el bot responde con texto.
            txt = "Reporte enviado (dry-run)." if payload.resume == "approve" else "Cancelado."
            return {"messages": [AIMessage(content=txt)]}

        if self.mode == "message":
            return {"messages": [AIMessage(content="Hola, soy el agente tech.")]}
        if self.mode == "approval":
            return {
                "__interrupt__": [
                    _FakeInterrupt(
                        {
                            "action": "send_email_report",
                            "to": "me@example.com",
                            "subject": "Resumen",
                            "body": "Top 3...",
                            "message": "¿Confirmás el envío?",
                        }
                    )
                ]
            }
        if self.mode == "error":
            raise RuntimeError("429 ResourceExhausted: quota")
        raise AssertionError("modo desconocido")


def test_send_chat_message(monkeypatch):
    monkeypatch.setattr(chat_mod, "graph", _FakeGraph("message"))
    out = chat_mod.send_chat("hola", thread_id="t1")
    assert out["type"] == "message"
    assert "agente tech" in out["text"]
    assert out["thread_id"] == "t1"


def test_send_chat_generates_thread_id(monkeypatch):
    monkeypatch.setattr(chat_mod, "graph", _FakeGraph("message"))
    out = chat_mod.send_chat("hola")
    assert out["thread_id"].startswith("chat-")


def test_send_chat_approval(monkeypatch):
    monkeypatch.setattr(chat_mod, "graph", _FakeGraph("approval"))
    out = chat_mod.send_chat("mandá el reporte por email", thread_id="t2")
    assert out["type"] == "approval"
    assert out["detail"]["action"] == "send_email_report"
    assert out["detail"]["to"] == "me@example.com"
    assert "Confirmás" in out["prompt"]


def test_resume_approve(monkeypatch):
    g = _FakeGraph("approval")
    monkeypatch.setattr(chat_mod, "graph", g)
    out = chat_mod.resume_chat("t2", "approve")
    assert out["type"] == "message"
    assert "dry-run" in out["text"].lower()
    assert g.resumed_with == "approve"


def test_resume_reject(monkeypatch):
    g = _FakeGraph("approval")
    monkeypatch.setattr(chat_mod, "graph", g)
    out = chat_mod.resume_chat("t2", "reject")
    assert out["type"] == "message"
    assert "cancelado" in out["text"].lower()


def test_send_chat_error_is_friendly(monkeypatch):
    monkeypatch.setattr(chat_mod, "graph", _FakeGraph("error"))
    out = chat_mod.send_chat("hola", thread_id="t3")
    assert out["type"] == "error"
    assert "saturado" in out["text"].lower()  # mensaje de cuota amable
    assert "Traceback" not in out["text"]


def test_endpoints_via_testclient(monkeypatch):
    """Smoke test HTTP real (in-process) de /chat y /chat/resume."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(chat_mod, "graph", _FakeGraph("message"))
    from app.server import app

    client = TestClient(app)

    r = client.post("/chat", json={"message": "hola", "thread_id": "t4"})
    assert r.status_code == 200
    assert r.json()["type"] == "message"

    r = client.post("/chat/resume", json={"thread_id": "t4", "decision": "approve"})
    assert r.status_code == 200
    assert r.json()["type"] == "message"
