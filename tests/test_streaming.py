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


from langchain_core.messages import AIMessageChunk  # noqa: E402  (top ya lo importa)
import app.chat as chat_mod


class _FakeStreamGraph:
    """Grafo falso para streaming. `mode` elige el escenario.

    - "message": emite 2 chunks de texto del nodo agent, sin interrupt.
    - "approval": emite 1 chunk y deja un interrupt pendiente en get_state.
    - "error": lanza al iterar el stream.
    """

    def __init__(self, mode):
        self.mode = mode

    def stream(self, payload, config=None, stream_mode=None):
        assert stream_mode == "messages"
        if self.mode == "error":
            raise RuntimeError("429 ResourceExhausted: quota")
        yield (AIMessageChunk(content="Hola"), {"langgraph_node": "agent"})
        # Un chunk de otro nodo / vacío que NO debe emitirse como token:
        yield (AIMessageChunk(content=""), {"langgraph_node": "tools"})
        if self.mode == "message":
            yield (AIMessageChunk(content=", soy el bot."), {"langgraph_node": "agent"})

    def get_state(self, config):
        class _State:
            interrupts = ()
        if self.mode == "approval":
            class _Interrupt:
                value = {
                    "action": "send_email_report",
                    "to": "me@example.com",
                    "subject": "Resumen",
                    "body": "Top 3...",
                    "message": "¿Confirmás el envío?",
                }
            _State.interrupts = (_Interrupt(),)
        return _State()


def _collect(gen):
    return [(e["event"], e["data"]) for e in gen]


def test_stream_chat_message(monkeypatch):
    monkeypatch.setattr(chat_mod, "graph", _FakeStreamGraph("message"))
    events = _collect(chat_mod.stream_chat("hola", thread_id="t1"))
    kinds = [k for k, _ in events]
    # Solo los chunks del nodo agent con texto -> 2 tokens, luego done.
    assert kinds == ["token", "token", "done"]
    import json
    assert json.loads(events[0][1])["text"] == "Hola"
    assert json.loads(events[1][1])["text"] == ", soy el bot."
    assert json.loads(events[2][1])["thread_id"] == "t1"


def test_stream_chat_approval(monkeypatch):
    monkeypatch.setattr(chat_mod, "graph", _FakeStreamGraph("approval"))
    events = _collect(chat_mod.stream_chat("mandá el email", thread_id="t2"))
    kinds = [k for k, _ in events]
    assert kinds[-1] == "approval"          # termina en approval, no en done
    assert "done" not in kinds
    import json
    data = json.loads(events[-1][1])
    assert data["action"] == "send_email_report"
    assert data["thread_id"] == "t2"


def test_stream_chat_error(monkeypatch):
    monkeypatch.setattr(chat_mod, "graph", _FakeStreamGraph("error"))
    events = _collect(chat_mod.stream_chat("hola", thread_id="t3"))
    assert events[-1][0] == "error"
    import json
    assert "saturado" in json.loads(events[-1][1])["text"].lower()


def test_stream_chat_generates_thread_id(monkeypatch):
    monkeypatch.setattr(chat_mod, "graph", _FakeStreamGraph("message"))
    events = _collect(chat_mod.stream_chat("hola"))
    import json
    tid = json.loads(events[-1][1])["thread_id"]
    assert tid.startswith("chat-")


import app.briefing as briefing_mod


def test_stream_briefing_message(monkeypatch):
    # _stream_graph_events usa chat_mod.graph, así que parcheamos ESE grafo.
    monkeypatch.setattr(chat_mod, "graph", _FakeStreamGraph("message"))
    events = _collect(briefing_mod.stream_briefing(thread_id="b1"))
    kinds = [k for k, _ in events]
    assert kinds == ["token", "token", "done"]
    import json
    assert json.loads(events[-1][1])["thread_id"] == "b1"


def test_stream_briefing_generates_thread_id(monkeypatch):
    monkeypatch.setattr(chat_mod, "graph", _FakeStreamGraph("message"))
    events = _collect(briefing_mod.stream_briefing())
    import json
    assert json.loads(events[-1][1])["thread_id"].startswith("briefing-")


def test_stream_briefing_error(monkeypatch):
    monkeypatch.setattr(chat_mod, "graph", _FakeStreamGraph("error"))
    events = _collect(briefing_mod.stream_briefing(thread_id="b2"))
    assert events[-1][0] == "error"


def _parse_sse(body: str):
    """Parsea el cuerpo text/event-stream en una lista de (event, data)."""
    events = []
    cur_event, cur_data = None, None
    for line in body.splitlines():
        if line.startswith("event:"):
            cur_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            cur_data = line[len("data:"):].strip()
        elif line == "":
            if cur_event is not None:
                events.append((cur_event, cur_data))
            cur_event, cur_data = None, None
    if cur_event is not None:
        events.append((cur_event, cur_data))
    return events


def test_chat_stream_endpoint(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(chat_mod, "graph", _FakeStreamGraph("message"))
    from app.server import app

    client = TestClient(app)
    r = client.post("/chat/stream", json={"message": "hola", "thread_id": "s1"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(r.text)
    kinds = [k for k, _ in events]
    assert "token" in kinds
    assert kinds[-1] == "done"


def test_briefing_stream_endpoint(monkeypatch):
    from fastapi.testclient import TestClient

    # stream_briefing corre sobre app.chat.graph (via _stream_graph_events), así
    # que parcheamos chat_mod.graph, no briefing_mod.graph.
    monkeypatch.setattr(chat_mod, "graph", _FakeStreamGraph("message"))
    from app.server import app

    client = TestClient(app)
    r = client.post("/briefing/stream", json={"thread_id": "s2"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(r.text)
    assert events[-1][0] == "done"


def test_chat_stream_endpoint_error_path(monkeypatch):
    """Un fallo del grafo llega al wire como un evento `error` terminal, con 200
    (el error va DENTRO del stream SSE, no como status HTTP)."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(chat_mod, "graph", _FakeStreamGraph("error"))
    from app.server import app

    client = TestClient(app)
    r = client.post("/chat/stream", json={"message": "hola", "thread_id": "s3"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(r.text)
    assert events[-1][0] == "error"
