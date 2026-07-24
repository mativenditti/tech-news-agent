"""Tests de la lógica de tools que no requiere red: aprobación del email, render
HTML, envío SMTP (mockeado) y RAG."""

from __future__ import annotations

import pytest

from app.email_render import markdown_to_html
from app.tools import _is_approved, _send_via_smtp


def test_is_approved_variants():
    assert _is_approved(True) is True
    assert _is_approved("approve") is True
    assert _is_approved("sí") is True
    assert _is_approved("ok") is True
    assert _is_approved({"approve": True}) is True
    assert _is_approved({"decision": "yes"}) is True

    assert _is_approved(False) is False
    assert _is_approved("reject") is False
    assert _is_approved("no") is False
    assert _is_approved({"approve": False}) is False
    assert _is_approved(None) is False


def test_markdown_to_html():
    html = markdown_to_html(
        "# Titulares\n\n"
        "- OpenAI lanzó [GPT-X](https://openai.com/x)\n"
        "- Escape de <script>alert(1)</script> y **negrita**"
    )
    # Encabezado y lista.
    assert "<h1>Titulares</h1>" in html
    assert "<li>" in html
    # Link markdown → <a href>.
    assert '<a href="https://openai.com/x">GPT-X</a>' in html
    # Bold.
    assert "<strong>negrita</strong>" in html
    # El contenido peligroso queda escapado (no hay un <script> ejecutable).
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_markdown_to_html_passthrough_si_ya_es_html():
    """Si el LLM manda un documento HTML completo, no se re-escapa: se usa tal cual.

    Regresión: antes el body HTML terminaba con las etiquetas escapadas
    (&lt;h1&gt;) y llegaba como texto crudo al cliente de correo.
    """
    body = (
        "<html><head><style>.h{color:red}</style></head>"
        "<body><h1>Tech Briefing</h1><p>Hola</p></body></html>"
    )
    html = markdown_to_html(body)
    # Las etiquetas reales deben quedar como HTML, NO escapadas.
    assert "<h1>Tech Briefing</h1>" in html
    assert "&lt;h1&gt;" not in html


def test_send_via_smtp_calls_smtp(monkeypatch):
    """Con SMTP configurado, _send_via_smtp abre la conexión, hace login y manda."""
    import app.tools as tools
    from app.config import settings

    calls: dict[str, object] = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            calls["host"] = host
            calls["port"] = port

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starttls(self):
            calls["starttls"] = True

        def login(self, user, password):
            calls["login"] = (user, password)

        def send_message(self, msg):
            calls["msg"] = msg

    monkeypatch.setattr(tools.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(settings, "email_smtp_host", "smtp.example.com")
    monkeypatch.setattr(settings, "email_smtp_port", 587)
    monkeypatch.setattr(settings, "email_smtp_user", "user@example.com")
    monkeypatch.setattr(settings, "email_smtp_password", "secret")
    monkeypatch.setattr(settings, "email_smtp_use_tls", True)

    tools._send_via_smtp("dest@example.com", "Asunto", "# Hola\n\ncuerpo")

    assert calls["host"] == "smtp.example.com"
    assert calls["starttls"] is True
    assert calls["login"] == ("user@example.com", "secret")
    msg = calls["msg"]
    assert msg["To"] == "dest@example.com"
    assert msg["Subject"] == "Asunto"
    # multipart/alternative: texto plano + HTML.
    assert msg.is_multipart()
    subtypes = {part.get_content_subtype() for part in msg.iter_parts()}
    assert {"plain", "html"} <= subtypes


def test_send_via_smtp_sin_host_falla(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "email_smtp_host", "")
    with pytest.raises(RuntimeError):
        _send_via_smtp("dest@example.com", "Asunto", "cuerpo")


def test_rag_roundtrip(monkeypatch, tmp_path):
    """Ingerir un artículo y recuperarlo del vector store (embeddings fake)."""
    # Aislar el vector store a un dir temporal y limpiar el cache del factory.
    import app.rag as rag
    from app.config import settings

    settings.embeddings_provider = "fake"
    settings.chroma_dir = str(tmp_path / "chroma")
    rag.get_vectorstore.cache_clear()

    n = rag.ingest_articles(
        [
            {
                "title": "Meta lanza nuevo modelo",
                "url": "https://example.com/meta",
                "content": "Meta presentó un nuevo modelo open source para desarrolladores.",
            }
        ]
    )
    assert n >= 1

    docs = rag.rag_retrieve("modelo de Meta", k=2)
    assert len(docs) >= 1
    assert any("Meta" in d.metadata.get("title", "") for d in docs)
