"""Tests de la lógica de tools que no requiere red: aprobación del email y RAG."""

from __future__ import annotations

from app.tools import _is_approved


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
