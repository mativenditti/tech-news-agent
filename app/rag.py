"""RAG sobre el corpus propio de noticias.

Cuando `web_search` trae artículos, se ingieren acá (chunk + embed + pgvector).
Luego `rag_search` puede recuperarlos para profundizar/citar sin volver a la web.

Embeddings: por defecto `google` (text-embedding-004, reales). El provider `fake`
(deterministas, sin costo ni red) queda disponible para tests offline; se elige en .env.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

_COLLECTION = "tech_news"


def _chunk_id(url: str, chunk_index: int) -> str:
    """Id determinista para un chunk, derivado de (url, índice).

    Permite upsert: re-ingerir el mismo artículo sobrescribe en vez de duplicar.
    """
    raw = f"{url}::{chunk_index}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _get_embeddings() -> Embeddings:
    if settings.embeddings_provider == "google":
        # Import perezoso: sólo si realmente se usa. Reusa GOOGLE_API_KEY.
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=settings.google_api_key or None,
        )
    # Default PoC: embeddings deterministas locales.
    from langchain_community.embeddings import FakeEmbeddings

    return FakeEmbeddings(size=256)


@lru_cache
def get_vectorstore() -> PGVector:
    """Vector store sobre Postgres + pgvector (settings.database_url)."""
    return PGVector(
        embeddings=_get_embeddings(),
        collection_name=_COLLECTION,
        connection=settings.database_url,
        use_jsonb=True,
        create_extension=True,
    )


_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def ingest_articles(articles: list[dict]) -> int:
    """Indexa artículos (dicts con keys: title, url, content) en el vector store.

    Devuelve la cantidad de chunks agregados. Idempotencia básica: usa la url como
    parte del id del documento para no duplicar el mismo artículo.
    """
    docs: list[Document] = []
    ids: list[str] = []
    for art in articles:
        content = (art.get("content") or "").strip()
        if not content:
            continue
        title = art.get("title") or ""
        url = art.get("url") or ""
        full = f"{title}\n\n{content}"
        for i, chunk in enumerate(_splitter.split_text(full)):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"title": title, "url": url, "chunk": i},
                )
            )
            ids.append(_chunk_id(url, i))
    if not docs:
        return 0
    get_vectorstore().add_documents(docs, ids=ids)
    return len(docs)


def rag_retrieve(query: str, k: int = 4) -> list[Document]:
    """Recupera los k chunks más relevantes del corpus ingerido."""
    return get_vectorstore().similarity_search(query, k=k)
