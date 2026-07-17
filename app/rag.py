"""RAG sobre el corpus propio de noticias.

Cuando `web_search` trae artículos, se ingieren acá (chunk + embed + Chroma).
Luego `rag_search` puede recuperarlos para profundizar/citar sin volver a la web.

Embeddings: por defecto `fake` (deterministas, sin costo ni deps externas) para
que el PoC corra out-of-the-box. Cambiar a `google` en .env para embeddings reales.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

_COLLECTION = "tech_news"


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
def get_vectorstore() -> Chroma:
    """Vector store Chroma persistido en disco (settings.chroma_dir)."""
    return Chroma(
        collection_name=_COLLECTION,
        embedding_function=_get_embeddings(),
        persist_directory=settings.chroma_dir,
    )


_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def ingest_articles(articles: list[dict]) -> int:
    """Indexa artículos (dicts con keys: title, url, content) en el vector store.

    Devuelve la cantidad de chunks agregados. Idempotencia básica: usa la url como
    parte del id del documento para no duplicar el mismo artículo.
    """
    docs: list[Document] = []
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
    if not docs:
        return 0
    get_vectorstore().add_documents(docs)
    return len(docs)


def rag_retrieve(query: str, k: int = 4) -> list[Document]:
    """Recupera los k chunks más relevantes del corpus ingerido."""
    return get_vectorstore().similarity_search(query, k=k)
