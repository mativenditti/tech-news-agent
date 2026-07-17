"""Tools del agente, declaradas con @tool de LangChain.

- web_search: Tavily. El resultado pasa por sanitize_web_content antes de volver
  al LLM, y los artículos se ingieren en el vector store (cierra el loop RAG).
- rag_search: recupera del corpus propio de noticias ingeridas.
- send_email_report: dry-run. Gateada por HITL: llama a interrupt() y espera la
  confirmación humana antes de "enviar".
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool
from langgraph.types import interrupt

from app.config import settings
from app.guardrails import sanitize_web_content
from app.rag import ingest_articles, rag_retrieve

logger = logging.getLogger("tech_news_agent.tools")


@tool
def web_search(query: str) -> str:
    """Busca en la web noticias tech y detalles actuales.

    Usala para traer titulares recientes o para encontrar algo concreto pedido por
    el usuario (por ejemplo, el repositorio de GitHub de un proyecto). Devuelve un
    resumen de los resultados con sus fuentes (título + link).
    """
    from langchain_tavily import TavilySearch

    search = TavilySearch(
        max_results=5, tavily_api_key=settings.tavily_api_key or None
    )
    raw = search.invoke({"query": query})

    # TavilySearch devuelve {"results": [{title, url, content, ...}], ...}
    results = raw.get("results", []) if isinstance(raw, dict) else []

    # Cierra el loop RAG: ingerimos lo que trajimos para poder recuperarlo luego.
    try:
        ingest_articles(results)
    except Exception:  # el RAG no debe romper la búsqueda
        logger.exception("Fallo al ingerir artículos en el vector store")

    if not results:
        return "No encontré resultados para esa búsqueda."

    lines = []
    for r in results:
        title = r.get("title", "(sin título)")
        url = r.get("url", "")
        snippet = sanitize_web_content(r.get("content", ""), max_chars=500)
        lines.append(f"- {title}\n  {url}\n  {snippet}")
    return "Resultados de la web:\n" + "\n".join(lines)


@tool
def rag_search(query: str) -> str:
    """Recupera noticias/artículos ya ingeridos en conversaciones anteriores.

    Usala para profundizar o citar fuentes previas sin volver a buscar en la web.
    """
    docs = rag_retrieve(query, k=4)
    if not docs:
        return "No tengo artículos previos indexados sobre eso."
    lines = []
    for d in docs:
        title = d.metadata.get("title", "")
        url = d.metadata.get("url", "")
        lines.append(f"- {title} ({url})\n  {d.page_content[:400]}")
    return "Notas previas relevantes:\n" + "\n".join(lines)


@tool
def send_email_report(subject: str, body: str, to: str | None = None) -> str:
    """Envía por email un reporte resumido de noticias tech.

    ANTES de enviar, un humano debe confirmar. Esta tool pausa la ejecución
    (human-in-the-loop) y presenta la propuesta de reporte; sólo si se aprueba se
    "envía" (en modo dry-run: se loguea, no se manda de verdad).
    """
    recipient = to or settings.email_to

    # HITL: pausa el grafo y espera la decisión humana (approve / reject).
    decision = interrupt(
        {
            "action": "send_email_report",
            "to": recipient,
            "subject": subject,
            "body": body,
            "message": (
                f"¿Confirmás el envío de este reporte a {recipient}?\n\n"
                f"Asunto: {subject}\n\n{body}"
            ),
        }
    )

    # `decision` es lo que el cliente pasa en Command(resume=...).
    approved = _is_approved(decision)
    if not approved:
        return "Envío cancelado por el usuario. El reporte no se mandó."

    if settings.email_dry_run:
        logger.info(
            "[EMAIL DRY-RUN] From=%s To=%s Subject=%s\n%s",
            settings.email_from,
            recipient,
            subject,
            body,
        )
        return (
            f"(dry-run) Reporte listo para {recipient} con asunto \"{subject}\". "
            "No se envió de verdad porque EMAIL_DRY_RUN está activo."
        )

    # Punto de extensión: acá iría el SMTP real (fuera de scope del PoC).
    raise NotImplementedError("Envío SMTP real no implementado (usar EMAIL_DRY_RUN).")


def _is_approved(decision: object) -> bool:
    """Interpreta la respuesta del humano de forma tolerante."""
    if isinstance(decision, bool):
        return decision
    if isinstance(decision, dict):
        val = decision.get("approve", decision.get("decision"))
        if isinstance(val, bool):
            return val
        decision = val
    if isinstance(decision, str):
        return decision.strip().lower() in {"approve", "yes", "y", "si", "sí", "ok", "true"}
    return False


# Tools que el LLM puede invocar libremente vs. la que está gateada por HITL.
SAFE_TOOLS = [web_search, rag_search]
HITL_TOOLS = [send_email_report]
ALL_TOOLS = SAFE_TOOLS + HITL_TOOLS
