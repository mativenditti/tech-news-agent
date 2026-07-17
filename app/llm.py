"""Factory del LLM. Vía LangChain-nativa para Gemini (langchain-google-genai),
no el SDK crudo. Default: gemini-flash-latest (alias que apunta a la flash vigente).
"""

from __future__ import annotations

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings


@lru_cache
def get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """Devuelve un ChatGoogleGenerativeAI reusable.

    temperature 0 por defecto para respuestas estables; el nodo de briefing usa
    un valor más alto para que los titulares no sean monótonos.
    """
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=temperature,
        max_output_tokens=4096,
        timeout=60,
        google_api_key=settings.google_api_key or None,
    )
