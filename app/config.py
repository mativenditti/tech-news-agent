"""Configuración central del agente, leída desde variables de entorno / .env.

Un único objeto `settings` (patrón pydantic-settings) que el resto de los módulos
importa. Así evitamos leer os.environ desperdigado por todos lados.
"""

from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Cargamos el .env al entorno del proceso ANTES de instanciar Settings.
# Clave para LangSmith: langchain lee LANGSMITH_TRACING / LANGSMITH_API_KEY /
# LANGSMITH_PROJECT desde os.environ directamente, no desde nuestro objeto Settings.
# Sin esto, el tracing quedaría apagado aunque la key esté en el .env.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM ---
    google_api_key: str = ""
    # Gemini 3 flash. Los modelos 2.x están retirados para cuentas nuevas (404).
    # Gemini 3 son modelos "thinking": emiten un thought_signature en cada
    # functionCall que la API exige devolver idéntico en el turno siguiente. Requiere
    # langchain-google-genai>=3.1.0 (tenemos 4.x) para que se serialice bien en
    # tool-calling multi-turno; con <3 se dispara "400 missing thought_signature".
    llm_model: str = "gemini-3-flash-preview"
    # Tope de tokens de razonamiento interno ("thinking") por respuesta. Gemini 3
    # razona antes de responder, aun en respuestas triviales ("de nada" gastaba
    # ~416 tokens de reasoning → ~15s). Un tope bajo acelera mucho lo trivial y deja
    # margen para decisiones (elegir tools). 0 lo desactiva; None deja el default
    # dinámico del modelo (lento en lo trivial). Este es el budget del chat.
    llm_thinking_budget: int = 512
    # Budget para el briefing, que sí necesita razonar (elegir el "top 3", priorizar
    # impacto). Más alto que el de chat a cambio de más latencia, aceptable porque el
    # briefing es proactivo/menos sensible a la espera.
    llm_thinking_budget_briefing: int = 2048

    # --- Web search ---
    tavily_api_key: str = ""

    # --- LangSmith (tracing) ---
    # langchain lee estas env vars directamente; las declaramos para documentarlas
    # y para poder chequear el estado desde /health. El load_dotenv() de arriba
    # asegura que estén en os.environ para que langchain las vea.
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "tech-news-agent"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # --- Email ---
    # En dry-run (default seguro) la tool no manda nada: sólo loguea el reporte.
    email_dry_run: bool = True
    email_from: str = "tech-news-agent@example.com"
    email_to: str = "me@example.com"
    # SMTP: sólo se usa cuando email_dry_run=False. Sirve para Gmail (App Password),
    # SendGrid/Mailgun SMTP, o cualquier servidor SMTP. use_tls = STARTTLS (puerto 587).
    email_smtp_host: str = ""
    email_smtp_port: int = 587
    email_smtp_user: str = ""
    email_smtp_password: str = ""
    email_smtp_use_tls: bool = True

    # --- RAG / embeddings ---
    # google -> embeddings reales (text-embedding-004, 768 dims, reusa GOOGLE_API_KEY)
    # fake   -> embeddings deterministas locales (para tests offline)
    embeddings_provider: str = "google"  # "google" | "fake"
    # Postgres + pgvector. Formato SQLAlchemy async-agnostic con driver psycopg.
    database_url: str = "postgresql+psycopg://tech:tech@localhost:5432/tech_news"

    # --- Rate limits (por conversación) ---
    max_web_search_calls: int = 5
    max_email_calls: int = 2

    # --- CORS (para el front React) ---
    # Lista separada por comas de orígenes permitidos, ej:
    # "http://localhost:5173,https://mi-front.vercel.app". "*" permite todos (solo dev).
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Settings cacheados (se instancian una sola vez por proceso)."""
    return Settings()


settings = get_settings()
