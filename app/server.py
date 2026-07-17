"""Servidor LangServe (FastAPI).

Expone:
- POST /chat        -> API de chat para el front React (JSON simple)
- POST /chat/resume -> reanuda el HITL del email (approve/reject)
- /agent/*          -> el grafo como Runnable (invoke, stream, playground)
- POST /briefing    -> el briefing proactivo
- GET  /health      -> estado + si LangSmith tracing está activo
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.briefing import run_briefing, stream_briefing
from app.chat import resume_chat, send_chat, stream_chat
from app.config import settings
from app.graph import graph

app = FastAPI(
    title="Tech News Agent",
    version="0.1.0",
    description="Agente LangChain/LangGraph de noticias tech con RAG, guardrails y HITL.",
)

# CORS: permite que el front React (otro origen) consuma la API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Expone el grafo compilado como endpoints /agent/invoke, /agent/stream, /agent/playground.
# LangServe serializa el estado del grafo (messages, etc.) automáticamente.
add_routes(app, graph, path="/agent")


# --- API de chat (para el front React) ----------------------------------------


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None
    user_role: str | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    decision: str  # "approve" | "reject"


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    """Un turno de chat. Devuelve {type: message|approval|error, ...}."""
    return send_chat(req.message, thread_id=req.thread_id, user_role=req.user_role)


@app.post("/chat/resume")
def chat_resume(req: ResumeRequest) -> dict:
    """Reanuda el HITL del email con la decisión del usuario (approve|reject)."""
    return resume_chat(req.thread_id, req.decision)


class BriefingRequest(BaseModel):
    thread_id: str | None = None
    user_role: str | None = None


@app.post("/briefing")
def briefing(req: BriefingRequest) -> dict:
    """Dispara el briefing proactivo (top 3 titulares + follow-up)."""
    return run_briefing(thread_id=req.thread_id, user_role=req.user_role)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> EventSourceResponse:
    """Igual que /chat pero streameando la respuesta token a token (SSE)."""
    return EventSourceResponse(
        stream_chat(req.message, thread_id=req.thread_id, user_role=req.user_role)
    )


@app.post("/briefing/stream")
def briefing_stream(req: BriefingRequest) -> EventSourceResponse:
    """Igual que /briefing pero streameando el briefing token a token (SSE)."""
    return EventSourceResponse(
        stream_briefing(thread_id=req.thread_id, user_role=req.user_role)
    )


@app.get("/health")
def health() -> dict:
    # El tracing está "activo" sólo si tracing=true Y hay api key. No exponemos la key.
    tracing_active = settings.langsmith_tracing and bool(settings.langsmith_api_key)
    return {
        "status": "ok",
        "model": settings.llm_model,
        "langsmith_tracing": settings.langsmith_tracing,
        "langsmith_api_key_set": bool(settings.langsmith_api_key),
        "langsmith_active": tracing_active,
        "langsmith_project": settings.langsmith_project,
        "email_dry_run": settings.email_dry_run,
    }
