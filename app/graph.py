"""Grafo LangGraph del agente.

Flujo:
    guardrails_in → agent → (tools ↔ agent)* → END

- guardrails_in: prompt-injection + scope sobre el último mensaje del usuario.
  Si falla, escribe una respuesta de rechazo y rutea a END (no llega al LLM).
- agent: LLM (Gemini) con tools bindeadas + system prompt rol-adaptativo.
- tools: ToolNode que ejecuta las tools. La de email dispara interrupt() adentro
  (HITL). El rate limiting se chequea en el ruteo antes de ejecutar.

Se compila con un checkpointer (MemorySaver) para tener memoria de conversación por
thread_id y para poder pausar/reanudar en el interrupt del HITL.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.guardrails import (
    INJECTION_REFUSAL,
    SCOPE_REFUSAL,
    check_rate_limit,
    is_in_scope,
    is_prompt_injection,
)
from app.llm import get_llm
from app.prompts import build_system_prompt
from app.state import AgentState
from app.tools import ALL_TOOLS

_TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


def _last_human_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


# --- Nodos ---------------------------------------------------------------------


def guardrails_in(state: AgentState) -> dict:
    """Guardrail de entrada: injection + scope sobre el último turno del usuario."""
    text = _last_human_text(state["messages"])

    if is_prompt_injection(text):
        return {"messages": [AIMessage(content=INJECTION_REFUSAL)], "blocked": True}

    if not is_in_scope(text):
        return {"messages": [AIMessage(content=SCOPE_REFUSAL)], "blocked": True}

    return {"blocked": False}


def agent(state: AgentState) -> dict:
    """Nodo LLM: arma el system prompt rol-adaptativo y responde (con tools)."""
    llm = get_llm().bind_tools(ALL_TOOLS)
    system = SystemMessage(content=build_system_prompt(state.get("user_role")))
    response = llm.invoke([system, *state["messages"]])
    return {"messages": [response]}


# ToolNode ejecuta las tool_calls del último AIMessage y agrega los ToolMessage.
_tool_node = ToolNode(ALL_TOOLS)


def tools_node(state: AgentState) -> dict:
    """Ejecuta tools, aplicando rate limiting por conversación.

    Para cada tool_call: si superó el tope, devolvemos un ToolMessage de error en
    vez de ejecutarla (el LLM lo ve y explica al usuario). Si pasa, ejecutamos vía
    ToolNode y actualizamos el contador.
    """
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", []) or []
    counts = dict(state.get("tool_call_counts") or {})

    allowed_calls = []
    blocked_messages = []
    for call in tool_calls:
        name = call["name"]
        if check_rate_limit(counts, name):
            allowed_calls.append(call)
            counts[name] = counts.get(name, 0) + 1
        else:
            blocked_messages.append(
                ToolMessage(
                    content=(
                        f"Límite de llamadas a `{name}` alcanzado en esta "
                        "conversación. No la ejecuté."
                    ),
                    tool_call_id=call["id"],
                    name=name,
                )
            )

    new_messages = list(blocked_messages)
    if allowed_calls:
        if len(allowed_calls) == len(tool_calls):
            # Caso normal (nada bloqueado): pasamos el AIMessage ORIGINAL intacto.
            # Reconstruirlo perdería additional_kwargs/response_metadata, donde
            # Gemini guarda el thought_signature que exige al devolver el
            # tool_result (si no, tira 400 "missing thought_signature").
            ai_msg = last
        else:
            # Se bloqueó alguna call por rate limit: hay que filtrar tool_calls.
            # Copiamos el mensaje original preservando su metadata (model_copy)
            # en vez de crear un AIMessage nuevo desde cero.
            ai_msg = last.model_copy(update={"tool_calls": allowed_calls})
        result = _tool_node.invoke({"messages": [ai_msg]})
        new_messages.extend(result["messages"])

    return {"messages": new_messages, "tool_call_counts": counts}


# --- Ruteo ---------------------------------------------------------------------


def route_after_guardrails(state: AgentState) -> str:
    return END if state.get("blocked") else "agent"


def route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


# --- Construcción --------------------------------------------------------------


def build_graph(checkpointer=None):
    """Arma y compila el grafo. Si no se pasa checkpointer, usa MemorySaver."""
    workflow = StateGraph(AgentState)
    workflow.add_node("guardrails_in", guardrails_in)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tools_node)

    workflow.set_entry_point("guardrails_in")
    workflow.add_conditional_edges(
        "guardrails_in", route_after_guardrails, {"agent": "agent", END: END}
    )
    workflow.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", END: END}
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=checkpointer or MemorySaver())


# Instancia por defecto (usada por el server de LangServe).
graph = build_graph()
