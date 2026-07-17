# tech-news-agent

Agente de AI (LangChain + LangGraph) que te mantiene al día con las últimas noticias tech y arma reportes legibles. Monitoreo con LangSmith, expuesto por HTTP con LangServe.

## Qué hace

- **Briefing proactivo:** un "top 3 titulares más picantes de la semana" con una pregunta de follow-up para elegir por dónde profundizar.
- **Ida y vuelta on-demand con adaptación de rol:** "explicame el hackeo como si fuera hincha de Boca" o "¿qué impacto tiene el modelo de Meta en mi laburo de dev?" — el agente ajusta el tono al rol, sin cambiar los hechos.
- **Tareas al vuelo (tools):** "buscame un repo de GitHub para probar lo de Meta" → activa la búsqueda web y trae el link en la misma conversación.
- **RAG + skills:** las noticias que trae se indexan en un vector store; después las puede recuperar y citar.
- **Guardrails:** anti prompt-injection (input y contenido web), scope/on-topic, y rate limiting de tools.
- **Human-in-the-loop (HITL):** antes de "enviar" un reporte por email, el grafo se pausa y espera confirmación humana.

## Stack

- **LangChain + LangGraph** — orquestación (grafo stateful con tools, memoria e interrupts).
- **langchain-google-genai** — LLM Gemini (`gemini-3-flash-preview`).
- **Tavily** — búsqueda web (`langchain-tavily`).
- **Chroma** — vector store local para RAG.
- **LangSmith** — tracing/monitoreo (sólo variables de entorno, sin código).
- **LangServe + FastAPI** — exposición HTTP.

## Arquitectura

Un único grafo LangGraph, compilado como Runnable y expuesto por LangServe:

```
entrada → guardrails_in → agent (LLM+tools) ⇄ tools → END
                              │
                              └─ send_email_report → interrupt (HITL) → dry-run
```

- `guardrails_in`: corta antes del LLM si detecta prompt-injection o pedido fuera de scope.
- `agent`: Gemini con tools bindeadas y system prompt rol-adaptativo.
- `tools`: ejecuta las tools con rate limiting por conversación. La de email dispara `interrupt()` (HITL).
- Memoria de conversación vía `MemorySaver` (checkpointer) por `thread_id`.

## Setup

Requisitos: Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Editá .env con tus claves (GOOGLE_API_KEY, TAVILY_API_KEY, LANGSMITH_*)
```

Variables clave (ver `.env.example`):

- `GOOGLE_API_KEY` — requerida para el LLM (Google AI Studio).
- `TAVILY_API_KEY` — requerida para la búsqueda web.
- `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` — habilita el tracing (opcional pero recomendado).
- `EMAIL_DRY_RUN=true` — el email no se envía de verdad, sólo se loguea.
- `EMBEDDINGS_PROVIDER=fake` — embeddings locales para el PoC (sin costo ni deps extra).

## Correr

```bash
uvicorn app.server:app --reload
```

Endpoints:

- `GET  /health` — estado + si el tracing está activo.
- `POST /chat` — un turno de chat (JSON). Body: `{"message": "...", "thread_id": null, "user_role": "dev backend"}`.
- `POST /chat/resume` — reanuda el HITL del email. Body: `{"thread_id": "...", "decision": "approve"|"reject"}`.
- `POST /chat/stream` — igual que `/chat`, pero streamea la respuesta token a token (SSE). Ver [Streaming (SSE)](#streaming-sse).
- `POST /briefing` — dispara el briefing proactivo. Body: `{"thread_id": null, "user_role": "dev backend"}`.
- `POST /briefing/stream` — igual que `/briefing`, pero streameando (SSE).
- `/agent/invoke`, `/agent/stream`, `/agent/playground` — el grafo expuesto por LangServe.

## Human-in-the-loop (email)

Cuando el agente decide llamar a `send_email_report`, el grafo se pausa: la respuesta incluye un `__interrupt__` con la propuesta de reporte. Para reanudar, se reinvoca en el **mismo `thread_id`** con un `Command(resume=...)`:

- `resume="approve"` → "envía" el reporte (en dry-run, lo loguea).
- `resume="reject"` → cancela, no lo manda.

## Streaming (SSE)

`POST /chat/stream` y `POST /briefing/stream` devuelven la respuesta como un stream **Server-Sent Events** (`Content-Type: text/event-stream`), para que el front la muestre token a token en vez de esperar a que termine todo el ida y vuelta con el LLM.

Cada evento tiene un `event:` (tipo) y un `data:` (JSON):

| `event`    | `data`                                                        | Cuándo |
|------------|---------------------------------------------------------------|--------|
| `token`    | `{"text": "<fragmento>"}`                                     | Un fragmento de la respuesta. El front lo **appendea** al mensaje. |
| `approval` | `{"prompt","action","to","subject","body","thread_id"}`       | El grafo se pausó por el HITL del email. Cierra el stream; seguí con `POST /chat/resume`. |
| `done`     | `{"thread_id": "..."}`                                        | Terminó OK. Cierra el stream. |
| `error`    | `{"text": "<mensaje amable>", "thread_id": "..."}`           | Falló algo (cuota, red). El status HTTP sigue siendo 200: el error viaja **dentro** del stream. |

Consumo desde el front (fetch + lectura del stream):

```js
// POST /chat/stream -> text/event-stream
let text = "";
for await (const evt of readSSE(response)) {
  if (evt.event === "token")    text += JSON.parse(evt.data).text;   // append en la UI
  if (evt.event === "approval") showApprovalButtons(JSON.parse(evt.data));
  if (evt.event === "done")     finalize(JSON.parse(evt.data).thread_id);
  if (evt.event === "error")    showError(JSON.parse(evt.data).text);
}
```

## Tests

```bash
pytest
```

Los tests corren offline (LLM y tools falsos): cubren los guardrails, la lógica de aprobación del email, el roundtrip de RAG, el ciclo interrupt/resume del HITL y el streaming SSE (helpers, generadoras y endpoints). No necesitan claves de API.

## Notas de producción (fuera de scope del PoC)

- **Memoria persistente:** cambiar `MemorySaver` por `SqliteSaver`/`PostgresSaver`.
- **Scheduling del briefing:** el `POST /briefing` es manual; en producción iría detrás de un cron.
- **Embeddings reales:** setear `EMBEDDINGS_PROVIDER=google` (reusa `GOOGLE_API_KEY`).
- **Email real:** implementar el envío SMTP en `send_email_report` y poner `EMAIL_DRY_RUN=false`.
