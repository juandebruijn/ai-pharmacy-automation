"""B – C — API REST del agente: FastAPI + observabilidad.

    GET  /health               estado del servicio y de sus dependencias
    POST /consultar            una consulta al agente, con memoria por thread_id
    GET  /estado/{thread_id}   lo que la memoria SQLite tiene de una conversación
    GET  /metricas             consultas, errores, latencia, herramientas y bloqueos

Uso (desde la raíz del repo):
    uvicorn entrega_final.api:app --port 8000
    # documentación interactiva: http://localhost:8000/docs
"""

import time
import uuid
from contextlib import asynccontextmanager

import chromadb
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from entrega_final import erp
from entrega_final.agente import consultar, construir_agente, crear_checkpointer, historial
from entrega_final.config import DB_ERP, DB_MEMORIA, MAX_ITERACIONES, MODELO_AGENTE
from entrega_final.observabilidad import Metricas, configurar_logger, log

logger = configurar_logger()


# --------------------------------------------------------------------------- #
# Contratos
# --------------------------------------------------------------------------- #
class ConsultaRequest(BaseModel):
    pregunta: str = Field(min_length=1, examples=["¿Tienen Ibupirac 600 en la sucursal Centro?"])
    thread_id: str | None = Field(
        None, pattern=r"^[A-Za-z0-9_-]{1,64}$",
        description="Omitir en el primer mensaje: la API genera uno. Reenviarlo para seguir la conversación.",
        examples=["cliente-demo-01"],
    )


class ConsultaResponse(BaseModel):
    thread_id: str
    respuesta: str
    herramientas: list[str] = Field(description="Herramientas que usó el agente en este turno, en orden.")
    latencia_ms: float


class Mensaje(BaseModel):
    rol: str
    contenido: str


class EstadoResponse(BaseModel):
    thread_id: str
    total_mensajes: int
    mensajes: list[Mensaje]
    reservas: list[dict]


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def crear_app(grafo=None) -> FastAPI:
    """
    grafo=None construye el agente real con memoria SQLite. Los tests pueden pasar un
    grafo con un LLM falso para probar la API sin red ni cuota.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Lo costoso se construye una sola vez al arrancar, no en cada request.
        erp.inicializar()
        app.state.grafo = grafo or construir_agente(checkpointer=crear_checkpointer())
        app.state.metricas = Metricas()
        log(logger, "api_iniciada", modelo=MODELO_AGENTE, memoria=str(DB_MEMORIA),
            max_iteraciones=MAX_ITERACIONES)
        yield
        log(logger, "api_detenida")

    app = FastAPI(
        title="Asistente de farmacia — API del agente",
        version="1.0.0",
        description="Agente LangGraph sobre el RAG de la Entrega 3, con memoria persistente.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_de_requests(request: Request, call_next):
        """Una línea JSON por request HTTP, con un request_id para cruzarla con los demás eventos."""
        request.state.request_id = uuid.uuid4().hex[:12]
        inicio = time.perf_counter()
        respuesta = await call_next(request)
        log(logger, "http_request", request_id=request.state.request_id, metodo=request.method,
            ruta=request.url.path, status=respuesta.status_code,
            latencia_ms=round((time.perf_counter() - inicio) * 1000, 1))
        respuesta.headers["X-Request-ID"] = request.state.request_id
        return respuesta

    @app.get("/health", tags=["operación"])
    def health():
        """Liveness + chequeo de dependencias: base vectorial, ERP y memoria."""
        from entrega_3.rag_pipeline import COLECCION, DIR_CHROMA
        try:
            colecciones = {c.name: c.count() for c in
                           chromadb.PersistentClient(path=str(DIR_CHROMA)).list_collections()}
        except Exception as exc:  # la API puede vivir sin Chroma, pero no responde políticas
            colecciones = {"error": str(exc)}
        base_ok = COLECCION in colecciones
        return {
            "status": "ok" if base_ok else "degradado",
            "modelo": MODELO_AGENTE,
            "base_vectorial": colecciones,
            "erp": DB_ERP.exists(),
            "memoria": DB_MEMORIA.exists(),
        }

    @app.post("/consultar", response_model=ConsultaResponse, tags=["agente"])
    def consultar_agente(body: ConsultaRequest, request: Request):
        """Una consulta al agente ReAct. Reenviando el thread_id se sigue la misma conversación."""
        metricas: Metricas = request.app.state.metricas
        grafo = request.app.state.grafo
        thread_id = body.thread_id or uuid.uuid4().hex[:16]
        rid = getattr(request.state, "request_id", None)
        inicio = time.perf_counter()

        def ms() -> float:
            return round((time.perf_counter() - inicio) * 1000, 1)

        try:
            salida = consultar(grafo, body.pregunta, thread_id, {"request_id": rid, "canal": "api"})
        except Exception as exc:
            metricas.registrar_error(ms())
            log(logger, "consulta_error", 40, request_id=rid, thread_id=thread_id,
                error=type(exc).__name__, detalle=str(exc)[:300])
            raise HTTPException(status_code=502, detail="El agente no pudo responder. Probá de nuevo.") from exc

        latencia = ms()
        metricas.registrar_consulta(latencia, salida["herramientas"], salida["tokens"],
                                    salida["limite_alcanzado"])
        log(logger, "consulta_respondida", request_id=rid, thread_id=thread_id, latencia_ms=latencia,
            herramientas=salida["herramientas"], iteraciones=salida["iteraciones"],
            tokens=salida["tokens"],
            limite_alcanzado=salida["limite_alcanzado"])
        return ConsultaResponse(
            thread_id=thread_id, respuesta=salida["respuesta"], herramientas=salida["herramientas"],
            latencia_ms=latencia,
        )

    @app.get("/estado/{thread_id}", response_model=EstadoResponse, tags=["agente"])
    def estado(thread_id: str, request: Request):
        """Historial de la conversación (desde el checkpointer SQLite) y sus reservas en el ERP."""
        mensajes = historial(request.app.state.grafo, thread_id)
        if not mensajes:
            raise HTTPException(status_code=404, detail=f"No hay conversación para el thread_id {thread_id!r}.")
        return EstadoResponse(thread_id=thread_id, total_mensajes=len(mensajes),
                              mensajes=[Mensaje(**m) for m in mensajes],
                              reservas=erp.obtener_reservas(thread_id))

    @app.get("/metricas", tags=["operación"])
    def metricas(request: Request):
        """Métricas de la instancia desde que arrancó: volumen, errores, latencia, herramientas y bloqueos."""
        return request.app.state.metricas.resumen()

    return app


app = crear_app()
