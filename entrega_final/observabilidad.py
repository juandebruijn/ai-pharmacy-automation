"""C — Observabilidad: logs JSON y métricas operativas.

    Logs     una línea JSON por evento, a consola y a entrega_final/logs/api.jsonl
    Métricas contadores en memoria que expone GET /metricas
    Trazas   LangSmith (se activan solas con LANGSMITH_TRACING=true en el .env;
             el análisis de costos está en costos_langsmith.py)

Un log JSON se puede filtrar con jq, cargar en pandas o mandar a cualquier
agregador sin parsear texto libre: cada campo es una columna.
"""

import json
import logging
import math
import os
import sys
import threading
from collections import Counter
from datetime import datetime, timezone

from entrega_final.config import DIR_LOGS

# --------------------------------------------------------------------------- #
# Costo de referencia (USD por millón de tokens)
# --------------------------------------------------------------------------- #
# La capa gratuita de Gemini no cobra. Para estimar el costo en producción se usa
# la tarifa de la capa paga de gemini-3.1-flash-lite (US$ 0,25 entrada / 1,50 salida).
# Se pueden pisar desde el .env si el proveedor cambia la tarifa.
PRECIO_ENTRADA_USD_MTOK = float(os.getenv("PRECIO_ENTRADA_USD_MTOK", "0.25"))
PRECIO_SALIDA_USD_MTOK = float(os.getenv("PRECIO_SALIDA_USD_MTOK", "1.50"))


def costo_usd(tokens_entrada: int, tokens_salida: int) -> float:
    return (tokens_entrada * PRECIO_ENTRADA_USD_MTOK + tokens_salida * PRECIO_SALIDA_USD_MTOK) / 1e6


# --------------------------------------------------------------------------- #
# Logs JSON
# --------------------------------------------------------------------------- #
class FormateadorJSON(logging.Formatter):
    """Cada registro sale como un objeto JSON en una sola línea (formato JSON Lines)."""

    def format(self, record: logging.LogRecord) -> str:
        evento = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "nivel": record.levelname,
            "logger": record.name,
            "evento": record.getMessage(),
        }
        # Los campos estructurados viajan en extra={"datos": {...}}.
        evento.update(getattr(record, "datos", {}))
        if record.exc_info:
            evento["excepcion"] = self.formatException(record.exc_info)
        return json.dumps(evento, ensure_ascii=False, default=str)


def configurar_logger(nombre: str = "farmacia.api", archivo: bool = True) -> logging.Logger:
    logger = logging.getLogger(nombre)
    if logger.handlers:  # ya configurado (uvicorn --reload importa el módulo dos veces)
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    consola = logging.StreamHandler(sys.stdout)
    consola.setFormatter(FormateadorJSON())
    logger.addHandler(consola)
    if archivo:
        DIR_LOGS.mkdir(parents=True, exist_ok=True)
        a_disco = logging.FileHandler(DIR_LOGS / "api.jsonl", encoding="utf-8")
        a_disco.setFormatter(FormateadorJSON())
        logger.addHandler(a_disco)
    return logger


def log(logger: logging.Logger, evento: str, nivel: int = logging.INFO, **datos) -> None:
    logger.log(nivel, evento, extra={"datos": datos})


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def percentil(valores: list[float], p: float) -> float:
    """Percentil por el método nearest-rank: el valor que deja al p % de las muestras por debajo."""
    if not valores:
        return 0.0
    ordenados = sorted(valores)
    rango = max(1, math.ceil(p / 100 * len(ordenados)))
    return ordenados[rango - 1]


class Metricas:
    """
    Contadores del proceso. Un Lock porque FastAPI atiende los endpoints sync en
    un threadpool: dos requests simultáneas podrían pisar el mismo contador.

    Se reinician con la API (son métricas operativas de la instancia, como las de
    un /metrics de Prometheus). Lo histórico queda en el log JSON y en LangSmith.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.desde = datetime.now(timezone.utc)
        self.consultas = 0
        self.errores = 0
        self.latencias_ms: list[float] = []
        self.herramientas: Counter = Counter()
        self.bloqueos: Counter = Counter()          # validador -> cantidad
        self.enmascarados: Counter = Counter()      # tipo de dato -> cantidad
        self.limite_iteraciones = 0
        self.tokens_entrada = 0
        self.tokens_salida = 0

    def registrar_consulta(self, latencia_ms: float, herramientas: list[str],
                           tokens: dict | None = None, limite: bool = False) -> None:
        with self._lock:
            self.consultas += 1
            self.latencias_ms.append(latencia_ms)
            self.herramientas.update(herramientas)
            self.limite_iteraciones += int(limite)
            self.tokens_entrada += (tokens or {}).get("entrada", 0)
            self.tokens_salida += (tokens or {}).get("salida", 0)

    def registrar_error(self, latencia_ms: float) -> None:
        with self._lock:
            self.consultas += 1
            self.errores += 1
            self.latencias_ms.append(latencia_ms)

    def registrar_bloqueo(self, validador: str, latencia_ms: float = 0.0) -> None:
        with self._lock:
            self.consultas += 1
            self.bloqueos[validador] += 1
            # La latencia de un bloqueo (~0 ms, no hay LLM) no entra al promedio ni
            # al p95: los bajaría artificialmente y escondería una regresión del agente.

    def registrar_enmascarado(self, reemplazos: dict[str, int]) -> None:
        with self._lock:
            self.enmascarados.update(reemplazos)

    def resumen(self) -> dict:
        with self._lock:
            lat = list(self.latencias_ms)
            respondidas = self.consultas - self.errores - sum(self.bloqueos.values())
            return {
                "desde": self.desde.isoformat(timespec="seconds"),
                "consultas": {
                    "total": self.consultas,
                    "respondidas": respondidas,
                    "errores": self.errores,
                    "bloqueadas": sum(self.bloqueos.values()),
                    "tasa_error": round(self.errores / self.consultas, 4) if self.consultas else 0.0,
                },
                "latencia_ms": {
                    "promedio": round(sum(lat) / len(lat), 1) if lat else 0.0,
                    "p95": round(percentil(lat, 95), 1),
                    "max": round(max(lat), 1) if lat else 0.0,
                },
                "herramientas": dict(self.herramientas.most_common()),
                "bloqueos": {
                    "entrada": dict(self.bloqueos),
                    "salida_enmascarados": dict(self.enmascarados),
                    "limite_iteraciones": self.limite_iteraciones,
                },
                "tokens": {
                    "entrada": self.tokens_entrada,
                    "salida": self.tokens_salida,
                    "costo_estimado_usd": round(costo_usd(self.tokens_entrada, self.tokens_salida), 6),
                    "promedio_por_consulta": round(
                        (self.tokens_entrada + self.tokens_salida) / respondidas, 1) if respondidas else 0.0,
                },
            }
