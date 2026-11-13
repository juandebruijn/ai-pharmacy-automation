"""Configuración compartida por el agente, la API y los tests.

Todo lo que cambia entre máquinas sale del .env; todo lo demás es una constante
con nombre, para que el informe pueda citarla.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent
DIR_FINAL = RAIZ / "entrega_final"

# El .env se carga antes de importar LangChain: si trae LANGSMITH_TRACING=true,
# cada invocación del grafo queda registrada como traza.
load_dotenv(RAIZ / ".env")

# La consola de Windows usa cp1252 por defecto y rompe los acentos del dominio.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
# Mismo criterio que la Entrega 3: capa gratuita de Gemini, modelo lite porque
# cada consulta al agente son 2 o 3 llamadas (decidir tool -> observar -> responder).
MODELO_AGENTE = os.getenv("GEMINI_MODEL_AGENTE", "gemini-3.1-flash-lite")

# --------------------------------------------------------------------------- #
# Límites del grafo ReAct (A.2)
# --------------------------------------------------------------------------- #
# Vueltas agente -> herramientas permitidas por consulta. La 5.ª vez que el
# modelo pide una tool, el grafo corta y deriva a un humano.
MAX_ITERACIONES = int(os.getenv("MAX_ITERACIONES", "4"))
# Red de seguridad de LangGraph: super-steps totales por invocación. Cada vuelta
# son 2 (agente + herramientas); con 4 iteraciones alcanzan 2*4 + 2 = 10.
RECURSION_LIMIT = 2 * MAX_ITERACIONES + 4
# Ventana de mensajes que ve el LLM. El historial completo queda en el checkpointer.
VENTANA_MENSAJES = 12

# --------------------------------------------------------------------------- #
# Persistencia
# --------------------------------------------------------------------------- #
DIR_DATOS = Path(os.getenv("DIR_DATOS", DIR_FINAL / "data"))
DB_MEMORIA = Path(os.getenv("DB_MEMORIA", DIR_DATOS / "memoria_agente.sqlite"))
DB_ERP = Path(os.getenv("DB_ERP", DIR_DATOS / "erp_farmacia.sqlite"))
CATALOGO_JSON = DIR_FINAL / "data" / "catalogo.json"
DIR_LOGS = Path(os.getenv("DIR_LOGS", DIR_FINAL / "logs"))

# --------------------------------------------------------------------------- #
# Dominio
# --------------------------------------------------------------------------- #
SUCURSALES = {
    "SUC-001": "Centro",
    "SUC-002": "Norte",
    "SUC-003": "Oeste",
}
# DOC-005: la reserva se mantiene 48 horas corridas desde la confirmación.
HORAS_RESERVA = 48
# schemas.py (Entrega 1): WhatsApp es canal minorista.
MAX_UNIDADES_RESERVA = 20

FRASE_ESCAPE = (
    "No poseo información sobre eso en las políticas de la farmacia. "
    "Te derivo con un empleado de la sucursal."
)
FRASE_LIMITE = (
    "Tu consulta necesita más pasos de los que puedo resolver por este medio. "
    "Te derivo con un empleado de la sucursal."
)
