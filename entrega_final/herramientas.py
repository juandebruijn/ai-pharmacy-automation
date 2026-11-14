"""A.1 — Herramientas del agente de la farmacia.

El LLM elige cuál usar leyendo SOLO el nombre, el docstring y la firma de cada
@tool (bind_tools los manda como JSON Schema). Por eso cada docstring dice
cuándo usarla, cuándo NO usarla, qué formato tienen los argumentos y qué
devuelve. Los dominios cerrados (sucursales) van como Literal: el modelo no
puede inventar una sucursal que no existe.

    buscar_politicas   retriever   el RAG de la Entrega 3 (ChromaDB, chunks 500/100)
    consultar_stock    sensor      lee stock y precio del ERP, por sucursal
    reservar_producto  actuador    bloquea stock por 48 h en el ERP (efecto real)
    consultar_reserva  sensor      lee las reservas de la conversación

Todas devuelven JSON: una salida predecible se interpreta igual en cada vuelta,
y un error es un campo "error" accionable, no una excepción que rompe el grafo.
"""

import json
from functools import lru_cache
from typing import Literal

import chromadb
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from entrega_final import erp
from entrega_final.config import SUCURSALES

Sucursal = Literal["SUC-001", "SUC-002", "SUC-003"]

K_POLITICAS = 4


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


@lru_cache(maxsize=1)
def _retriever():
    """
    El retriever de la Entrega 3, importado y no reescrito: misma base, misma
    función de embedding y mismo filtro de vigencia. Usa la colección de chunks
    500/100 de B.2; si todavía no se creó, cae a la de documentos enteros.
    """
    from entrega_3.rag_pipeline import COLECCION, COLECCION_CHUNKS, DIR_CHROMA, crear_retriever

    existentes = {c.name for c in chromadb.PersistentClient(path=str(DIR_CHROMA)).list_collections()}
    coleccion = COLECCION_CHUNKS if COLECCION_CHUNKS in existentes else COLECCION
    return crear_retriever(coleccion, K_POLITICAS)


def _thread_id(config: RunnableConfig) -> str:
    return (config or {}).get("configurable", {}).get("thread_id", "sin-thread")


# --------------------------------------------------------------------------- #
# 1. Retriever — conocimiento documental (RAG de la Entrega 3)
# --------------------------------------------------------------------------- #
@tool
def buscar_politicas(consulta: str) -> str:
    """Busca en las políticas vigentes de la farmacia los fragmentos que responden una consulta.

    Usar para cualquier pregunta sobre REGLAS del negocio: zonas y horarios de
    envío a domicilio, retiro en sucursal y plazo de reserva, medios de pago,
    descuentos bancarios, convenios con obras sociales y prepagas (OSDE, PAMI,
    Swiss Medical, IOMA...), requisitos de receta, psicotrópicos, cadena de frío,
    horarios de atención, devoluciones y qué pasa si no hay stock.
    NO usar para saber si hay stock o cuánto cuesta un producto: eso está en
    consultar_stock. Las políticas son la única fuente autorizada sobre reglas:
    si no aparece en los fragmentos, no está permitido afirmarlo.

    Args:
        consulta: La pregunta del cliente reformulada en una oración clara y
            completa, en español (ej. "envío a domicilio a Ramos Mejía los sábados").

    Returns:
        JSON con "fragmentos": lista de {doc_id, categoria, sucursal, texto}.
        Cada afirmación de la respuesta final debe citar su doc_id entre
        corchetes, por ejemplo [DOC-005]. Si la lista viene vacía, no hay
        política que responda.
    """
    documentos = _retriever().invoke(consulta)
    return _json({
        "fragmentos": [
            {
                "doc_id": d.metadata.get("doc_id") or d.id,
                "categoria": d.metadata.get("categoria"),
                "sucursal": d.metadata.get("sucursal"),
                "texto": d.page_content,
            }
            for d in documentos
        ]
    })


# --------------------------------------------------------------------------- #
# 2. Sensor — stock y precio en el ERP
# --------------------------------------------------------------------------- #
@tool
def consultar_stock(producto: str, sucursal_id: Sucursal | None = None) -> str:
    """Consulta en el ERP el stock disponible y el precio de lista de un medicamento, por sucursal.

    Usar cuando el cliente pregunta si hay un producto, cuántas unidades quedan,
    cuánto cuesta o en qué sucursal conseguirlo, y SIEMPRE antes de
    reservar_producto (para obtener el producto_id). Es la única fuente válida
    de precios: nunca informar un precio que no salga de esta herramienta.
    NO usar para preguntas sobre reglas (envíos, convenios, recetas): eso es
    buscar_politicas.

    Args:
        producto: Nombre comercial o droga tal como lo escribió el cliente
            (ej. "Ibupirac", "amoxicilina", "tafirol 500").
        sucursal_id: SUC-001 (Centro), SUC-002 (Norte) o SUC-003 (Oeste).
            Omitir para ver todas las sucursales.

    Returns:
        JSON con "resultados": lista de {producto_id, sucursal_id, sucursal,
        nombre_comercial, droga, presentacion, precio_lista_ars,
        unidades_disponibles, requiere_receta, psicotropico, cadena_frio}.
        El precio es de lista, sin descuentos de obra social. Si no hay
        coincidencias devuelve un campo "error".
    """
    resultados = erp.buscar_productos(producto, sucursal_id)
    if not resultados:
        return _json({"error": f"No hay ningún producto que coincida con '{producto}' en el "
                               "catálogo. Pedile al cliente el nombre comercial o la droga."})
    salida: dict = {"resultados": resultados}
    # Avisos explícitos: en la carga de prueba el modelo leyó psicotropico=true y
    # aun así ofreció reservar, y ante stock 0 negó el encargo sin consultar las
    # políticas. Un aviso en la observación pesa más que un booleano suelto.
    avisos = []
    if any(r["psicotropico"] for r in resultados):
        avisos.append("Psicotrópico: NO se reserva ni se vende por WhatsApp; la dispensa es solo "
                      "presencial con receta oficial. Consultá buscar_politicas para citar la norma.")
    if all(r["unidades_disponibles"] == 0 for r in resultados):
        avisos.append("Sin stock: antes de responder, consultá buscar_politicas sobre qué alternativas "
                      "hay cuando un producto no tiene stock.")
    if avisos:
        salida["avisos"] = avisos
    return _json(salida)


# --------------------------------------------------------------------------- #
# 3. Actuador — reserva de stock (acción con efecto)
# --------------------------------------------------------------------------- #
@tool
def reservar_producto(producto_id: int, cantidad: int, config: RunnableConfig) -> str:
    """Reserva unidades de un producto en una sucursal por 48 horas para retirar (acción con efecto en el ERP).

    Usar SOLO cuando el cliente pide explícitamente reservar, apartar, encargar
    para retirar o "guardar" un producto, y ya se conoce el producto_id de la
    sucursal elegida (obtenerlo antes con consultar_stock). Si el cliente no dijo
    en qué sucursal ni cuántas unidades, preguntárselo antes de llamar. NO usar
    para consultas informativas ni para "ver si hay".
    La herramienta rechaza psicotrópicos, cantidades mayores a 20 y pedidos sin
    stock suficiente; en ese caso explicar el motivo al cliente.

    Args:
        producto_id: El producto_id exacto que devolvió consultar_stock para la
            sucursal elegida (cada sucursal tiene su propio producto_id).
        cantidad: Unidades a reservar, entre 1 y 20.

    Returns:
        JSON con "ok": true y la "reserva" (codigo RES-XXXXXX, producto,
        sucursal, cantidad, precio_total, estado, vence_en), o "ok": false y el
        "motivo". Estado "pendiente_receta" significa que el farmacéutico debe
        validar la receta antes de entregar.
    """
    return _json(erp.crear_reserva(producto_id, cantidad, _thread_id(config)))


@tool
def consultar_reserva(config: RunnableConfig, codigo: str | None = None) -> str:
    """Devuelve las reservas que el cliente hizo en esta conversación, con su estado y vencimiento.

    Usar cuando el cliente pregunta por "mi reserva", si sigue vigente, hasta
    cuándo se la guardan o cuánto tiene que pagar al retirar. Solo ve las
    reservas de esta misma conversación.
    NO usar para crear una reserva nueva (eso es reservar_producto).

    Args:
        codigo: Código RES-XXXXXX si el cliente lo menciona. Omitir para listar
            todas las reservas de la conversación.

    Returns:
        JSON con "reservas": lista de {codigo, producto, presentacion, sucursal,
        cantidad, precio_total, estado, vence_en}. Estados posibles:
        confirmada, pendiente_receta, vencida, cancelada.
    """
    reservas = erp.obtener_reservas(_thread_id(config), codigo)
    if not reservas:
        return _json({"reservas": [], "aviso": "No hay reservas registradas en esta conversación."})
    return _json({"reservas": reservas})


HERRAMIENTAS = [buscar_politicas, consultar_stock, reservar_producto, consultar_reserva]
NOMBRES_SUCURSALES = ", ".join(f"{k} ({v})" for k, v in SUCURSALES.items())
