"""
B.5 — ETL y purga semántica.

Toma la base canónica (base_conocimiento.json) y el lote crudo cargado a mano
(ingesta_cruda.json) y:

  1. Normaliza las claves mal nombradas y los tipos mal cargados.
  2. Resuelve la colisión de IDs reasignando el primer id libre.
  3. Purga semántica: vectoriza, detecta los pares por debajo del umbral de
     distancia coseno y elimina el duplicado de la colección.

Uso:
    python etl_purga.py
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

import vector_db

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

MODELO = "gemini-embedding-001"
DIMENSION = 768
RAIZ = Path(__file__).resolve().parent
BASE_JSON = RAIZ / "base_conocimiento.json"
CRUDO_JSON = RAIZ / "ingesta_cruda.json"

# Umbral de duplicado semántico. Medido sobre la base real (ver informe, B.5):
#
#   casi-duplicados reales   : 0.0398 / 0.0773 / 0.1115
#   par LEGÍTIMO más cercano : 0.1248  (DOC-013, régimen de psicotrópicos vigente,
#                              contra DOC-020, el mismo régimen derogado)
#
# 0.12 entra en esa ventana. El margen es fino a propósito: ese par habla del mismo
# tema con el mismo vocabulario y solo se diferencia por la vigencia, así que un
# umbral más laxo fusionaría la norma vigente con la derogada — que es justo el
# desastre que la Killer Query 2 pone a prueba.
UMBRAL_DUPLICADO = 0.12

cliente = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

VERDADEROS = {"true", "1", "si", "sí", "verdadero"}


def vectorizar(textos: list[str]) -> np.ndarray:
    respuesta = cliente.models.embed_content(
        model=MODELO,
        contents=textos,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT", output_dimensionality=DIMENSION
        ),
    )
    matriz = np.asarray([e.values for e in respuesta.embeddings], dtype="float32")
    return matriz / np.linalg.norm(matriz, axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# 1 — Normalización de claves y tipos
# --------------------------------------------------------------------------- #
def normalizar(registro: dict, incidencias: list[str]) -> dict:
    doc_id = registro["id"]

    texto = registro.get("descripcion_semantica")
    if texto is None:
        texto = registro["descripcion"]
        incidencias.append(f"[CLAVE] {doc_id}: 'descripcion' -> 'descripcion_semantica'")

    metadatos = registro.get("metadatos")
    if metadatos is None:
        metadatos = registro["metadata"]
        incidencias.append(f"[CLAVE] {doc_id}: 'metadata' -> 'metadatos'")
    metadatos = dict(metadatos)

    vigente = metadatos.get("vigente", True)
    if not isinstance(vigente, bool):
        convertido = str(vigente).strip().lower() in VERDADEROS
        incidencias.append(
            f"[TIPO] {doc_id}: 'vigente' venía como string {vigente!r} -> bool {convertido}"
        )
        metadatos["vigente"] = convertido

    tags = metadatos.get("tags_regionales", [])
    if isinstance(tags, str):
        metadatos["tags_regionales"] = [t.strip() for t in tags.split(",") if t.strip()]
        incidencias.append(
            f"[TIPO] {doc_id}: 'tags_regionales' venía como string -> lista de "
            f"{len(metadatos['tags_regionales'])} tags"
        )

    return {"id": doc_id, "descripcion_semantica": texto, "metadatos": metadatos}


# --------------------------------------------------------------------------- #
# 2 — Colisión de IDs
# --------------------------------------------------------------------------- #
def resolver_colisiones(canonicos: list[dict], entrantes: list[dict],
                        incidencias: list[str]) -> list[dict]:
    ocupados = {r["id"] for r in canonicos}
    resueltos = []

    for registro in entrantes:
        if registro["id"] in ocupados:
            numeros = {int(m.group(1)) for i in ocupados if (m := re.fullmatch(r"DOC-(\d+)", i))}
            siguiente = 1
            while siguiente in numeros:
                siguiente += 1
            nuevo = f"DOC-{siguiente:03d}"
            incidencias.append(
                f"[ID] {registro['id']}: el id ya existe en la base con OTRO contenido "
                f"-> reasignado a {nuevo}"
            )
            registro = {**registro, "id": nuevo}
        ocupados.add(registro["id"])
        resueltos.append(registro)

    return resueltos


# --------------------------------------------------------------------------- #
# 3 — Purga semántica
# --------------------------------------------------------------------------- #
def purgar(canonicos: list[dict], entrantes: list[dict]):
    """
    Vectoriza todo junto y elimina los entrantes que dicen lo mismo que un canónico.

    Solo se eliminan registros del lote crudo: un documento canónico nunca se borra
    automáticamente, por más cerca que caiga de otro.
    """
    todos = canonicos + entrantes
    matriz = vectorizar([r["descripcion_semantica"] for r in todos])

    # Con vectores de norma 1, la distancia coseno es 1 - producto interno.
    distancias = 1.0 - (matriz @ matriz.T)

    eliminados = []
    indices_purgados = set()

    for i in range(len(canonicos), len(todos)):           # solo entrantes
        for j in range(len(canonicos)):                   # contra los canónicos
            if distancias[i, j] <= UMBRAL_DUPLICADO:
                eliminados.append(
                    {
                        "eliminado": todos[i]["id"],
                        "conservado": todos[j]["id"],
                        "distancia": round(float(distancias[i, j]), 4),
                        "texto_eliminado": todos[i]["descripcion_semantica"][:95],
                        "texto_conservado": todos[j]["descripcion_semantica"][:95],
                    }
                )
                indices_purgados.add(i)
                break

    finales = [r for k, r in enumerate(todos) if k not in indices_purgados]
    return finales, eliminados


def main() -> int:
    incidencias: list[str] = []

    with BASE_JSON.open(encoding="utf-8") as f:
        canonicos = [normalizar(r, incidencias) for r in json.load(f)]
    with CRUDO_JSON.open(encoding="utf-8") as f:
        entrantes = [normalizar(r, incidencias) for r in json.load(f)]

    print("=" * 78)
    print("ETL — Normalización de claves y tipos")
    print("=" * 78)
    print(f"  base canónica : {len(canonicos):3d} registros")
    print(f"  lote entrante : {len(entrantes):3d} registros\n")
    for incidencia in incidencias:
        print(f"  {incidencia}")

    print("\n" + "=" * 78)
    print("ETL — Colisión de IDs")
    print("=" * 78)
    marca = len(incidencias)
    entrantes = resolver_colisiones(canonicos, entrantes, incidencias)
    for incidencia in incidencias[marca:]:
        print(f"  {incidencia}")

    print("\n" + "=" * 78)
    print(f"Purga semántica — distancia coseno <= {UMBRAL_DUPLICADO}")
    print("=" * 78)
    finales, eliminados = purgar(canonicos, entrantes)
    for caso in eliminados:
        print(f"\n  ELIMINADO {caso['eliminado']}  (distancia {caso['distancia']} a {caso['conservado']})")
        print(f"    se conserva : {caso['conservado']}  \"{caso['texto_conservado']}...\"")
        print(f"    se descarta : {caso['eliminado']}  \"{caso['texto_eliminado']}...\"")

    # La purga se aplica sobre la colección: los que quedan se upsertean y los
    # duplicados se borran por id.
    coleccion = vector_db.obtener_coleccion()
    coleccion.upsert(
        ids=[r["id"] for r in finales],
        documents=[r["descripcion_semantica"] for r in finales],
        metadatas=[vector_db.aplanar_metadatos(r["metadatos"]) for r in finales],
    )
    if eliminados:
        coleccion.delete(ids=[caso["eliminado"] for caso in eliminados])

    print("\n" + "=" * 78)
    print("Resultado")
    print("=" * 78)
    print(f"  {len(canonicos)} canónicos + {len(entrantes)} entrantes - {len(eliminados)} duplicados "
          f"= {len(finales)} documentos")
    print(f"  colección sincronizada: {coleccion.count()} registros")
    print("\n  Por qué un SELECT DISTINCT no habría encontrado nada de esto:")
    print("    DISTINCT compara bytes. Los pares purgados no comparten ni el id, ni el texto,")
    print("    ni una sola oración completa: dicen lo mismo con otras palabras. Para SQL son")
    print("    filas perfectamente distintas y las deja pasar todas. Solo el espacio vectorial")
    print("    las detecta. Al revés tampoco sirve: jamás habría visto la colisión de DOC-010,")
    print("    donde el id coincide pero el contenido es otro.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
