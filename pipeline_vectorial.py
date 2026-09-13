"""
A.4 / A.5 — Índice FAISS persistido sobre la Base de Conocimiento.

  1. Lee la credencial desde el .env (nunca hardcodeada).
  2. Genera los embeddings de las `descripcion_semantica` con gemini-embedding-001.
  3. Construye un IndexFlatIP sobre vectores normalizados a norma 1: con ‖v‖ = 1 el
     producto interno ES la similitud coseno, la misma métrica que usa ChromaDB en
     la Parte B, así que los números de las dos partes son comparables.
  4. Persiste con faiss.write_index() y recarga con faiss.read_index() si ya existe.
  5. Corre búsqueda semántica top-K sobre 3 consultas de prueba.

Uso:
    python pipeline_vectorial.py             # carga de disco si existe; si no, construye y persiste
    python pipeline_vectorial.py --volatil   # A.5: construye en RAM y NO persiste
    python pipeline_vectorial.py --borrar    # A.5: borra el índice para simular el reinicio
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

# La consola de Windows usa cp1252 por defecto y rompe los acentos del dominio.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

MODELO = "gemini-embedding-001"
DIMENSION = 768
RAIZ = Path(__file__).resolve().parent
BASE_JSON = RAIZ / "base_conocimiento.json"
DIR_INDICE = RAIZ / "indice_faiss"
ARCHIVO_INDICE = DIR_INDICE / "farmacia.index"
ARCHIVO_IDS = DIR_INDICE / "farmacia_ids.json"
TOP_K = 3

# Escritas como las escribiría un cliente por WhatsApp: con jerga y sin las
# palabras exactas del documento que tienen que recuperar.
CONSULTAS_DE_PRUEBA = [
    "che, me lo pueden acercar en moto hasta el bajo de Vicente López?",
    "soy jubilado, los remedios de la presión los tengo que pagar?",
    "puedo abonar escaneando con la billetera del celular?",
]

cliente = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def vectorizar(textos: list[str], tipo: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
    """
    Devuelve la matriz (n, 768) de embeddings normalizados, en float32 como pide FAISS.

    Las descripciones y las consultas se vectorizan con `task_type` distinto: es un
    documento largo contra una pregunta corta, y el modelo rinde mejor si se lo declara.
    """
    respuesta = cliente.models.embed_content(
        model=MODELO,
        contents=textos,
        config=types.EmbedContentConfig(task_type=tipo, output_dimensionality=DIMENSION),
    )
    matriz = np.asarray([e.values for e in respuesta.embeddings], dtype="float32")
    return matriz / np.linalg.norm(matriz, axis=1, keepdims=True)


def cargar_documentos() -> list[dict]:
    with BASE_JSON.open(encoding="utf-8") as f:
        return json.load(f)


def construir_indice(documentos: list[dict]) -> tuple[faiss.Index, list[str]]:
    ids = [doc["id"] for doc in documentos]
    textos = [doc["descripcion_semantica"] for doc in documentos]

    print(f"  Vectorizando {len(textos)} descripciones con {MODELO}...")
    matriz = vectorizar(textos)

    indice = faiss.IndexFlatIP(matriz.shape[1])
    indice.add(matriz)
    print(f"  Índice construido: {indice.ntotal} vectores de {matriz.shape[1]} dimensiones.")
    return indice, ids


def persistir(indice: faiss.Index, ids: list[str]) -> None:
    """write_index() + el archivo de ids."""
    DIR_INDICE.mkdir(parents=True, exist_ok=True)
    # write_index() es C++ y abre el archivo con fopen(), que en Windows interpreta la
    # ruta con la codepage ANSI: si el repo está clonado en una carpeta con acentos
    # (acá, "5° Cuatrimestre") la ruta se corrompe al cruzar a C++ y FAISS falla aunque
    # el directorio exista. Parados adentro del directorio, el nombre que cruza es ASCII.
    previo = os.getcwd()
    try:
        os.chdir(DIR_INDICE)
        faiss.write_index(indice, ARCHIVO_INDICE.name)
    finally:
        os.chdir(previo)

    # FAISS guarda vectores, no identificadores: la fila i del índice es el id[i].
    # Sin este mapeo el índice devuelve "fila 7" y nadie sabe qué documento es.
    ARCHIVO_IDS.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    print(f"  Persistido en {ARCHIVO_INDICE.relative_to(RAIZ)} "
          f"({ARCHIVO_INDICE.stat().st_size / 1024:.1f} KB).")


def recargar() -> tuple[faiss.Index, list[str]] | None:
    """read_index() si el índice existe en disco. None si no hay nada."""
    if not (ARCHIVO_INDICE.exists() and ARCHIVO_IDS.exists()):
        return None
    previo = os.getcwd()
    try:
        os.chdir(DIR_INDICE)
        indice = faiss.read_index(ARCHIVO_INDICE.name)
    finally:
        os.chdir(previo)
    return indice, json.loads(ARCHIVO_IDS.read_text(encoding="utf-8"))


def buscar(indice: faiss.Index, ids: list[str], documentos: list[dict], consulta: str) -> None:
    por_id = {doc["id"]: doc for doc in documentos}
    similitudes, filas = indice.search(vectorizar([consulta], "RETRIEVAL_QUERY"), TOP_K)

    for puesto, (similitud, fila) in enumerate(zip(similitudes[0], filas[0]), start=1):
        doc = por_id[ids[fila]]
        # Con vectores de norma 1, la distancia coseno es exactamente 1 - producto interno.
        print(f"    {puesto}. {doc['id']}  similitud={similitud:.4f}  distancia={1 - similitud:.4f}"
              f"  ({doc['metadatos']['categoria']})")
        print(f"       {doc['descripcion_semantica'][:100]}...")


def main() -> int:
    parser = argparse.ArgumentParser(description="Índice FAISS de la Base de Conocimiento (A.4/A.5).")
    parser.add_argument("--volatil", action="store_true",
                        help="A.5: construye el índice en RAM y NO llama a write_index().")
    parser.add_argument("--borrar", action="store_true",
                        help="A.5: borra indice_faiss/ para simular el reinicio del servidor.")
    args = parser.parse_args()

    if args.borrar:
        if DIR_INDICE.exists():
            shutil.rmtree(DIR_INDICE)
            print("[A.5] indice_faiss/ borrado. El índice ya no existe en disco.")
        else:
            print("[A.5] indice_faiss/ no existía.")
        return 0

    documentos = cargar_documentos()
    print(f"Base de conocimiento: {len(documentos)} documentos.\n")

    recuperado = None if args.volatil else recargar()

    if recuperado is not None:
        indice, ids = recuperado
        print(f"[read_index] Índice recargado desde disco: {indice.ntotal} vectores.")
        print("             0 llamadas a la API, 0 tokens consumidos.\n")
    else:
        print("[build] --volatil: se construye en RAM y NO se persiste."
              if args.volatil else "[build] no hay índice en disco.")
        indice, ids = construir_indice(documentos)
        if args.volatil:
            print("  [A.5] write_index() OMITIDO a propósito: cuando este proceso termine,\n"
                  "        el índice muere con la RAM y hay que volver a pagar los tokens.\n")
        else:
            persistir(indice, ids)
            print()

    print(f"Búsqueda semántica top-{TOP_K}")
    print("=" * 78)
    for numero, consulta in enumerate(CONSULTAS_DE_PRUEBA, start=1):
        print(f'\n[{numero}] "{consulta}"')
        buscar(indice, ids, documentos, consulta)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
