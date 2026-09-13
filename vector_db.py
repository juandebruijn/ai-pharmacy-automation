"""
B.1 y B.3 — ChromaDB persistente y evento de negocio en caliente.

B.1  PersistentClient con ruta en disco, colección con hnsw:space="cosine",
     ingesta con upsert.
B.3  Evento de negocio en caliente con upsert, verificado con get(ids=[...]).

Uso:
    python vector_db.py --ingesta   # B.1
    python vector_db.py --evento    # B.3
"""

import argparse
import json
import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import GoogleGenaiEmbeddingFunction
from dotenv import load_dotenv

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

MODELO = "gemini-embedding-001"
DIMENSION = 768
RAIZ = Path(__file__).resolve().parent
BASE_JSON = RAIZ / "base_conocimiento.json"
DIR_CHROMA = RAIZ / "chroma_db"
COLECCION = "politicas_farmacia"

class EmbeddingFarmacia(GoogleGenaiEmbeddingFunction):
    """
    La función de embedding de ChromaDB para Gemini, con una sola diferencia: las
    CONSULTAS se vectorizan con task_type="RETRIEVAL_QUERY" en vez de
    "RETRIEVAL_DOCUMENT".

    No es un adorno. Una política de 900 caracteres y una pregunta de WhatsApp de
    diez palabras no son el mismo tipo de texto, y el modelo lo sabe. Medimos las
    dos variantes sobre las 24 consultas de C.2:

        consulta como DOCUMENTO -> peor caso dentro 0.2798, mejor fuera 0.2696
                                   (brecha NEGATIVA: ningún umbral separa)
        consulta como CONSULTA  -> peor caso dentro 0.3400, mejor fuera 0.3631
                                   (brecha +0.0231: el umbral de C.2 existe)

    Con la función tal como viene, el umbral de aceptación no se puede definir.
    """

    def __init__(self):
        super().__init__(model_name=MODELO, dimension=DIMENSION,
                         task_type="RETRIEVAL_DOCUMENT")
        self._como_consulta = GoogleGenaiEmbeddingFunction(
            model_name=MODELO, dimension=DIMENSION, task_type="RETRIEVAL_QUERY"
        )

    def embed_query(self, input):
        return self._como_consulta(input)


# --------------------------------------------------------------------------- #
# B.1 — Migración a ChromaDB
# --------------------------------------------------------------------------- #
def obtener_coleccion():
    """
    PersistentClient y no Client(): Client() es in-memory y nos devolvería al
    problema de volatilidad que A.5 demuestra con FAISS.
    """
    cliente = chromadb.PersistentClient(path=str(DIR_CHROMA))
    return cliente.get_or_create_collection(
        name=COLECCION,
        # Toma la GEMINI_API_KEY del entorno, que load_dotenv() ya cargó desde el .env.
        embedding_function=EmbeddingFarmacia(),
        # Coseno y no L2: las descripciones tienen largos distintos y con distancia
        # euclídea el documento más largo arrastra la distancia por magnitud.
        metadata={"hnsw:space": "cosine"},
    )


def aplanar_metadatos(metadatos: dict) -> dict:
    """
    ChromaDB solo acepta str, int, float o bool en los metadatos: una lista rompe
    el upsert, así que tags_regionales se guarda como string separado por comas.
    """
    return {
        clave: ", ".join(valor) if isinstance(valor, list) else valor
        for clave, valor in metadatos.items()
    }


def ingestar() -> None:
    """
    upsert y no add: add falla con IDExistsError en la segunda corrida y obliga a
    limpiar la colección a mano. Con upsert el script es idempotente y reconstruir
    la base entera desde el JSON es siempre seguro.
    """
    with BASE_JSON.open(encoding="utf-8") as f:
        documentos = json.load(f)

    coleccion = obtener_coleccion()
    antes = coleccion.count()
    coleccion.upsert(
        ids=[doc["id"] for doc in documentos],
        documents=[doc["descripcion_semantica"] for doc in documentos],
        metadatas=[aplanar_metadatos(doc["metadatos"]) for doc in documentos],
    )
    print(f"[B.1] Colección '{COLECCION}' en chroma_db/ (PersistentClient, hnsw:space=cosine)")
    print(f"      upsert de {len(documentos)} documentos: {antes} -> {coleccion.count()} registros.")


# --------------------------------------------------------------------------- #
# B.3 — Evento de negocio en caliente
# --------------------------------------------------------------------------- #
def evento_en_caliente() -> None:
    """
    La ventana de corte del reparto se adelanta por una contingencia mientras hay
    clientes con la conversación abierta preguntando si les llega hoy. Es el dato
    que A.1 identificó como el más volátil del dominio.
    """
    coleccion = obtener_coleccion()
    if coleccion.count() == 0:
        raise SystemExit("La colección está vacía. Corré primero: python vector_db.py --ingesta")

    previo = coleccion.get(ids=["DOC-004"])
    print("[B.3] ANTES del evento:")
    print(f"      {previo['documents'][0][:110]}...\n")

    coleccion.upsert(
        ids=["DOC-004"],
        documents=[
            "CONTINGENCIA VIGENTE HOY: por un alerta meteorológico la mensajería suspendió la "
            "ronda de la tarde y la ventana de corte del reparto se adelanta a las trece horas "
            "en las tres sucursales. Un pedido confirmado después de las trece no sale hoy y se "
            "reprograma para el primer reparto de mañana, sin excepción y sin importar la urgencia "
            "declarada por el cliente. El retiro en sucursal no está afectado y sigue disponible "
            "durante todo el horario de atención: es la alternativa que el sistema tiene que "
            "ofrecer cuando el cliente necesita el medicamento en el día."
        ],
        metadatas=[
            aplanar_metadatos(
                {
                    "categoria": "logistica",
                    "intencion_relacionada": "consulta_envio",
                    "sucursal": "TODAS",
                    "vigente": True,
                    "tags_regionales": ["contingencia", "corte adelantado", "no llega hoy"],
                }
            )
        ],
    )

    posterior = coleccion.get(ids=["DOC-004"])
    print("[B.3] DESPUÉS del upsert (verificado con get):")
    print(f"      {posterior['documents'][0][:110]}...")
    print(f"      metadatos: {posterior['metadatas'][0]}")
    print(f"      total de registros: {coleccion.count()} (no se duplicó nada)")


def main() -> int:
    parser = argparse.ArgumentParser(description="ChromaDB persistente (B.1 y B.3).")
    parser.add_argument("--ingesta", action="store_true", help="B.1: carga base_conocimiento.json con upsert.")
    parser.add_argument("--evento", action="store_true", help="B.3: evento en caliente + get de verificación.")
    args = parser.parse_args()

    if args.ingesta:
        ingestar()
    if args.evento:
        evento_en_caliente()
    if not (args.ingesta or args.evento):
        parser.print_help()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
