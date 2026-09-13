"""
B.1 – B.4 y B.6 — ChromaDB persistente, evento en caliente y búsqueda híbrida.

B.1  PersistentClient con ruta en disco, colección con hnsw:space="cosine",
     ingesta con upsert.
B.3  Evento de negocio en caliente con upsert, verificado con get(ids=[...]).
B.4  buscar_farmacia(): búsqueda semántica combinada con filtro duro por
     operadores nativos dentro del where.
B.6  Las tres killer queries.

Uso:
    python vector_db.py --ingesta                  # B.1
    python vector_db.py --evento                   # B.3
    python vector_db.py --buscar "me llega hoy?"   # B.4
    python vector_db.py --buscar "..." --categoria logistica --n 5
    python vector_db.py --killer                   # B.6
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

# C.2 — Umbral de aceptación. Distancia coseno: cuanto más chica, más parecido.
#
# No está elegido a ojo: sale de medir 14 consultas que sí tienen respuesta en la
# base contra 10 que no (ver informe_entrega2.md, C.2).
#   peor caso DENTRO del catálogo : 0.3400
#   mejor caso FUERA del catálogo : 0.3631
# 0.35 parte esa brecha. Si nada lo supera, el sistema no devuelve el más cercano:
# dice que no tiene la información.
UMBRAL_DISTANCIA = 0.35

RESPUESTA_SIN_MATCH = (
    "No tengo esa información en mi base de conocimiento. "
    "Te derivo con un empleado de la farmacia."
)


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


# --------------------------------------------------------------------------- #
# B.4 — Búsqueda híbrida
# --------------------------------------------------------------------------- #
def buscar_farmacia(query_semantica: str, filtro_categoria: str | None = None,
                    solo_vigentes: bool = True, n_resultados: int = 3) -> dict:
    """
    Búsqueda semántica (query_texts) + filtro duro por operadores nativos ($and,
    $eq) dentro del where.

    No hay post-filtering: ningún if de Python descarta resultados después de la
    query. Si filtráramos a posteriori, el motor calcularía similitud contra
    documentos que ya sabíamos que iban a descartarse y podríamos terminar con
    menos de n_resultados sin enterarnos.
    """
    condiciones = []
    if solo_vigentes:
        condiciones.append({"vigente": {"$eq": True}})
    if filtro_categoria:
        condiciones.append({"categoria": {"$eq": filtro_categoria}})

    # ChromaDB rechaza un $and de un solo elemento.
    if len(condiciones) > 1:
        where = {"$and": condiciones}
    elif condiciones:
        where = condiciones[0]
    else:
        where = None

    crudo = obtener_coleccion().query(
        query_texts=[query_semantica],
        n_results=n_resultados,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    resultados = [
        {
            "id": doc_id,
            "distancia": round(float(distancia), 4),
            "supera_umbral": float(distancia) <= UMBRAL_DISTANCIA,
            "metadatos": metadatos,
            "texto": documento,
        }
        for doc_id, documento, metadatos, distancia in zip(
            crudo["ids"][0], crudo["documents"][0], crudo["metadatas"][0], crudo["distances"][0]
        )
    ]

    hay_respuesta = any(r["supera_umbral"] for r in resultados)
    return {
        "consulta": query_semantica,
        "where": where,
        "resultados": resultados,
        "hay_respuesta": hay_respuesta,
        "respuesta_sugerida": None if hay_respuesta else RESPUESTA_SIN_MATCH,
    }


def imprimir(salida: dict) -> None:
    print(f'Consulta: "{salida["consulta"]}"')
    print(f"where:    {json.dumps(salida['where'], ensure_ascii=False)}")
    print(f"umbral:   distancia <= {UMBRAL_DISTANCIA}")
    for puesto, r in enumerate(salida["resultados"], start=1):
        estado = "ACEPTADO" if r["supera_umbral"] else "descartado por umbral"
        meta = r["metadatos"]
        print(f"  {puesto}. {r['id']}  distancia={r['distancia']:.4f}  [{estado}]")
        print(f"     categoria={meta['categoria']}  vigente={meta['vigente']}")
        print(f"     {r['texto'][:110]}...")
    if not salida["hay_respuesta"]:
        print(f'\n  >> Sin coincidencias sobre el umbral. El sistema responde:\n'
              f'     "{salida["respuesta_sugerida"]}"')
    print()


# --------------------------------------------------------------------------- #
# B.6 — Killer Queries
# --------------------------------------------------------------------------- #
def killer_queries() -> None:
    # Re-ingesta para partir del estado base: si quedó corrido --evento, DOC-004 está mutado.
    ingestar()
    print()

    print("=" * 78)
    print("KILLER QUERY 1 — Poder semántico: jerga sin ninguna palabra del documento")
    print("=" * 78)
    imprimir(buscar_farmacia(
        "mi vieja tiene 80 años y toma pastillas para el corazón todos los días, "
        "¿le sale algo o no paga nada?",
        filtro_categoria="cobertura",
    ))

    trampa = ("el médico me hizo la receta en su papel con membrete, ¿sirve para el alprazolam? "
              "¿me lo pueden enviar a domicilio?")
    print("=" * 78)
    print("KILLER QUERY 2 — El metadato salva el día")
    print("=" * 78)
    print("DOC-020 es el régimen de psicotrópicos DEROGADO (vigente=False) y describe")
    print("exactamente lo que el cliente pregunta, así que la semántica cruda lo pone primero.\n")
    print("--- (a) SIN el filtro de vigencia ---")
    imprimir(buscar_farmacia(trampa, solo_vigentes=False))
    print("--- (b) CON el filtro de vigencia en el where ---")
    imprimir(buscar_farmacia(trampa, solo_vigentes=True))

    print("=" * 78)
    print("KILLER QUERY 3 — Prueba de estrés: consulta fuera del catálogo")
    print("=" * 78)
    imprimir(buscar_farmacia("¿hacen análisis de sangre y electrocardiogramas en la farmacia?"))


def main() -> int:
    parser = argparse.ArgumentParser(description="ChromaDB y búsqueda híbrida (B.1–B.4, B.6).")
    parser.add_argument("--ingesta", action="store_true", help="B.1: carga base_conocimiento.json con upsert.")
    parser.add_argument("--evento", action="store_true", help="B.3: evento en caliente + get de verificación.")
    parser.add_argument("--buscar", metavar="CONSULTA", help="B.4: búsqueda híbrida.")
    parser.add_argument("--categoria", help="filtro duro por categoría.")
    parser.add_argument("--n", type=int, default=3, help="cantidad de resultados (default 3).")
    parser.add_argument("--incluir-no-vigentes", action="store_true", help="desactiva el filtro de vigencia.")
    parser.add_argument("--killer", action="store_true", help="B.6: las tres killer queries.")
    args = parser.parse_args()

    if args.ingesta:
        ingestar()
    if args.evento:
        evento_en_caliente()
    if args.buscar:
        imprimir(buscar_farmacia(args.buscar, args.categoria,
                                 not args.incluir_no_vigentes, args.n))
    if args.killer:
        killer_queries()
    if not (args.ingesta or args.evento or args.buscar or args.killer):
        parser.print_help()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
