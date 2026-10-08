"""
B.1 – B.4 y B.6 — ChromaDB persistente, evento en caliente y búsqueda híbrida.

B.1  PersistentClient con ruta en disco, colección con hnsw:space="cosine",
     ingesta con upsert.
B.3  Evento de negocio en caliente con upsert, verificado con get(ids=[...]).
B.4  buscar_farmacia(): búsqueda semántica combinada con filtro duro por
     operadores nativos dentro del where.
B.6  Las tres killer queries.
C.2  Calibración del umbral de aceptación.

Uso:
    python vector_db.py --ingesta                  # B.1
    python vector_db.py --evento                   # B.3
    python vector_db.py --buscar "me llega hoy?"   # B.4
    python vector_db.py --buscar "..." --categoria logistica --n 5
    python vector_db.py --buscar "..." --sucursal SUC-002 --intencion consulta_envio
    python vector_db.py --killer                   # B.6
    python vector_db.py --calibrar                 # C.2
"""

import argparse
import json
import sys
import time
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
# No está elegido a ojo: sale de `python vector_db.py --calibrar`, que mide 41
# consultas que sí tienen respuesta en la base contra 25 que no (informe, C.2).
# Los dos grupos se SOLAPAN (brecha -0.0202), así que no existe un corte perfecto:
# 0.340 y 0.345 empatan con 2 errores sobre 66 (97 %) y se toma 0.345, el centro
# de ese tramo, porque 0.340 cae exactamente sobre una consulta válida.
# Si nada lo supera, el sistema no devuelve el más cercano: dice que no tiene la
# información. El re-ranker que ataca el solapamiento queda para la Entrega 3.
UMBRAL_DISTANCIA = 0.345

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
    dos variantes sobre las 24 consultas de la primera calibración de C.2:

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
                    solo_vigentes: bool = True, n_resultados: int = 3,
                    sucursal: str | None = None, intencion: str | None = None) -> dict:
    """
    Búsqueda semántica (query_texts) + filtro duro por operadores nativos ($and,
    $eq, $in) dentro del where.

    sucursal e intencion son los dos campos que el orquestador ya tiene antes de
    buscar: sucursal_id viene en el request de la API (Entrega 1) e intencion es
    la etiqueta que extrae app.py. La sucursal se resuelve con $in y no con $eq
    porque las políticas de red (sucursal="TODAS") aplican a todas.

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
    if sucursal:
        condiciones.append({"sucursal": {"$in": [sucursal, "TODAS"]}})
    if intencion:
        condiciones.append({"intencion_relacionada": {"$eq": intencion}})

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
        print(f"     categoria={meta['categoria']}  sucursal={meta['sucursal']}  "
              f"intencion={meta['intencion_relacionada']}  vigente={meta['vigente']}")
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


# --------------------------------------------------------------------------- #
# C.2 — Calibración del umbral de aceptación
# --------------------------------------------------------------------------- #
# (consulta, documento que la responde), escritas como pregunta un cliente por
# WhatsApp y sin repetir el vocabulario del documento. DOC-104, DOC-105 y DOC-021
# existen recién después del ETL de B.5.
CALIBRACION_DENTRO = [
    ("me mandan algo a Recoleta?", "DOC-001"),
    ("llegan con el delivery hasta Congreso?", "DOC-001"),
    ("hacen envíos a Martínez o San Isidro?", "DOC-002"),
    ("vivo en Nordelta, me lo pueden llevar?", "DOC-002"),
    ("reparten en Morón los sábados?", "DOC-003"),
    ("estoy en Ramos Mejía, a qué hora sale el cadete?", "DOC-003"),
    ("hasta qué hora puedo pedir para que me llegue hoy?", "DOC-004"),
    ("si compro a la tarde me llega en el día?", "DOC-004"),
    ("lo reservo por acá y lo paso a buscar mañana?", "DOC-005"),
    ("cuánto tiempo me guardan el pedido si lo voy a retirar?", "DOC-005"),
    ("me pueden mandar la insulina o se corta la cadena de frío?", "DOC-006"),
    ("la vacuna viaja en heladerita?", "DOC-006"),
    ("aceptan mercado pago?", "DOC-007"),
    ("puedo pagar en efectivo cuando me lo traen?", "DOC-007"),
    ("se puede pagar con tarjeta de débito?", "DOC-007"),
    ("puedo abonar escaneando con la billetera del celular?", "DOC-007"),
    ("el descuento del banco se suma al de la prepaga?", "DOC-008"),
    ("los miércoles hay promo con algún banco?", "DOC-008"),
    ("tengo OSDE 210, cuánto me cubren?", "DOC-009"),
    ("con OSDE me hacen descuento en los remedios de todos los meses?", "DOC-009"),
    ("soy jubilada, la metformina me sale gratis?", "DOC-010"),
    ("mi abuelo tiene PAMI, qué le cubren?", "DOC-010"),
    ("trabajan con Swiss Medical?", "DOC-011"),
    ("qué tengo que llevar para usar Medifé?", "DOC-011"),
    ("me venden clonazepam con la receta común?", "DOC-013"),
    ("qué receta necesito para el rivotril?", "DOC-013"),
    ("me sirve la receta digital que me mandó el médico por mail?", "DOC-014"),
    ("si les mando foto de la receta alcanza?", "DOC-014"),
    ("necesito receta para comprar ibuprofeno?", "DOC-015"),
    ("el paracetamol es de venta libre?", "DOC-015"),
    ("me venden amoxicilina sin receta? tengo la caja vieja", "DOC-016"),
    ("necesito antibiótico para la garganta, sin receta se puede?", "DOC-016"),
    ("están abiertos el domingo?", "DOC-017"),
    ("hay alguna farmacia de turno a la noche?", "DOC-017"),
    ("me equivoqué de remedio, lo puedo devolver?", "DOC-018"),
    ("si no lo uso me devuelven la plata?", "DOC-018"),
    ("y si no tienen el remedio en esta sucursal qué hago?", "DOC-019"),
    ("me lo pueden encargar si no les queda?", "DOC-019"),
    ("dan la vacuna de la gripe?", "DOC-104"),
    ("venden leche para bebé con alergia a la proteína?", "DOC-105"),
    ("tienen alguien que me asesore con protector solar?", "DOC-021"),
]

# Sin respuesta en la base. La primera mitad son "vecinas" del dominio —cosas que
# una farmacia podría hacer pero esta no documenta—: son las que estresan el umbral.
CALIBRACION_FUERA = [
    "¿hacen análisis de sangre y electrocardiogramas en la farmacia?",
    "¿me pueden tomar la presión ahí?",
    "¿venden lentes de contacto o anteojos?",
    "¿hacen recetas magistrales o preparados?",
    "¿tienen test de embarazo?",
    "¿me pueden aplicar una inyección intramuscular?",
    "¿venden alimento para perros o remedios veterinarios?",
    "¿tienen sillas de ruedas o muletas en alquiler?",
    "¿trabajan con IOSFA?",
    "¿puedo pagar en cuotas sin interés con Naranja?",
    "¿hacen tarjeta de fidelidad o puntos?",
    "¿hacen perforación de orejas?",
    "¿me pueden recomendar un médico clínico de la zona?",
    "¿tienen el certificado de vacunas digital?",
    "¿hacen envíos al interior del país por correo?",
    "¿cuál es el CUIT de la farmacia para facturar?",
    "¿a qué hora juega Boca hoy?",
    "¿cuánto está el dólar blue?",
    "¿me pasás una receta de milanesas?",
    "¿buscan empleados? quiero dejar mi CV",
    "¿dónde queda el banco más cercano?",
    "¿cómo cambio la clave del wifi?",
    "¿venden cargadores de celular?",
    "¿qué película dan en el cine?",
    "¿alquilan el local de al lado?",
]


def _top1(consulta: str, intentos: int = 3) -> tuple[str, float]:
    # La API de embeddings devuelve 500 INTERNAL de vez en cuando: con 66 consultas
    # seguidas es casi seguro que alguna lo pise, así que se reintenta.
    for intento in range(1, intentos + 1):
        try:
            r = buscar_farmacia(consulta, n_resultados=1)["resultados"][0]
            return r["id"], r["distancia"]
        except ValueError:
            if intento == intentos:
                raise
            time.sleep(2 * intento)


def calibrar_umbral() -> None:
    """
    Mide el top-1 de cada consulta de calibración y barre umbrales candidatos.

    Si los dos grupos no se tocan, el umbral es el punto medio de la brecha. Si se
    solapan —que es lo que pasa con un lote suficientemente grande— no existe un
    corte perfecto y el umbral pasa a ser el que MENOS errores comete. Si varios
    empatan se toma el del centro del tramo, para no quedar pegado a una consulta:
    con la variación de milésimas de la API, un umbral al borde cambia de veredicto
    de una corrida a otra.
    """
    coleccion = obtener_coleccion()
    presentes = set(coleccion.get(include=[])["ids"])
    dentro_validas = [(c, d) for c, d in CALIBRACION_DENTRO if d in presentes]
    if len(dentro_validas) < len(CALIBRACION_DENTRO):
        print("  (se omiten consultas cuyo documento no está en la colección: "
              "corré etl_purga.py para incluirlas)")

    print(f"[C.2] Calibración sobre {coleccion.count()} registros · umbral actual {UMBRAL_DISTANCIA}\n")
    print(f"=== DENTRO del catálogo ({len(dentro_validas)}) ===")
    dentro = []
    for consulta, esperado in dentro_validas:
        doc, dist = _top1(consulta)
        dentro.append(dist)
        nota = "" if doc == esperado else f"   <- esperaba {esperado}"
        print(f"  {dist:.4f}  {doc}  {consulta}{nota}")

    print(f"\n=== FUERA del catálogo ({len(CALIBRACION_FUERA)}) ===")
    fuera = []
    for consulta in CALIBRACION_FUERA:
        doc, dist = _top1(consulta)
        fuera.append(dist)
        print(f"  {dist:.4f}  {doc}  {consulta}")

    peor_dentro, mejor_fuera = max(dentro), min(fuera)
    print("\n=== Distribución ===")
    print(f"  dentro : {min(dentro):.4f} – {peor_dentro:.4f}")
    print(f"  fuera  : {mejor_fuera:.4f} – {max(fuera):.4f}")
    print(f"  brecha : {mejor_fuera - peor_dentro:+.4f}"
          + ("   (solapamiento: ningún umbral separa perfecto)" if mejor_fuera <= peor_dentro else ""))

    print("\n=== Barrido de umbrales ===")
    print("  umbral  falsos rechazos  falsas aceptaciones  errores")
    candidatos = []
    for milesimas in range(280, 401, 5):
        umbral = milesimas / 1000
        fr = sum(d > umbral for d in dentro)
        fa = sum(d <= umbral for d in fuera)
        candidatos.append((fr + fa, umbral, fr, fa))
        actual = "   <- actual" if abs(umbral - UMBRAL_DISTANCIA) < 1e-9 else ""
        print(f"  {umbral:.3f}  {fr:>15}  {fa:>19}  {fr + fa:>7}{actual}")

    minimo = min(c[0] for c in candidatos)
    empatados = [c for c in candidatos if c[0] == minimo]
    errores, umbral, fr, fa = empatados[len(empatados) // 2]  # centro del tramo
    total = len(dentro) + len(fuera)
    print(f"\n  Mejor umbral: {umbral:.3f} -> {errores} errores sobre {total} consultas "
          f"({fr} falsos rechazos, {fa} falsas aceptaciones), "
          f"acierto {100 * (total - errores) / total:.1f} %")


def main() -> int:
    parser = argparse.ArgumentParser(description="ChromaDB y búsqueda híbrida (B.1–B.4, B.6).")
    parser.add_argument("--ingesta", action="store_true", help="B.1: carga base_conocimiento.json con upsert.")
    parser.add_argument("--evento", action="store_true", help="B.3: evento en caliente + get de verificación.")
    parser.add_argument("--buscar", metavar="CONSULTA", help="B.4: búsqueda híbrida.")
    parser.add_argument("--categoria", help="filtro duro por categoría.")
    parser.add_argument("--sucursal", help="filtro duro por sucursal (SUC-001/002/003); suma las de red.")
    parser.add_argument("--intencion", help="filtro duro por intencion_relacionada.")
    parser.add_argument("--n", type=int, default=3, help="cantidad de resultados (default 3).")
    parser.add_argument("--incluir-no-vigentes", action="store_true", help="desactiva el filtro de vigencia.")
    parser.add_argument("--killer", action="store_true", help="B.6: las tres killer queries.")
    parser.add_argument("--calibrar", action="store_true", help="C.2: calibración del umbral.")
    args = parser.parse_args()

    if args.ingesta:
        ingestar()
    if args.evento:
        evento_en_caliente()
    if args.buscar:
        imprimir(buscar_farmacia(args.buscar, args.categoria,
                                 not args.incluir_no_vigentes, args.n,
                                 args.sucursal, args.intencion))
    if args.killer:
        killer_queries()
    if args.calibrar:
        calibrar_umbral()
    if not (args.ingesta or args.evento or args.buscar or args.killer or args.calibrar):
        parser.print_help()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
