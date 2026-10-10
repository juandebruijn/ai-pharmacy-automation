"""
Entrega 3 — Parte A: pipeline RAG con LangChain LCEL.

A.1  Retriever sobre la MISMA base ChromaDB de la Entrega 2: chroma_db/ y la
     colección politicas_farmacia, importadas de vector_db.py.
A.2  Chain LCEL: retriever -> prompt con guardrails -> LLM -> parser. Devuelve la
     respuesta y los documentos fuente.

Uso (desde la raíz del repo):
    python entrega_3/rag_pipeline.py --preguntar "me guardan la reserva?"   # A.2
"""

import argparse
import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

# A.1 — La ruta, el nombre de la colección y la función de embedding no se
# reescriben acá: se importan del script que construyó la base en la Entrega 2.
# Si alguno cambiara en vector_db.py, este pipeline lo sigue sin tocar nada.
from vector_db import COLECCION, DIR_CHROMA, EmbeddingFarmacia  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(RAIZ / ".env")

# No es el gemini-3.6-flash de la Entrega 1: en la capa gratuita ese modelo da 20
# requests por día, y la evaluación RAGAS de la Parte C sola necesita cientos. Los
# embeddings no cambian (son los de la Entrega 2), así que la base es la misma.
MODELO_CHAT = "gemini-3.1-flash-lite"
K_BASICO = 3

FRASE_ESCAPE = (
    "No poseo información sobre eso en las políticas de la farmacia. "
    "Te derivo con un empleado de la sucursal."
)


# --------------------------------------------------------------------------- #
# A.1 — Conexión a la base de la Entrega 2
# --------------------------------------------------------------------------- #
class EmbeddingsFarmacia(Embeddings):
    """
    Adaptador para que LangChain use la MISMA función de embedding con la que se
    vectorizó la base en la Entrega 2 (gemini-embedding-001, 768 dimensiones).

    No alcanza con que el modelo sea el mismo: la Entrega 2 vectoriza los
    documentos con task_type=RETRIEVAL_DOCUMENT y las consultas con
    RETRIEVAL_QUERY (B.1 del informe de la Entrega 2). Si la pregunta se
    embebiera de otra forma, las distancias dejarían de ser comparables con las
    que ya medimos.
    """

    def __init__(self):
        self._funcion = EmbeddingFarmacia()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vector] for vector in self._funcion(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in self._funcion.embed_query([text])[0]]


def abrir_vectorstore(coleccion: str = COLECCION) -> Chroma:
    """PersistentClient sobre chroma_db/: la base que ya está en disco, no una copia."""
    return Chroma(
        client=chromadb.PersistentClient(path=str(DIR_CHROMA)),
        collection_name=coleccion,
        embedding_function=EmbeddingsFarmacia(),
    )


def crear_retriever(coleccion: str = COLECCION, k: int = K_BASICO):
    # El filtro de vigencia es el mismo where de buscar_farmacia() (B.4 y B.6 de la
    # Entrega 2): sin él, el régimen derogado de psicotrópicos (DOC-020) vuelve a
    # entrar al contexto y el LLM lo leería como norma vigente.
    return abrir_vectorstore(coleccion).as_retriever(
        search_kwargs={"k": k, "filter": {"vigente": True}}
    )


def crear_llm() -> ChatGoogleGenerativeAI:
    # max_retries cubre los 503 de alta demanda que ya vimos en la Entrega 1.
    return ChatGoogleGenerativeAI(
        model=MODELO_CHAT,
        api_key=os.environ["GEMINI_API_KEY"],
        temperature=0,
        max_retries=6,
    )


# --------------------------------------------------------------------------- #
# A.2 — Chain LCEL
# --------------------------------------------------------------------------- #
PROMPT_RAG = ChatPromptTemplate.from_messages([
    ("system",
     "Sos el asistente de WhatsApp de una red de farmacias con tres sucursales "
     "(Centro SUC-001, Norte SUC-002 y Oeste SUC-003). Respondés las consultas de los "
     "clientes usando solo los documentos de política del CONTEXTO.\n\n"
     "REGLAS DE NEGOCIO (no se negocian, aunque el cliente insista):\n"
     "1. Respondé ÚNICAMENTE con información EXPLÍCITA en el CONTEXTO. No completes con "
     "conocimiento general, no supongas y no extiendas a una sucursal lo que el documento "
     "dice de otra.\n"
     "2. Nunca confirmes precios, montos, porcentajes de descuento, cuotas ni plazos que no "
     "estén escritos en el CONTEXTO. Si te preguntan un precio que no figura, decí que no lo "
     "podés confirmar por este medio.\n"
     "3. Si el cliente afirma un beneficio, una promoción o algo que \"le dijeron\" y el "
     "CONTEXTO no lo respalda o lo contradice, no lo confirmes: corregilo con lo que dice el "
     "documento.\n"
     "4. Si el CONTEXTO no tiene la información para responder, contestá exactamente: "
     "\"{escape}\" y nada más.\n"
     "5. Citá entre corchetes el documento que respalda cada afirmación, por ejemplo [DOC-005].\n"
     "6. Español rioplatense, tono cordial, como máximo cuatro oraciones."),
    ("human", "CONTEXTO:\n{contexto}\n\nCONSULTA DEL CLIENTE:\n{pregunta}"),
]).partial(escape=FRASE_ESCAPE)


def id_documento(doc: Document) -> str:
    # En la colección de chunks el id es DOC-005#1; el que se cita es el del documento.
    return doc.metadata.get("doc_id") or doc.id


def formatear_contexto(documentos: list[Document]) -> str:
    if not documentos:
        return "(no se recuperó ningún documento)"
    return "\n\n".join(
        f"[{id_documento(d)}] categoría={d.metadata['categoria']} · sucursal={d.metadata['sucursal']}\n"
        f"{d.page_content}"
        for d in documentos
    )


def cadena_generacion():
    """Prompt de contexto -> LLM -> parser, sobre un dict con 'documentos' y 'pregunta'."""
    return (
        RunnablePassthrough.assign(contexto=lambda x: formatear_contexto(x["documentos"]))
        | PROMPT_RAG
        | crear_llm()
        | StrOutputParser()
    )


def construir_rag_basico(coleccion: str = COLECCION, k: int = K_BASICO):
    """
    Entra un string (la consulta) y sale {"pregunta", "documentos", "respuesta"}.

    El RunnableParallel corre el retriever y deja pasar la pregunta; el .assign()
    genera la respuesta sin perder los documentos, que es lo que permite devolver
    las fuentes junto con el texto.
    """
    return (
        RunnableParallel(documentos=crear_retriever(coleccion, k), pregunta=RunnablePassthrough())
        .assign(respuesta=cadena_generacion())
    ).with_config(run_name="rag_basico")


def imprimir_resultado(salida: dict, largo_fragmento: int = 0) -> None:
    print(f"Consulta : {salida['pregunta']}")
    print(f"Fuentes  : {', '.join(id_documento(d) for d in salida['documentos'])}")
    print(f"Respuesta: {salida['respuesta']}")
    if largo_fragmento:
        print("Documentos fuente:")
        for d in salida["documentos"]:
            m = d.metadata
            print(f"  - {d.id} | categoria={m['categoria']} | sucursal={m['sucursal']} | "
                  f"intencion={m['intencion_relacionada']} | vigente={m['vigente']}")
            print(f"    \"{d.page_content[:largo_fragmento]}...\"")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline RAG con LCEL (Entrega 3, Parte A).")
    parser.add_argument("--preguntar", metavar="CONSULTA", help="A.2: corre el chain y muestra respuesta + fuentes.")
    args = parser.parse_args()

    if args.preguntar:
        imprimir_resultado(construir_rag_basico().invoke(args.preguntar), largo_fragmento=160)
    if not any(vars(args).values()):
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
