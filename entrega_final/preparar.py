"""B.3 — Deja todo listo para levantar la API en una máquina recién clonada.

chroma_db/ y las bases SQLite no se commitean (.gitignore): se reconstruyen
desde los JSON del repo, con los mismos scripts de las Entregas 2 y 3.

    1. Colección politicas_farmacia        vector_db.ingestar()           (E2, B.1)
    2. ETL + purga semántica               etl_purga.main()               (E2, B.5)
    3. Colección de chunks 500/100         reindexar_con_chunks()         (E3, B.2)
    4. ERP simulado (productos, reservas)  erp.inicializar()

Solo gasta embeddings (gemini-embedding-001): ninguna llamada al modelo de chat.
Es idempotente: se puede correr dos veces sin duplicar nada.

Uso (desde la raíz del repo):
    python -m entrega_final.preparar
"""

import contextlib
import io
import os
import sys
import time

from entrega_final.config import DB_ERP, RAIZ

sys.path.insert(0, str(RAIZ))


def main() -> int:
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("Falta GEMINI_API_KEY en el .env (ver .env.example).")

    import etl_purga
    import vector_db
    from entrega_3.rag_pipeline import COLECCION_CHUNKS, reindexar_con_chunks
    from entrega_final import erp

    pasos = [
        ("1/4 Base vectorial de la Entrega 2", vector_db.ingestar),
        ("2/4 ETL y purga semántica", etl_purga.main),
        ("3/4 Chunking 500/100 de la Entrega 3", reindexar_con_chunks),
    ]
    for titulo, paso in pasos:
        print(f"{titulo}...", flush=True)
        for intento in range(1, 5):
            try:
                # La salida detallada de cada script ya está documentada en su entrega.
                with contextlib.redirect_stdout(io.StringIO()):
                    paso()
                break
            except Exception as exc:  # chromadb lo envuelve en ValueError; google-genai, en ClientError
                # Capa gratuita: 100 requests de embedding por minuto, y cada texto es
                # una request. Los tres pasos son idempotentes (upsert / recrear la
                # colección), así que alcanza con esperar a la ventana siguiente y repetir.
                if "RESOURCE_EXHAUSTED" not in str(exc) or intento == 4:
                    raise
                print(f"    cuota de embeddings por minuto agotada; reintento {intento}/3 en 65 s",
                      flush=True)
                time.sleep(65)

    filas = erp.inicializar()
    print(f"4/4 ERP simulado: {filas} filas en productos ({DB_ERP.name})")

    import chromadb
    from vector_db import COLECCION, DIR_CHROMA
    cliente = chromadb.PersistentClient(path=str(DIR_CHROMA))
    for nombre in (COLECCION, COLECCION_CHUNKS):
        print(f"    {nombre}: {cliente.get_collection(nombre).count()} registros")
    print("\nListo. Levantar la API con:  uvicorn entrega_final.api:app --port 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
