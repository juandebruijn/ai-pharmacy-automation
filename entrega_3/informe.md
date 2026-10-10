# Entrega 3 — Del índice vectorial a un RAG evaluado

**Dominio:** Atención automatizada de una red de farmacias por WhatsApp (el mismo de las Entregas 1 y 2).

La Entrega 2 dejó una base de conocimiento que **encuentra** el documento correcto, pero no **responde**: `buscar_farmacia()` devuelve tres políticas y una distancia, y alguien tiene que leerlas. Esta entrega cierra ese hueco con un pipeline RAG en LangChain LCEL que contesta al cliente, con reglas de negocio que lo blindan contra la invención, y después lo mide: primero a mano, con una matriz de escenarios, y después con RAGAS.

| Archivo | Contenido |
| :--- | :--- |
| `rag_pipeline.py` | Parte A: retriever sobre la base de la Entrega 2 y chain LCEL |

> **Sobre el modelo.** La Entrega 1 usó `gemini-3.6-flash`, pero en la capa gratuita ese modelo da **20 requests por día** y RAGAS solo necesita cientos. El RAG genera con `gemini-3.1-flash-lite`. Los embeddings **no cambian** (`gemini-embedding-001`, 768 dimensiones): la base es exactamente la de la Entrega 2.

---

## Parte A — Pipeline RAG con LangChain LCEL

### A.1 — Conexión a la base de la Entrega 2

El retriever no reconstruye nada: abre `chroma_db/` con un `PersistentClient` y apunta a la colección `politicas_farmacia`, con sus 23 registros (los 20 documentos canónicos más los 3 que sumó el ETL de B.5). La ruta, el nombre de la colección y la función de embedding **no están reescritos** en `rag_pipeline.py`: se importan de `vector_db.py`, el script que construyó la base.

```python
from vector_db import COLECCION, DIR_CHROMA, EmbeddingFarmacia

def abrir_vectorstore(coleccion: str = COLECCION) -> Chroma:
    return Chroma(
        client=chromadb.PersistentClient(path=str(DIR_CHROMA)),
        collection_name=coleccion,
        embedding_function=EmbeddingsFarmacia(),
    )
```

Si alguno de esos tres valores cambiara en la Entrega 2, el pipeline lo sigue sin que nadie tenga que acordarse de copiarlo.

**Por qué un adaptador y no `GoogleGenerativeAIEmbeddings`.** No alcanza con usar el mismo modelo. En B.1 de la Entrega 2 medimos que la base solo separa bien lo que está dentro del catálogo de lo que no cuando los documentos se vectorizan con `task_type=RETRIEVAL_DOCUMENT` y las consultas con `RETRIEVAL_QUERY`. `EmbeddingsFarmacia` expone a LangChain la misma `EmbeddingFarmacia` de `vector_db.py`, así que la pregunta se embebe exactamente igual que en la Entrega 2. La prueba: la consulta *"me mandan algo a Recoleta?"* devuelve el mismo top-3 con las mismas distancias por los dos caminos: DOC-001 **0,2638**, DOC-002 **0,3097** y DOC-004 **0,3326**, tanto desde `buscar_farmacia()` de la Entrega 2 como desde el retriever de LangChain.

**El filtro de vigencia viaja con el retriever.** `as_retriever(search_kwargs={"k": 3, "filter": {"vigente": True}})` es el mismo `where` de la Killer Query 2: sin él, el régimen derogado de psicotrópicos (DOC-020) vuelve a entrar al contexto, y un LLM no tiene forma de saber que un párrafo que dice *"alcanzaba con una receta manuscrita"* ya no rige.

### A.2 — Chain LCEL completo

```python
def cadena_generacion():
    return (
        RunnablePassthrough.assign(contexto=lambda x: formatear_contexto(x["documentos"]))
        | PROMPT_RAG          # 2. prompt de contexto con las reglas de negocio
        | crear_llm()         # 3. LLM
        | StrOutputParser()   # 4. parser
    )

def construir_rag_basico(coleccion=COLECCION, k=K_BASICO):
    return (
        RunnableParallel(documentos=crear_retriever(coleccion, k),   # 1. retriever
                         pregunta=RunnablePassthrough())
        .assign(respuesta=cadena_generacion())
    )
```

El `RunnableParallel` corre el retriever y deja pasar la pregunta; el `.assign()` agrega la respuesta **sin pisar** los documentos. Por eso el chain devuelve `{"pregunta", "documentos", "respuesta"}`: la respuesta generada y los documentos fuente que la fundamentan, en la misma salida.

Cada documento entra al prompt con su ID y sus metadatos (`[DOC-005] categoría=logistica · sucursal=TODAS`), para que el modelo pueda citarlo y para que no extienda a una sucursal lo que el documento dice de otra.

**Las reglas de negocio del prompt.** Son las que en la Entrega 1 hacían falta y no podían aplicarse, porque el modelo no tenía las políticas:

| Regla | Por qué |
| :--- | :--- |
| Responder **ÚNICAMENTE** con información **EXPLÍCITA** en el contexto; no extender a una sucursal lo que se dice de otra | La alucinación asertiva de A.2 de la Entrega 1: el modelo completa con conocimiento general y suena seguro |
| **Nunca confirmar precios**, montos, porcentajes, cuotas ni plazos que no estén escritos | El precio vive en la tabla `productos` (Entrega 1), no en la base vectorial. Un precio inventado es una venta con reclamo |
| Si el cliente afirma un beneficio que *"le dijeron"*, no confirmarlo: corregirlo con el documento | El ataque de complacencia: el modelo tiende a darle la razón al cliente |
| Frase de escape fija: *"No poseo información sobre eso en las políticas de la farmacia. Te derivo con un empleado de la sucursal."* | Una respuesta idéntica cada vez se puede detectar en código (Parte C la usa) y deriva a un humano, igual que el `fuera_de_alcance` de la Entrega 1 |
| Citar el documento entre corchetes, `[DOC-005]` | Trazabilidad: cada afirmación se puede verificar contra la fuente |

```bash
python entrega_3/rag_pipeline.py --preguntar "¿Cuánto tiempo me guardan la reserva?"
```
