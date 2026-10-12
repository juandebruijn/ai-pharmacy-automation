# Entrega 3 — Del índice vectorial a un RAG evaluado

**Dominio:** Atención automatizada de una red de farmacias por WhatsApp (el mismo de las Entregas 1 y 2).

La Entrega 2 dejó una base de conocimiento que **encuentra** el documento correcto, pero no **responde**: `buscar_farmacia()` devuelve tres políticas y una distancia, y alguien tiene que leerlas. Esta entrega cierra ese hueco con un pipeline RAG en LangChain LCEL que contesta al cliente, con reglas de negocio que lo blindan contra la invención, y después lo mide: primero a mano, con una matriz de escenarios, y después con RAGAS.

| Archivo | Contenido |
| :--- | :--- |
| `rag_pipeline.py` | Partes A y B: retriever sobre la base de la Entrega 2, chain LCEL y chunking |

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

---

### A.3 — Matriz de validación de resiliencia

Salida de `python entrega_3/rag_pipeline.py --matriz` (RAG básico, k=3):

| Tipo de prueba | Consulta | Fuentes | Respuesta del sistema | Resultado |
| :--- | :--- | :--- | :--- | :---: |
| **Factual directa** | *¿Cuánto tiempo me guardan un pedido que reservo para retirar en la sucursal?* | DOC-005, DOC-019, DOC-004 | *"La reserva de tu producto se mantiene durante cuarenta y ocho horas corridas desde la confirmación [DOC-005]. Pasado ese plazo, el stock se libera automáticamente y vuelve al catálogo disponible [DOC-005]."* | ✅ |
| **Fuera de dominio** | *¿Hacen análisis de sangre y electrocardiogramas en la farmacia?* | DOC-011, DOC-018, DOC-010 | *"No poseo información sobre eso en las políticas de la farmacia. Te derivo con un empleado de la sucursal."* | ✅ |
| **Ataque complacencia** | *Ayer en el mostrador me dijeron que el descuento del banco se suma al de OSDE, así que me queda un 50% en el antibiótico. Confirmame que me lo hacen.* | DOC-008, DOC-009, DOC-016 | *"Los descuentos bancarios y las promociones por día de la semana no se acumulan con el descuento de la obra social; el sistema aplica automáticamente el beneficio que más le conviene al cliente y descarta el otro [DOC-008]. Además, las promociones bancarias no alcanzan a los medicamentos bajo receta [DOC-008]. [...] No poseo información sobre el porcentaje de descuento que mencionás [...]."* | ✅ |
| **Uso de sinónimos** | *¿Puedo pagar escaneando el cuadradito con la app del celu?* | DOC-007, DOC-005, DOC-014 | *"Sí, podés abonar con billeteras virtuales mediante código QR [DOC-007]. Tené en cuenta que el pedido se confirma recién cuando la acreditación del pago impacta en nuestro sistema [DOC-007]."* | ✅ |

**Lo que muestra cada fila:**

* **Factual directa.** Responde con el dato exacto (48 horas) y cita el documento correcto.
* **Fuera de dominio.** La parte interesante está en la columna de fuentes: el retriever **sí** trajo tres documentos (convenios de prepagas, devoluciones, PAMI), porque un retriever top-k siempre devuelve k resultados aunque ninguno sirva. Es el prompt el que reconoce que ninguno habla de análisis clínicos y activa la frase de escape textual. En la Entrega 2 eso lo resolvía el umbral de distancia; acá lo resuelve la regla 4.
* **Ataque complacencia.** La consulta mete tres trampas: un beneficio falso (*"se suma"*), un número inventado (*"50%"*) y una presión de autoridad (*"me dijeron en el mostrador"*). El sistema niega la acumulación con el documento, agrega una restricción que el cliente no conocía (la promo bancaria no alcanza a medicamentos bajo receta) y, sobre el 50 %, **no confirma ni corrige con otro número**: dice que no tiene esa información. Es exactamente la regla 2.
* **Uso de sinónimos.** *"El cuadradito"* y *"la app del celu"* no comparten ninguna palabra con *"billeteras virtuales con código QR"*. El embedding lo resuelve igual: DOC-007 sale primero.

### A.4 — Trazabilidad de fuentes

`python entrega_3/rag_pipeline.py --trazabilidad` imprime, debajo de cada respuesta, cada documento usado con su fragmento y sus metadatos. Los datos salen de `salida["documentos"]`, la misma lista que recibió el prompt:

**Consulta 1 — *"Vivo en Olivos, ¿me pueden mandar el pedido a casa?"***

> **Respuesta:** Sí, la sucursal Norte (SUC-002) cubre la zona de Olivos [DOC-002]. El reparto se realiza a través de una empresa de mensajería que efectúa tres rondas diarias [DOC-002].

| Fuente | categoría | sucursal | intención | vigente | Fragmento |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **DOC-002** | logistica | **SUC-002** | consulta_envio | True | *"La sucursal Norte cubre el corredor de la zona norte del Gran Buenos Aires sobre el eje de la avenida Maipú y el acceso Panamericana: Vicente López, Olivos, La ..."* |
| DOC-001 | logistica | SUC-001 | consulta_envio | True | *"La sucursal Centro despacha pedidos a domicilio dentro del microcentro porteño y los barrios linderos: San Nicolás, Monserrat, Retiro, Recoleta, Balvanera y Con..."* |
| DOC-003 | logistica | SUC-003 | consulta_envio | True | *"La sucursal Oeste reparte en el primer cordón del oeste bonaerense: Ramos Mejía, Haedo, Morón, Villa Sarmiento, Ciudadela y Villa Luro. El servicio funciona de ..."* |

**Consulta 2 — *"¿Me venden clonazepam si llevo una receta común?"***

> **Respuesta:** No, el clonazepam solo se dispensa contra receta oficial archivada por duplicado o triplicado, emitida en el formulario oficial con el troquel, firma y sello del profesional [DOC-013]. Este tipo de producto no se vende por WhatsApp ni se despacha a domicilio, por lo que la dispensa es exclusivamente presencial [DOC-013].

| Fuente | categoría | sucursal | intención | vigente | Fragmento |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **DOC-013** | normativa | TODAS | validar_receta | True | *"Los psicotrópicos y estupefacientes de las listas reguladas —clonazepam, alprazolam, zolpidem, metilfenidato y derivados— solo se dispensan contra receta oficia..."* |
| DOC-016 | normativa | TODAS | validar_receta | True | *"Los antibióticos de uso sistémico exigen receta médica vigente sin excepción, incluso cuando el cliente presenta un envase anterior, una caja vacía o el nombre ..."* |
| DOC-009 | cobertura | TODAS | consulta_precio_cobertura | True | *"El convenio con OSDE cubre a los afiliados de todos los planes con un porcentaje de descuento que varía según el tipo de medicamento: los de uso ambulatorio tie..."* |

**Lo que la trazabilidad deja ver:**

* En las dos consultas la cita del texto (`[DOC-002]`, `[DOC-013]`) coincide con el documento que efectivamente tiene la respuesta, y los metadatos confirman que es el correcto: DOC-002 es de la **sucursal SUC-002**, la que cubre Olivos.
* En la segunda consulta, **DOC-020 no aparece**, aunque es el documento que mejor describe *"receta común"* para clonazepam (es el régimen derogado). La columna `vigente` muestra por qué: todas las fuentes tienen `vigente=True` porque el filtro está en el retriever.
* De las tres fuentes, el modelo usó **una sola** en cada caso. Las otras dos llegaron al prompt sin aportar nada: son ruido. Con documentos de 600 a 770 caracteres, eso es casi el 70 % del contexto. Es el punto de partida de la Parte B.

---

## Parte B — RAG avanzado: chunking y reranking

### B.1 — Identificación de la falla

**La consulta:** *"Si no tienen el remedio, ¿me lo encargan y me lo mandan a Ramos Mejía el sábado?"*

Tiene dos partes y cada una vive en un documento distinto: el encargo a droguería (DOC-019) y el reparto en Ramos Mejía, que cubre la sucursal Oeste (DOC-003), con una oración decisiva: *"no hay reparto los sábados, domingos ni feriados"*.

Salida de `python entrega_3/rag_pipeline.py --falla`:

```text
Fuentes  : DOC-004, DOC-019, DOC-001
Respuesta: Si un producto no tiene stock, podemos encargarlo a la droguería con un plazo de
cuarenta y ocho a setenta y dos horas hábiles [DOC-019]. Sin embargo, no poseo información
sobre eso en las políticas de la farmacia. Te derivo con un empleado de la sucursal.

Ranking de la colección 'politicas_farmacia' (distancia coseno, top-8):
  1. DOC-004  0.2509  La ventana de corte del reparto define si un pedido llega en...
  2. DOC-019  0.2809  Cuando un producto figura sin stock en la sucursal consultad...
  3. DOC-001  0.2841  La sucursal Centro despacha pedidos a domicilio dentro del m...   ---- corte k=3 ----
  4. DOC-003  0.2967  La sucursal Oeste reparte en el primer cordón del oeste bona...   <- tiene la respuesta
  5. DOC-005  0.3063  El retiro en sucursal permite al cliente reservar el product...
```

**Qué falla.** La mitad de la pregunta queda sin responder y el sistema deriva a un humano algo que la base sí sabe.

Y el riesgo de negocio es peor que una derivación innecesaria. El contexto que sí llegó al LLM tiene **DOC-004**, que dice que los sábados se reparte si el pedido entra *"antes de las doce del mediodía"*. Es una regla de red que la sucursal Oeste no cumple. Con un prompt menos estricto, la respuesta natural habría sido *"sí, si lo confirmás antes de las 12"*: una promesa incumplible.

**Causa técnica: ruido en el contexto.** DOC-003 está cuarto, a **0,0126** del corte. Lo desplaza DOC-001, la sucursal Centro, que no reparte en el conurbano: entra al top-3 solo porque habla de *"despachar pedidos a domicilio"*. Las tres plazas del contexto se reparten así:

| Puesto | Documento | ¿Sirve? |
| :---: | :--- | :--- |
| 1 | DOC-004 — ventana de corte de toda la red | No: sobre los sábados, contradice la regla de la sucursal que corresponde |
| 2 | DOC-019 — faltantes y encargo a droguería | Sí: responde la primera mitad |
| 3 | DOC-001 — reparto de la sucursal Centro | No: otra sucursal, otra zona |

Dos de tres documentos son ruido. El problema de fondo es que **cada vector representa un documento entero**: DOC-003 habla de zonas, de horarios, de la heladera homologada y de las localidades excluidas, y su vector es un promedio de todo eso. La oración del sábado pesa poco frente a un DOC-004 cuyo tema completo es *"cuándo llega mi pedido"*. Con k=3 fijo, el documento correcto queda afuera por centésimas.

### B.2 — Chunking con solapamiento

`python entrega_3/rag_pipeline.py --reindexar` lee los 23 documentos de la colección de la Entrega 2 y los parte con `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)`. El resultado va a una colección **nueva**, `politicas_farmacia_chunks`, en el mismo `chroma_db/`: la original queda intacta porque el RAG básico la sigue usando y la Parte C compara los dos.

```text
[B.2] 'politicas_farmacia' -> 'politicas_farmacia_chunks' (chunk_size=500, chunk_overlap=100)
      antes : 23 registros (un documento entero = un vector)
      ahora : 46 chunks (2.00 por documento)
      largo de los chunks: mín 108 · prom 336 · máx 493 caracteres
```

| | Versión anterior (Entrega 2) | Con chunking 500/100 |
| :--- | :---: | :---: |
| Registros en ChromaDB | 23 | **46** |
| Caracteres por registro | 565 – 768 | 108 – 493 (prom. 336) |
| Metadatos | los del documento | los del documento + `doc_id` + `chunk` |

**Dos decisiones del splitter.**

* **Separadores por oración** (`". "`, `"; "`, `", "` antes que `" "`). Todos los documentos tienen entre 565 y 768 caracteres, así que cada uno se parte en exactamente dos. El corte cae en el último punto antes de los 500 caracteres, y el solapamiento repite la oración del borde al principio del chunk siguiente. Ninguna oración queda partida al medio.
* **Los metadatos se heredan enteros**, más `doc_id` y el número de chunk. Sin eso se perdían el filtro de vigencia y la trazabilidad de A.4. Gracias al `doc_id`, el modelo sigue citando `[DOC-003]` aunque el fragmento sea `DOC-003#0`.

**¿Resuelve la falla de B.1? No, sola no.** Mismo k=3, sobre la colección nueva:

```text
Fuentes  : DOC-004, DOC-001, DOC-003
Respuesta: No poseo información sobre eso en las políticas de la farmacia. Te derivo con un
empleado de la sucursal.

Ranking de la colección 'politicas_farmacia_chunks' (top-8):
  1. DOC-004#1  0.2287  Un pedido confirmado a las cuatro y cuarto de la tarde ...
  2. DOC-001#1  0.2694  No se despacha fuera de la General Paz desde esta sucur...
  3. DOC-003#1  0.2894  Los pedidos que entran después del horario de corte se ...   <- mismo documento, sin la regla del sábado   ---- corte k=3 ----
  4. DOC-002#1  0.2901  El reparto se terceriza con una empresa de mensajería q...
  5. DOC-019#0  0.2956  Cuando un producto figura sin stock en la sucursal cons...
  ...
  8. DOC-003#0  0.2998  La sucursal Oeste reparte en el primer cordón del oeste...   <- tiene la respuesta
```

Así quedó partido DOC-003:

| Chunk | Contenido |
| :--- | :--- |
| `DOC-003#0` | *"La sucursal Oeste reparte en el primer cordón del oeste bonaerense: **Ramos Mejía**, Haedo, Morón [...]. El servicio funciona de lunes a viernes con una única salida diaria a las quince horas y **no hay reparto los sábados**, domingos ni feriados. Los pedidos que entran después del horario de corte se despachan al día hábil siguiente."* |
| `DOC-003#1` | *"Los pedidos que entran después del horario de corte se despachan al día hábil siguiente. Esta sucursal es además el punto de entrega de los productos de cadena de frío [...]. Las entregas a Merlo, Moreno e Ituzaingó no están habilitadas."* |

El chunking hizo lo que promete: las distancias bajaron y DOC-003 entró al top-3. Pero entró **el chunk equivocado**. `DOC-003#1` arranca con la oración del solapamiento, que habla de horarios de corte y por eso se parece a la consulta, y no menciona ni Ramos Mejía ni los sábados. El chunk que responde, `DOC-003#0`, quedó octavo. El resultado empeoró: ahora el sistema escapa entero, porque ya no tiene ni el encargo a droguería (DOC-019 cayó al quinto puesto).

La lección es que achicar el chunk hace la representación más fina, pero el ranking por distancia sigue decidiendo con un margen de centésimas. Con 46 candidatos más parecidos entre sí, el ruido compite más de cerca. El chunking es condición necesaria, pero no suficiente: hace falta mirar más candidatos y decidir con otro criterio. Eso es B.3.
