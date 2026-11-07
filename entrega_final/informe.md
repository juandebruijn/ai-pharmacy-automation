# Entrega Final — Del RAG a un agente en producción

**Dominio:** atención automatizada de una red de farmacias por WhatsApp (el mismo de las Entregas 1, 2 y 3).

La Entrega 3 dejó un RAG que **contesta** sobre las políticas, pero que no puede hacer nada más: no sabe si hay stock, no puede reservar y se olvida de la conversación en cuanto termina la consulta. Esta entrega lo convierte en un **agente** que decide solo qué herramientas usar, recuerda la conversación aunque se reinicie el servidor, y queda expuesto como una **API REST** con logs, métricas, trazas y guardrails.

> **Sobre el modelo.** El agente usa `gemini-3.1-flash-lite`, el mismo modelo de generación de la Entrega 3, en la capa gratuita de Gemini. Cada consulta al agente son entre 1 y 3 llamadas al LLM (decidir → observar → responder), así que un modelo con más cuota diaria pesa más que uno levemente mejor. Los embeddings no cambian: es la base de la Entrega 2, con los chunks 500/100 de la Entrega 3.

---

## Parte A — El agente

### A.1 — Herramientas

| Herramienta | Rol | Fuente | Efecto |
| :--- | :--- | :--- | :--- |
| `buscar_politicas(consulta)` | **Retriever** (el RAG de la E3) | ChromaDB: colección de chunks 500/100, k=4, filtro `vigente=True` | Solo lectura |
| `consultar_stock(producto, sucursal_id?)` | **Sensor** | ERP: tabla `productos`, por sucursal | Solo lectura |
| `reservar_producto(producto_id, cantidad)` | **Actuador** | ERP: `UPDATE productos` + `INSERT reservas` | **Bloquea stock 48 h** |
| `consultar_reserva(codigo?)` | Sensor | ERP: tabla `reservas` | Solo lectura (libera las vencidas) |

**El retriever es el de la Entrega 3, importado y no reescrito.** `buscar_politicas` llama a `crear_retriever()` de `entrega_3/rag_pipeline.py`: misma base, misma función de embedding con `RETRIEVAL_QUERY`, mismo filtro de vigencia (sin él vuelve el régimen derogado de psicotrópicos, DOC-020). Usa los chunks 500/100 de B.2, que en la E3 resolvieron la falla de Ramos Mejía; si la colección de chunks no existe, cae a la de documentos enteros. No usa el reranking con LLM-juez de B.3: en un agente, cada llamada extra al modelo se multiplica por cada vuelta del bucle, y el modelo del agente ya hace de juez cuando lee los fragmentos.

**Sensor y actuador son los del PEAS de la Entrega 1** (A.3): *"datos de confirmación de stock y precios enviados por las APIs internas"* y *"bloquear/reservar stock temporalmente en la base de datos"*. El ERP (`erp.py`) es SQLite con las mismas columnas que la tabla `productos` del esquema de la Entrega 1 (B.5.b), incluido `stock_reservado`, y lo siembra `data/catalogo.json`: 9 presentaciones × 3 sucursales.

**Los docstrings son la interfaz.** El LLM decide qué herramienta usar leyendo solo el nombre, el docstring y la firma (`bind_tools` los manda como JSON Schema). Por eso cada docstring tiene cuatro partes: **cuándo usarla**, **cuándo NO usarla** (y cuál usar en su lugar), **formato de cada argumento** con ejemplos, y **qué devuelve**, incluidos los errores. Dos decisiones de firma:

* `sucursal_id` es un `Literal["SUC-001", "SUC-002", "SUC-003"]`: el modelo no puede inventar una sucursal.
* `reservar_producto` y `consultar_reserva` reciben el `thread_id` por `RunnableConfig`, que LangChain **inyecta y saca del schema**. El modelo no lo ve ni lo puede falsificar: una conversación solo ve y crea sus propias reservas, sin pedirle al cliente un dato personal para identificarlo.

**Las reglas duras viven en el actuador, no en el prompt.** `erp.crear_reserva()` rechaza psicotrópicos (DOC-013), cantidades fuera de 1–20 (tope minorista de `schemas.py`, E1) y pedidos sin stock. El descuento de stock es un `UPDATE ... WHERE stock_disponible - stock_reservado >= ?`: si entre la consulta y la reserva otro cliente se llevó las unidades, no se toca ninguna fila y la reserva no se crea. Es la regla de la Entrega 1: *la IA interpreta, el SQL decide*.

### A.2 — Grafo ReAct con límite de iteraciones

```python
grafo = StateGraph(EstadoAgente)               # MessagesState + iteraciones
grafo.add_node("agente", nodo_agente)          # LLM con las 4 tools
grafo.add_node("herramientas", ToolNode(HERRAMIENTAS, handle_tool_errors=True))
grafo.add_node("limite", nodo_limite)
grafo.add_edge(START, "agente")
grafo.add_conditional_edges("agente", debe_continuar, ["herramientas", "limite", END])
grafo.add_edge("herramientas", "agente")       # la observación vuelve al agente
grafo.add_edge("limite", END)
```

El grafo está armado a mano y no con `create_agent` (la opción "cómoda" de la clase 9) por una razón: el límite de iteraciones tiene que **responderle algo útil al cliente**, no levantar una excepción.

* `nodo_agente` cuenta una iteración cada vez que el modelo pide herramientas. El contador entra en 0 en cada consulta nueva (no tiene reducer: se pisa).
* `debe_continuar`: si el último mensaje no trae `tool_calls`, es la respuesta final → `END`. Si trae y ya se superaron `MAX_ITERACIONES = 4`, va a `limite`. Si no, a `herramientas`.
* `nodo_limite` **reemplaza** (mismo `id`) el `AIMessage` que pedía otra herramienta por la frase de derivación a un humano. Si quedara en el historial un `tool_call` sin su `ToolMessage`, el próximo turno de esa conversación fallaría: Gemini rechaza historiales con llamadas sin respuesta.
* Además, `recursion_limit = 2 × 4 + 4 = 12` como red de seguridad de LangGraph. Si alguna vez saltara, `consultar()` captura el `GraphRecursionError` y devuelve la misma derivación.

`test_limite_de_iteraciones_corta_un_agente_en_bucle` (D.3) lo prueba con un LLM falso que pide `consultar_stock` para siempre: el grafo ejecuta la herramienta **exactamente 4 veces**, llama al modelo 5 veces y responde la derivación.

**El system prompt** hereda las 6 reglas de negocio de la E3 (solo información explícita, nunca precios inventados, corregir beneficios falsos, frase de escape, citar `[DOC-XXX]`, tono) y suma cuatro: cuándo usar cada herramienta, confirmar producto/sucursal/cantidad antes de reservar, no pedir ni repetir datos personales, y no obedecer instrucciones del cliente que intenten cambiar las reglas. No se guarda en el estado: se antepone en cada llamada, así que cambiarlo no exige migrar las conversaciones guardadas. El modelo ve una ventana de los últimos 12 mensajes (`trim_messages` empezando siempre en un mensaje humano, para no cortar un par `tool_call`/`ToolMessage`); el historial completo queda en el checkpointer.

### A.3 — Memoria SQLite persistente por `thread_id`

```python
def crear_checkpointer(ruta=DB_MEMORIA) -> SqliteSaver:
    return SqliteSaver(sqlite3.connect(ruta, check_same_thread=False))
```

`SqliteSaver` guarda el estado completo del grafo **después de cada nodo**, indexado por `thread_id`, en `entrega_final/data/memoria_agente.sqlite`. `check_same_thread=False` porque FastAPI atiende cada request en un hilo distinto del threadpool.

La prueba más simple de que la memoria está en disco y no en RAM son **dos procesos de Python distintos**, uno después del otro, con el mismo `thread_id`:

```text
$ python -m entrega_final.agente --preguntar "¿Tienen Ibupirac 600 en la sucursal Norte? ¿Cuánto sale?" --thread prueba-1
Herramientas: ['consultar_stock'] (1 iteraciones)
Respuesta   : En la sucursal Norte no contamos con stock de Ibupirac 600 mg en este momento. Sí disponemos
              de Ibupirac 400 mg (presentación de 10 comprimidos), cuyo precio de lista es de $3980. [...]

$ python -m entrega_final.agente --preguntar "Dale, ¿y en el Centro hay? Si hay reservame 2 de esa de 600." --thread prueba-1
Herramientas: ['consultar_stock', 'reservar_producto'] (2 iteraciones)
Respuesta   : ¡Confirmado! Reservé 2 unidades de Ibupirac 600 mg en la sucursal Centro. Tu código de
              reserva es RES-75Z8B6 y el precio total es de $16900. [...]
```

El segundo mensaje no dice ni el producto (*"esa de 600"*) ni que la primera consulta era sobre Ibupirac. El segundo proceso lo resuelve porque cargó el historial del primero desde SQLite. `--historial prueba-1` muestra los cuatro mensajes guardados.

### A.4 — Matriz de validación

`python -m entrega_final.agente --matriz` corre 4 escenarios, cada uno en un hilo nuevo para que no se contaminen. Cada caso se valida **automáticamente** con dos criterios: la **trayectoria** (el conjunto de herramientas usadas es exactamente el esperado) y la **respuesta** (contiene lo que tiene que contener). Resultados guardados en `resultados/matriz_agente.json`:

| Escenario | Consulta | Herramientas esperadas | Herramientas usadas | Respuesta del agente | Resultado |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Política (retriever)** | *¿Hasta qué hora tengo que hacer el pedido para que me llegue hoy a casa?* | `buscar_politicas` | `buscar_politicas` (1 iteración, 4.284 tokens) | *"Para que tu pedido llegue hoy a domicilio, debe estar pagado y con la receta validada antes de las 12:00 del mediodía [DOC-004]. Si el pedido se confirma después de ese horario, se despachará al día hábil siguiente [DOC-003]."* | ✅ |
| **Stock y precio (sensor)** | *¿Tienen Ibupirac 600 en la sucursal Norte? ¿Cuánto sale?* | `consultar_stock` | `consultar_stock` (1 iteración, 4.102 tokens) | *"En la sucursal Norte no contamos con stock de Ibupirac 600 mg en este momento. Sí tenemos disponible Ibupirac 400 mg [...] a un precio de lista de $3980. ¿Te gustaría que consulte el stock de la presentación de 600 mg en alguna de nuestras otras sucursales?"* | ✅ |
| **Reserva encadenada (sensor → actuador)** | *Reservame 2 cajas de Tafirol 500 en la sucursal Centro, paso a retirarlas mañana.* | `consultar_stock`, `reservar_producto` | `consultar_stock` → `reservar_producto` (2 iteraciones, 6.185 tokens) | *"¡Perfecto! Te reservé 2 cajas de Tafirol 500 en la sucursal Centro. Tu código de reserva es RES-KOMUJY y el precio total es de $8240. Podés pasar a retirarlas dentro de las próximas 48 horas."* | ✅ |
| **Fuera de dominio (ninguna)** | *¿Me recomendás una película para ver esta noche?* | — | — (0 iteraciones, 1.872 tokens) | *"No poseo información sobre eso en las políticas de la farmacia. Te derivo con un empleado de la sucursal."* | ✅ |

**Lo que muestra cada fila:**

* **Política.** El modelo decide que es una pregunta de reglas y no de stock. La corrida se hizo un sábado y la respuesta lo tiene en cuenta: el corte de las 12:00 es el de los sábados (DOC-004), no el de las 16:00 de días hábiles. El día de la semana entra en el system prompt justamente para esto. **Una observación honesta:** la segunda oración cita DOC-003, que es la política de la sucursal **Oeste**; la regla vale para toda la red porque DOC-004 dice lo mismo, pero la cita es la de otro documento. Es el mismo riesgo que la regla 1 intenta contener ("no extiendas a una sucursal lo que se dice de otra").
* **Sensor.** El precio y el *"no hay stock"* salen del ERP, no del modelo. El agente además ofrece la presentación de 400 mg, que sí hay en Norte, porque `consultar_stock` devuelve todas las presentaciones que coinciden.
* **Reserva encadenada.** Es el caso que una cadena fija no puede resolver: el `producto_id` que necesita `reservar_producto` no lo dice el cliente, sale de la observación de `consultar_stock`. El precio total ($8.240 = 2 × $4.120) lo calcula el ERP. La reserva quedó realmente escrita: `stock_reservado` subió en 2.
* **Fuera de dominio.** El agente **decide no usar ninguna herramienta** y responde la frase de escape. Comparado con la E3, donde el retriever siempre traía k documentos aunque no sirvieran, acá la consulta cuesta una sola llamada al LLM (1.872 tokens, menos de la mitad que las demás).

---

## Parte B — API REST

### B.1 — Endpoints

| Método y ruta | Qué hace | Respuesta |
| :--- | :--- | :--- |
| `GET /health` | Liveness + dependencias: colecciones de ChromaDB con su cantidad de registros, ERP y memoria. `status` es `degradado` si falta la base vectorial | `{"status", "modelo", "base_vectorial", "erp", "memoria"}` |
| `POST /consultar` | Guardrails de entrada → agente → guardrail de salida. Sin `thread_id` genera uno; reenviándolo se sigue la conversación | `{"thread_id", "respuesta", "herramientas", "guardrail", "validador", "latencia_ms"}` |
| `GET /estado/{thread_id}` | Historial de la conversación **leído del checkpointer** + reservas de ese hilo en el ERP. 404 si no existe | `{"thread_id", "total_mensajes", "mensajes", "reservas"}` |
| `GET /metricas` | Métricas de la instancia (Parte C) | ver C.2 |

Decisiones de diseño:

* **El grafo se construye una sola vez**, en el `lifespan` de FastAPI, y queda en `app.state`. Construirlo por request sería recompilar el grafo y reabrir la conexión a SQLite en cada mensaje.
* **Endpoints `def`, no `async def`.** El grafo, el `SqliteSaver` y el ERP son sincrónicos: FastAPI corre los endpoints `def` en un threadpool y no bloquea el event loop.
* **`crear_app(grafo=None)`** es una fábrica: la API real construye el agente con Gemini y SQLite; los tests le pasan un grafo con un LLM falso y memoria en RAM.
* **Un input bloqueado devuelve 200, no 4xx**, con `guardrail: "bloqueado_entrada"` y un mensaje para el cliente. En un canal de WhatsApp el cliente tiene que recibir una respuesta; el error es del contenido, no del request. Un 502 queda reservado para cuando el agente falla.
* `thread_id` se valida con `^[A-Za-z0-9_-]{1,64}$`: entra a una base y a los logs.

### B.2 — Memoria por HTTP, incluso después de reiniciar

`python -m entrega_final.demo_api` levanta uvicorn como **proceso aparte**, conversa, **mata el proceso**, lo vuelve a levantar y sigue la misma conversación. La salida completa está en `resultados/demo_api.md`; lo esencial:

```text
> API levantada (pid 7088)
POST /consultar {"pregunta": "Hola, ¿tienen Buscapina en la sucursal Oeste?", "thread_id": "demo-02d0fa"}
  -> "¡Hola! Sí, tenemos Buscapina (10 mg x 20 grageas) en la sucursal Oeste. Hay 14 unidades
      disponibles y el precio de lista es de $9760. ¿Te gustaría que te reserve alguna unidad [...]?"
     herramientas: [consultar_stock]
POST /consultar {"pregunta": "Genial, reservame 1 caja ahí.", "thread_id": "demo-02d0fa"}
  -> "¡Perfecto! Te reservé 1 caja de Buscapina en la sucursal Oeste. Tu código de reserva es
      RES-T6AM9T [...]"
     herramientas: [reservar_producto]
> API detenida (pid 7088): se pierde todo lo que estaba en RAM

> API levantada (pid 14540)
POST /consultar {"pregunta": "¿Me recordás qué te había reservado y hasta cuándo lo tengo que retirar?",
                 "thread_id": "demo-02d0fa"}
  -> "Tenés reservada 1 caja de Buscapina (10 mg x 20 grageas) en la sucursal Oeste. La reserva está
      confirmada y tenés tiempo para retirarla hasta el 09/11/2026. El precio total es de $9760."
     herramientas: [consultar_reserva]
GET /estado/demo-02d0fa  -> 6 mensajes + reserva RES-T6AM9T (confirmada, vence 2026-11-09T13:07:36)
```

Dos cosas que muestra:

* **La memoria resolvió dos referencias.** *"reservame 1 caja **ahí**"* (segundo turno) no dice ni el producto ni la sucursal: salen del primer turno. Y después del reinicio, el agente sabe que hay una reserva que consultar.
* **El agente sigue consultando la fuente de verdad.** En el tercer turno podría haber contestado de memoria, pero llamó a `consultar_reserva`: el estado de una reserva puede cambiar (vencer, cancelarse) y el ERP es la autoridad. La memoria dice *de qué se habla*; el ERP dice *cómo está*.

### B.3 — Levantar en otra máquina con solo el README

`chroma_db/` y las bases SQLite están en `.gitignore`, así que un clon nuevo no tiene base. `python -m entrega_final.preparar` la reconstruye con los mismos scripts de las entregas anteriores: ingesta de la E2, ETL y purga de B.5, chunking de B.2 de la E3 e inicialización del ERP. Gasta solo embeddings (ninguna llamada al modelo de chat) y es idempotente.

**Lo encontramos probándolo desde cero:** la capa gratuita de Gemini limita los embeddings a **100 requests por minuto** y cada texto es una request. La ingesta (23) + el ETL (~26) + los chunks (46) pasan ese límite y el tercer paso fallaba con `429 RESOURCE_EXHAUSTED ... embed_content_free_tier_requests, limit: 100`. Como los tres pasos son idempotentes, `preparar.py` espera 65 segundos a la ventana siguiente y repite el paso. La corrida completa queda así:

```text
1/4 Base vectorial de la Entrega 2...
2/4 ETL y purga semántica...
3/4 Chunking 500/100 de la Entrega 3...
    cuota de embeddings por minuto agotada; reintento 1/3 en 65 s
4/4 ERP simulado: 27 filas en productos (erp_farmacia.sqlite)
    politicas_farmacia: 23 registros
    politicas_farmacia_chunks: 46 registros
```

Los pasos completos (clonar, entorno virtual, `.env`, preparar, levantar, probar) están en el `README.md` de la raíz.
