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
