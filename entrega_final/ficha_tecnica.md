# Ficha técnica — Asistente de farmacia por WhatsApp

## 1. Identificación

| Campo | Valor |
| :--- | :--- |
| **Sistema** | Agente conversacional para la atención por WhatsApp de una red de farmacias |
| **Versión** | 1.0.0 (Entrega Final) |
| **Equipo** | Camila Belén Capua · Franco Salama · Juan Pablo Ruax Debruijn · Lorena Wajchman · Miguel Ángel Tadakuma · Tomás Rossi |
| **Repositorio** | [github.com/juandebruijn/ai-pharmacy-automation](https://github.com/juandebruijn/ai-pharmacy-automation) (rama `main`) |
| **Estado** | Prototipo funcional. ERP simulado; sin integración real con WhatsApp Business |

## 2. Problema y alcance

**Problema (Entrega 1).** Los empleados de mostrador interrumpen la atención presencial para contestar por WhatsApp consultas repetitivas de stock, precio, envíos y coberturas. En la Entrega 1 medimos que un LLM sin acceso a los datos **alucina** precios, coberturas y zonas de envío con total seguridad.

**Qué hace el sistema:**

* Responde sobre las **políticas** de la red (envíos, reservas, pagos, convenios, recetas, horarios) citando el documento.
* Consulta **stock y precio de lista** por sucursal en el ERP.
* **Reserva** stock por 48 h para retirar en sucursal y consulta el estado de las reservas.
* Recuerda la conversación por `thread_id`, aunque se reinicie el servidor.

**Qué NO hace (fuera de alcance):** cobrar, calcular el descuento final de una obra social, validar recetas (las reservas con receta quedan `pendiente_receta` para el farmacéutico), vender o reservar psicotrópicos, ni despachar pedidos a domicilio. Todo lo que no puede resolver lo deriva a un empleado con una frase fija.

## 3. Arquitectura

| Capa | Componente | Tecnología |
| :--- | :--- | :--- |
| Exposición | API REST: `/health`, `/consultar`, `/estado/{thread_id}`, `/metricas` | FastAPI 0.162 + uvicorn |
| Seguridad | 2 validadores de entrada (longitud, prompt injection) + 1 de salida (enmascarado de PII) | Python `re`, deterministas |
| Orquestación | Grafo ReAct: `agente` ⇄ `herramientas`, nodo `limite` (máx. 4 iteraciones) | LangGraph 1.2 (`StateGraph`, `ToolNode`) |
| Memoria | Checkpointer por `thread_id`, persistido en disco | `SqliteSaver` (langgraph-checkpoint-sqlite) |
| Modelo | LLM con tool calling, `temperature=0`, `max_retries=6` | `gemini-3.1-flash-lite` (Google AI, capa gratuita) |
| Conocimiento | 23 políticas → 46 chunks de 500/100 caracteres, filtro `vigente=True`, k=4 | ChromaDB persistente + `gemini-embedding-001` (768 dims) |
| Datos operativos | `productos` (9 presentaciones × 3 sucursales) y `reservas` | SQLite (ERP simulado) |
| Observabilidad | Logs JSON Lines, métricas de instancia, trazas por invocación | `logging` + LangSmith |

### Herramientas del agente

| Herramienta | Tipo | Lee / escribe | Reglas en código |
| :--- | :--- | :--- | :--- |
| `buscar_politicas` | Retriever (RAG de la E3) | Lee ChromaDB | Filtro de vigencia (excluye el régimen derogado DOC-020) |
| `consultar_stock` | Sensor | Lee `productos` | Avisos explícitos si es psicotrópico o no hay stock |
| `reservar_producto` | Actuador | Escribe `productos.stock_reservado` y `reservas` | Rechaza psicotrópicos, >20 unidades y stock insuficiente; `UPDATE` condicional atómico |
| `consultar_reserva` | Sensor | Lee `reservas` | Solo las del propio `thread_id` (inyectado, invisible al modelo); libera las vencidas |

## 4. Flujo de una consulta

1. `POST /consultar` → el middleware asigna un `request_id`.
2. **Guardrails de entrada.** Si bloquean: respuesta fija, 0 tokens, no entra a la memoria, métrica `bloqueos`.
3. **Agente.** Carga el historial del hilo desde SQLite, el LLM decide herramientas, observa, repite (máx. 4 vueltas) y responde.
4. **Guardrail de salida.** Enmascara DNI, CUIT, teléfono, email y tarjeta en la respuesta **y en la memoria**.
5. Log JSON `consulta_respondida`, métricas, traza en LangSmith con `thread_id` y `request_id`.

## 5. Resultados medidos

| Indicador | Valor | Fuente |
| :--- | :--- | :--- |
| Matriz de validación del agente (trayectoria + respuesta) | **4 / 4** | `resultados/matriz_agente.json` |
| Suite de seguridad | **33 / 33** tests, ~2 s, sin red | `security_tests.py` |
| Memoria tras reiniciar la API | ✅ el agente recuerda la reserva en un proceso nuevo | `resultados/demo_api.md` |
| Ataques bloqueados en la demo | 3 / 3 (2 injection, 1 longitud), 0 tokens gastados | `/metricas` |
| Tasa de error | 0 % (14 consultas) | `/metricas` |
| Latencia mediana / p95 | 4,2 s / 63,6 s (p95 dominado por reintentos de la capa gratuita) | `/metricas` |
| Tokens por consulta | 4.427 promedio · ~1.850 de piso fijo por llamada al LLM | LangSmith |
| Llamadas al LLM por consulta | 2,09 promedio (1 fuera de dominio, 3 en una reserva encadenada) | LangSmith |
| Costo estimado (capa paga) | **US$ 0,00124 por consulta** · US$ 37 / mes a 1.000 consultas / día | `costos_langsmith.py` |

## 6. Riesgos y mitigaciones

| Riesgo | Mitigación implementada | Riesgo residual |
| :--- | :--- | :--- |
| Alucinación de precios o políticas | Precios solo del ERP; políticas solo del retriever; frase de escape; citas `[DOC-XXX]` | Citas a un documento que no corresponde (C.4 del informe) |
| Acción indebida (reservar un psicotrópico, vender de más) | Reglas en el actuador, no en el prompt; `UPDATE` condicional | — |
| Prompt injection | Validador de 5 familias + reglas 3 y 8 del prompt + reglas en código | Ataques parafraseados sin las señales del regex |
| Fuga de datos personales | El prompt prohíbe pedirlos; enmascarado de salida y de memoria; el log no guarda la pregunta | DNI en palabras, nombres propios (haría falta NER) |
| Bucle de herramientas (costo) | `MAX_ITERACIONES=4` + `recursion_limit=12`, deriva a humano | — |
| Abuso de costo | Tope de 600 caracteres; ventana de 12 mensajes | Sin rate limit por cliente |
| Cuota de la capa gratuita (429) | `max_retries=6`; `preparar.py` reintenta los embeddings | Latencias de hasta 60 s; en producción, capa paga |
| Ruido en el contexto | Chunks 500/100 de la E3 | Sin reranking (por costo): DOC-105 en la consulta de insulina |

## 7. Operación

```bash
python -m entrega_final.preparar                       # una vez: base vectorial + ERP
uvicorn entrega_final.api:app --port 8000              # API (docs en /docs)
python -m pytest entrega_final/security_tests.py -v    # suite de seguridad
python -m entrega_final.agente --matriz                # matriz de validación
python -m entrega_final.demo_api                       # demo de memoria con reinicio
python -m entrega_final.costos_langsmith               # costos desde LangSmith
```

| Variable | Default | Para qué |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | — (obligatoria) | LLM y embeddings |
| `GEMINI_MODEL_AGENTE` | `gemini-3.1-flash-lite` | Modelo del agente |
| `MAX_ITERACIONES` | `4` | Vueltas agente → herramientas por consulta |
| `LANGSMITH_API_KEY` / `LANGSMITH_TRACING` / `LANGSMITH_PROJECT` | — / `true` / `ai-pharmacy-automation` | Trazas |
| `PRECIO_ENTRADA_USD_MTOK` / `PRECIO_SALIDA_USD_MTOK` | `0.25` / `1.50` | Costo estimado |

## 8. Próximos pasos

1. **Evaluar el agente con RAGAS** sobre el golden dataset de la E3, y medir si el reranking con LLM-juez compensa su costo por vuelta.
2. **Context caching** del system prompt + schemas de herramientas: el prefijo de ~1.850 tokens es idéntico en todas las llamadas.
3. **Router barato** antes del agente para saludos y fuera de dominio (hoy cuestan ~1.850 tokens cada uno).
4. **Integración real**: webhook de WhatsApp Business → `/consultar` con `thread_id` derivado del teléfono (hash, nunca el número en claro), y el ERP real en lugar del SQLite simulado.
5. **Métricas agregadas** (Prometheus) y rate limit por cliente.

---

## Guion de la defensa (15 minutos)

| Min. | Bloque | Quién | Contenido |
| :---: | :--- | :--- | :--- |
| 0 – 2 | Problema y evolución | Tomás | Del LLM que alucinaba (E1) a la base vectorial (E2), el RAG evaluado (E3) y el agente (esta entrega). Qué agrega cada capa |
| 2 – 4 | Agente | Miguel · Juan Pablo | Las 4 herramientas y por qué los docstrings son la interfaz; grafo con nodo `limite`; matriz 4/4 |
| 4 – 9 | **Demo en vivo** | Camila | `/docs` → `/health` → consulta de stock → reserva con *"ahí"* → **matar uvicorn y levantarlo** → *"¿qué te había reservado?"* → `/estado/{thread_id}` |
| 9 – 11 | Observabilidad | Lorena | Log JSON de la demo recién hecha, `/metricas`, traza en LangSmith y tabla de costos; el hallazgo de C.4 |
| 11 – 12 | Seguridad | Franco | Injection bloqueado en vivo (0 tokens), `pytest` de la suite en vivo (33 en 2 s) |
| 12 – 15 | Preguntas | Todos | — |

**Plan B para la demo.** Si la cuota de Gemini corta en vivo (429), se muestran `resultados/demo_api.md` y la traza de LangSmith de la corrida previa; la suite de seguridad no depende de la red.

### Preguntas probables y respuesta corta

| Pregunta | Respuesta |
| :--- | :--- |
| ¿Por qué un grafo a mano y no `create_agent`? | Para que el límite de iteraciones sea un nodo que le responde al cliente y repara el historial, no una excepción. |
| ¿Qué pasa si el modelo ignora el prompt y pide reservar un psicotrópico? | El actuador lo rechaza en código; hay un test. El prompt orienta, el código decide. |
| ¿Por qué guardrails con regex y no NeMo? | Deterministas, sin costo ni cuota, testeables offline. Cubren menos paráfrasis: lo compensa que las acciones tienen reglas en código. |
| ¿Dónde está la memoria y por qué sobrevive al reinicio? | `SqliteSaver` escribe el estado después de cada nodo en un archivo en disco, indexado por `thread_id`. |
| ¿Por qué el p95 es tan alto? | Reintentos ante 429/503 de la capa gratuita dentro de la llamada al LLM; las herramientas tardan milisegundos (traza de LangSmith). |
| ¿Cuánto cuesta en producción? | ~US$ 0,0012 por consulta; 87 % es entrada (prompt + schemas + historial que se reenvían en cada vuelta). |
| ¿Por qué no hay reranking como en la E3? | Cada llamada extra se multiplica por las vueltas del bucle; queda como próximo paso a medir con RAGAS. |
