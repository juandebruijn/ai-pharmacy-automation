# Entrega 2 — Del prompt saturado a la Base de Conocimiento vectorial

**Dominio:** atención automatizada de una red de farmacias por WhatsApp (el mismo de la Entrega 1).

Esta entrega construye la capa que el sistema de la Entrega 1 declaraba pero no tenía. El PEAS (A.3 de la Entrega 1) ya listaba, en la columna **Base de Conocimiento**, una *"base vectorial (RAG) con políticas de entrega a domicilio, zonas de cobertura, medios de pago y requisitos normativos para medicamentos bajo receta"*. Eso es exactamente lo que se implementa acá: **20 documentos de política comercial, logística y normativa**, vectorizados, persistidos y consultables con filtro duro.

> **Qué NO entra en la base vectorial.** El catálogo de medicamentos con su stock y su precio sigue viviendo en las tablas SQL de la Entrega 1 (`productos`, `convenios`). Son datos exactos, transaccionales y volátiles: se consultan con un `WHERE`, no con una distancia coseno. La base vectorial guarda lo que es narrativo y no cabe en una columna — *"¿me lo acercan a Olivos?"*, *"¿alcanza con esta receta?"* —, que es justamente lo que el empleado de mostrador hoy responde de memoria.

---

## Parte A — Embeddings y búsqueda semántica

### A.1 — Autopsia del contexto estático

Todos los números de esta tabla están medidos sobre la base real de esta entrega (`base_conocimiento.json`, 20 documentos) con el tokenizador del modelo que usa el pipeline de la Entrega 1.

| Problema | Aplicado a nuestro dominio |
| :--- | :--- |
| **Desangre de tokens** | Los 20 documentos concatenados son **3.095 tokens**; una consulta típica del canal (*"Hola! Tienen Ibupirac 600 x20? cuanto sale con OSDE y me lo mandan hoy?"*) son **29 tokens**. Meter la base entera en el System Prompt hace que el **99,1 % del input sea contexto que el cliente no pidió**: se pagan 107 tokens de política por cada token de pregunta. Con las 40–60 consultas diarias por sucursal que documentamos en B.1 de la Entrega 1 y tres sucursales, son ~150 consultas/día × 3.095 tokens = **464.250 tokens diarios** quemados en repetir las mismas políticas. Recuperando solo el top-3 (≈465 tokens) el mismo volumen baja a ~70.000: **85 % menos**. Y 20 documentos es el piso: la base real de una farmacia suma el vademécum, los convenios de cada plan de cada prepaga y la normativa ANMAT — ahí el prompt no se encarece, directamente no entra en la ventana de contexto. |
| **Lost in the Middle** | El dato que decide una venta suele ser una sola oración enterrada en un párrafo: *"las entregas en Tigre, Nordelta, Benavídez y Escobar quedan fuera del radio"* (DOC-002) o *"el tratamiento se dispensa completo, no se fracciona el envase"* (DOC-016). En un bloque de 3.095 tokens esas cláusulas caen en la zona media, que es donde la atención del modelo es más débil. El modo de falla no es que el sistema diga "no sé": es que **omite la excepción y contesta que sí**. Traducido al negocio, es un pedido tomado para una zona a la que el cadete no llega, o un antibiótico fraccionado. Es la misma alucinación asertiva que documentamos en A.2 de la Entrega 1, solo que ahora con el dato correcto presente en el prompt y no leído. |
| **Inconsistencia de estado concurrente** | Varias cosas de este dominio cambian dentro de la misma sesión de WhatsApp. La más filosa es la **ventana de corte del reparto** (DOC-004): a las 16:00 el sistema deja de poder prometer entrega para hoy, y un alerta meteorológico puede adelantarla a las 13:00 sin aviso (es el evento que simulamos en B.3). También cambian el **turno farmacéutico nocturno**, que rota semanalmente, y la **vigencia de un convenio**: el de IOMA está suspendido (DOC-012) y el día que se reactive, un prompt estático seguiría diciendo lo contrario. Un System Prompt es una foto tomada en el momento del deploy; el cliente que abrió la conversación a las 15:50 y confirma a las 16:10 recibiría una promesa que el sistema ya no puede cumplir. |

**Por qué `SELECT ... WHERE descripcion LIKE '%...%'` tampoco resuelve esto.** El `LIKE` compara cadenas de caracteres, no significados: la consulta *"mi vieja tiene 80 años y toma pastillas para el corazón todos los días, ¿le sale algo?"* no contiene ni una sola vez las palabras `PAMI`, `jubilado`, `cobertura` ni `crónico`, así que devuelve cero filas sobre un documento (DOC-010) que la responde entera — y lo verificamos en la Killer Query 1. En el otro extremo, `LIKE '%receta%'` matchea trece de los veinte documentos sin ningún orden de relevancia. El `LIKE` no tiene ranking: no sabe cuál de esos trece es *el* que responde. Un índice invertido con sinónimos mitigaría el primer problema, pero exige mantener a mano el diccionario de jerga de cada barrio (*cadete*, *el bajo*, *la libretita*), que es precisamente el trabajo que el embedding hace solo.

---

