# Entrega 2 — Del prompt saturado a la Base de Conocimiento vectorial

**Dominio:** Atención automatizada de una red de farmacias por WhatsApp (el mismo de la Entrega 1).

Esta entrega construye la capa que el sistema de la Entrega 1 declaraba pero no tenía. El PEAS (A.3 de la Entrega 1) ya listaba, en la columna **Base de Conocimiento**, una *"base vectorial (RAG) con políticas de entrega a domicilio, zonas de cobertura, medios de pago y requisitos normativos para medicamentos bajo receta"*. Eso es exactamente lo que se implementa acá: **20 documentos de política comercial, logística y normativa**, vectorizados, persistidos y consultables con filtro duro.

> **Qué NO entra en la base vectorial.** El catálogo de medicamentos con su stock y su precio sigue viviendo en las tablas SQL de la Entrega 1 (`productos`, `convenios`). Son datos exactos, transaccionales y volátiles: se consultan con un `WHERE`, no con una distancia coseno. La base vectorial guarda lo que es narrativo y no cabe en una columna — *"¿me lo acercan a Olivos?"*, *"¿alcanza con esta receta?"* —, que es justamente lo que el empleado de mostrador hoy responde de memoria.

---

## Parte A — Embeddings y búsqueda semántica

### A.1 — Autopsia del contexto estático

Todos los números de esta tabla están medidos sobre la base real de esta entrega (base_conocimiento.json, 20 documentos) utilizando el tokenizador estándar de los modelos evaluados en la Entrega 1 (Gemini y GPT-4o)

| Problema | Aplicado a nuestro dominio |
| :--- | :--- |
| **Desangre de tokens** | Los 20 documentos concatenados suman **3.095 tokens**; una consulta típica del canal (*"Hola! Tienen Ibupirac 600 x20? cuanto sale con OSDE y me lo mandan hoy?"*) son **29 tokens**. Meter la base entera en el System Prompt hace que el **99,1 % del input sea contexto que el cliente no pidió**: se pagan 107 tokens de política por cada token de pregunta. Con las 40–60 consultas diarias por sucursal que documentamos en B.1 de la Entrega 1 y tres sucursales, son ~150 consultas/día × 3.095 tokens = **464.250 tokens diarios** quemados en repetir las mismas políticas. Recuperando solo el top-3 (≈465 tokens) el mismo volumen baja a ~70.000: **85 % menos**. Y 20 documentos es el piso: la base real de una farmacia suma el vademécum completo, los convenios de cada plan de cada prepaga y la normativa ANMAT — ahí el prompt no solo se encarece, sino que supera la ventana de contexto. |
| **Lost in the Middle** | El dato que decide una venta suele ser una sola oración enterrada en un párrafo: *"las entregas en Tigre, Nordelta, Benavídez y Escobar quedan fuera del radio"* (DOC-002) o *"el tratamiento se dispensa completo, no se fracciona el envase"* (DOC-016). En un bloque de 3.095 tokens esas cláusulas caen en la zona media, donde la atención del modelo se degrada. El modo de falla no es que el sistema diga "no sé", sino que **omite la excepción y contesta afirmativamente**. Traducido al negocio: un pedido tomado para una zona a la que el cadete no llega, o una promesa de fraccionamiento ilegal. Es la misma alucinación asertiva de A.2 de la Entrega 1, pero con el dato correcto presente en el prompt y no procesado. |
| **Inconsistencia de estado concurrente** | Diversos aspectos del dominio cambian dinámicamente durante la operación. La más crítica es la **ventana de corte del reparto** (DOC-004): a las 16:00 hs el sistema deja de prometer entregas en el día, y un alerta meteorológico puede adelantar la suspensión del servicio a las 13:00 hs (evento simulado en B.3). También rotan los **turnos farmacéuticos de guardia** y la **vigencia de los convenios**: si un convenio con IOMA se suspende (DOC-012) o se reactiva, un System Prompt estático mantendría la información desactualizada durante toda la sesión. La base vectorial permite actualizar el metadato en caliente mediante un `upsert` sin alterar el resto de la Base de Conocimiento. |

**Por qué `SELECT ... WHERE descripcion LIKE '%...%'` tampoco resuelve esto.** El operador `LIKE` compara cadenas literales, no conceptos semánticos: la consulta *"mi vieja tiene 80 años y toma pastillas para el corazón todos los días, ¿le sale algo?"* no contiene los términos `PAMI`, `jubilado`, `cobertura` ni `crónico`, devolviendo cero filas sobre el documento (DOC-010) que responde la inquietud de forma exacta (verificado en la Killer Query 1). En el extremo opuesto, `LIKE '%receta%'` retorna trece de los veinte documentos sin ponderación de relevancia. Un `LIKE` carece de ranking de similitud. Un índice invertido con sinónimos paliaría el primer problema, pero exige mantener un diccionario manual de jerga regional (*cadete*, *el bajo*, *la libreta*), tarea que el espacio de embeddings resuelve de manera nativa.

---

### A.2 — Similitud coseno a mano

#### Los dos ejes

Reducimos el dominio a dos ejes con significación operativa directa en las intenciones más frecuentes de la **Matriz de Mapeo de Intenciones** (B.3 de la Entrega 1):

* **Eje X — Carga logística:** Nivel de alusión a entregas, zonas de cobertura, plazos y modalidad de retiro (`consulta_envio`).
* **Eje Y — Carga comercial:** Nivel de alusión a precios, descuentos, obras sociales y medios de pago (`consulta_precio_cobertura`).

| Vector | Qué representa | Coordenadas (logística, comercial) |
| :--- | :--- | :---: |
| `D1` | DOC-002 — Envío a domicilio zona norte | `(9, 1)` |
| `D2` | DOC-009 — Convenio OSDE, cobertura y descuento | `(1, 9)` |
| `D3` | DOC-005 — Retiro en sucursal con promoción bancaria | `(6, 6)` |
| `Q` | *"¿me lo mandan hoy y cuánto me sale con OSDE?"* | `(7, 5)` |

#### El cálculo en tres pasos

$$\text{Similitud} = \frac{A \cdot B}{\lVert A \rVert \times \lVert B \rVert}$$

**Q vs. D1 (envío)**

1. **Producto punto:** `(7 × 9) + (5 × 1)` = `63 + 5` = **68**
2. **Normas:** `‖Q‖ = √(7² + 5²) = √74 = 8,6023` · `‖D1‖ = √(9² + 1²) = √82 = 9,0554`
3. **División:** `68 / (8,6023 × 9,0554)` = `68 / 77,8979` = **0,8729**

**Q vs. D2 (cobertura)**

1. **Producto punto:** `(7 × 1) + (5 × 9)` = `7 + 45` = **52**
2. **Normas:** `‖Q‖ = 8,6023` · `‖D2‖ = √82 = 9,0554`
3. **División:** `52 / (8,6023 × 9,0554)` = `52 / 77,8979` = **0,6675**

**Q vs. D3 (mixto)**

1. **Producto punto:** `(7 × 6) + (5 × 6)` = `42 + 30` = **72**
2. **Normas:** `‖Q‖ = 8,6023` · `‖D3‖ = √(6² + 6²) = √72 = 8,4853`
3. **División:** `72 / (8,6023 × 8,4853)` = `72 / 72,9936` = **0,9864**

#### Validación con NumPy

```python
import numpy as np

def similitud_coseno(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

Q  = np.array([7.0, 5.0])   # "¿me lo mandan hoy y cuánto me sale con OSDE?"
D1 = np.array([9.0, 1.0])   # DOC-002 · Envío zona norte
D2 = np.array([1.0, 9.0])   # DOC-009 · Convenio OSDE
D3 = np.array([6.0, 6.0])   # DOC-005 · Retiro en sucursal con promo bancaria

for nombre, D in [("DOC-002", D1), ("DOC-009", D2), ("DOC-005", D3)]:
    print(nombre, round(float(similitud_coseno(Q, D)), 4))
```

Salida:

```
DOC-002 0.8729
DOC-009 0.6675
DOC-005 0.9864
```

Los tres valores coinciden con las cuentas manuales de arriba hasta el último decimal.
El ranking queda: **DOC-005 (0,9864) > DOC-002 (0,8729) > DOC-009 (0,6675)**.

Las tres cuentas manuales coinciden con NumPy hasta el último decimal. Lo interesante del resultado es cuál gana: la consulta es **mixta** (pregunta por entrega *y* por cobertura) y el documento que también mezcla las dos cargas le queda más cerca que cualquiera de los dos especialistas. El coseno mide **orientación, no magnitud**: `D3 = (6,6)` y un hipotético `(60,60)` darían idéntica similitud, que es la propiedad que hace que un documento largo no gane solo por ser largo.

#### Reflexión sobre el umbral de aceptación

Un ranking siempre devuelve un primero, incluso cuando ninguno sirve: el vecino más cercano existe aunque la consulta sea de otro planeta. Por eso el umbral no es un detalle de tuning, es **la única defensa contra la alucinación por proximidad**. En este ejercicio 2D un corte razonable estaría cerca de 0,85 de similitud (D2, con 0,6675, claramente no responde la consulta), pero el número que vale es el que medimos sobre embeddings reales: **distancia coseno ≤ 0,35**, calibrado en C.2 con 24 consultas. Si nada lo supera, el sistema **no** devuelve el más cercano: responde *"no tengo esa información"* y deriva a un humano.

---

