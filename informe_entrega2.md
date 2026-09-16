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

### A.3 — `base_conocimiento.json`

**Archivo:** [`base_conocimiento.json`](base_conocimiento.json) — **20 documentos** (mínimo pedido: 15).

Estructura exacta de cada registro:

```json
{
  "id": "DOC-002",
  "descripcion_semantica": "La sucursal Norte cubre el corredor de la zona norte del Gran Buenos Aires sobre el eje de la avenida Maipú y el acceso Panamericana: Vicente López, Olivos, La Lucila, Martínez, Acassuso y San Isidro, incluyendo la franja del bajo que va desde las vías del tren Mitre hasta el río. El reparto se terceriza con una empresa de mensajería que hace tres rondas por día. Las entregas en Tigre, Nordelta, Benavídez y Escobar quedan fuera del radio y se rechazan en el momento de la consulta, sin tomar el pedido. Cuando el cliente vive en el límite de la zona, el sistema pide el código postal antes de confirmar la entrega para no comprometer un plazo que no se puede cumplir.",
  "metadatos": {
    "categoria": "logistica",
    "intencion_relacionada": "consulta_envio",
    "sucursal": "SUC-002",
    "vigente": true,
    "tags_regionales": ["zona norte", "el bajo", "Panamericana", "mensajería", "acercar a casa"]
  }
}
```

#### Contenido de la base

| Categoría | Documentos | Qué cubre |
| :--- | :--- | :--- |
| `logistica` | DOC-001 … DOC-006 | Zonas de reparto por sucursal, ventana de corte, retiro en sucursal, cadena de frío |
| `pagos` | DOC-007, DOC-008 | Medios de pago aceptados, promociones bancarias y su no acumulación |
| `cobertura` | DOC-009 … DOC-012 | Convenios OSDE, PAMI, Swiss Medical/OMINT/Medifé, IOMA suspendido |
| `normativa` | DOC-013 … DOC-016, DOC-020 | Psicotrópicos, receta electrónica, venta libre, antibióticos, régimen derogado |
| `atencion` | DOC-017 … DOC-019 | Horarios y turno nocturno, devoluciones, faltantes y encargo a droguería |

#### El esquema justificado con la Regla del Arquitecto

> *Todo campo sobre el que haya que aplicar un filtro duro va como metadato. Lo narrativo o sensorial queda dentro del texto. Nunca al revés.*

| Campo | Tipo | Por qué es **metadato** y no texto |
| :--- | :--- | :--- |
| `categoria` | categórico (`logistica`, `pagos`, `cobertura`, `normativa`, `atencion`) | Es el primer corte duro de cualquier consulta. Una pregunta sobre cobertura no debe competir en distancia contra una política de reparto: no es que el documento de logística sea "menos parecido", es que **no es candidato**. |
| `intencion_relacionada` | categórico, cerrado a las **seis etiquetas** de la Matriz de Intenciones de la Entrega 1 | Es el enganche literal entre las dos entregas. Lo que el LLM clasifica en el paso 1 del flujo (`consulta_envio`, `validar_receta`, …) se convierte acá en un filtro. El conjunto es el mismo `ENUM` de `interacciones.intencion` del esquema MySQL de B.5.b. |
| `sucursal` | categórico (`SUC-001`, `SUC-002`, `SUC-003`, `TODAS`) | Viene directo del contrato de la API de la Entrega 1, donde `sucursal_id` es campo obligatorio del request. **El stock, el precio y la zona de reparto son por sucursal**: responder la política de otra sucursal es la misma clase de error que documentamos en A.2 de la Entrega 1. El valor `TODAS` marca las políticas de red y se resuelve con `$in`, no con `$eq`. |
| `vigente` | **booleano de estado** | El campo que impide que el sistema conteste con una norma derogada o un convenio caído. DOC-012 (IOMA suspendido) y DOC-020 (régimen de psicotrópicos derogado) existen en la base **para bloquear la respuesta equivocada, no para habilitarla**. Es el filtro que la Killer Query 2 pone a prueba. |
| `tags_regionales` | lista de strings | La jerga comercial y barrial: *cadete*, *el bajo*, *la libretita*, *remedios gratis*, *me lo consiguen*. No se usa para filtrar: sirve de documentación del vocabulario real del canal y es lo que el ETL **rescata y fusiona** cuando purga un casi-duplicado (B.5). |

Y lo que deliberadamente **quedó dentro del texto**, no como metadato: los horarios concretos, los barrios cubiertos, los plazos en horas, los nombres de las drogas reguladas, las excepciones. Nadie pregunta *"documentos cuya ventana de corte sea 16:00"* — preguntan *"¿me llega hoy?"*, y eso se resuelve por significado. Convertir esas cláusulas en metadatos habría multiplicado el esquema por veinte sin ganar un solo filtro útil.

---

### A.4 — Índice FAISS

**Script:** [`pipeline_vectorial.py`](pipeline_vectorial.py)

| Requisito del enunciado | Cómo se cumple |
| :--- | :--- |
| Credenciales desde `.env` | `load_dotenv()` y `os.environ["GEMINI_API_KEY"]`. No hay ninguna clave en el código ni en el historial de git. |
| Embeddings de `descripcion_semantica` | `gemini-embedding-001` a 768 dimensiones. La Entrega 1 ya usaba Gemini, así que no suma una credencial nueva. Las descripciones se vectorizan con `task_type="RETRIEVAL_DOCUMENT"` y las consultas con `"RETRIEVAL_QUERY"`: un documento largo y una pregunta corta no son el mismo tipo de texto, y en B.1 se muestra cuánto importa esa distinción. |
| `IndexFlatL2` o `IndexFlatIP` normalizado | **`IndexFlatIP` con vectores normalizados a norma 1.** Con ‖v‖ = 1 el producto interno *es* la similitud coseno, así que el ranking de FAISS coincide exactamente con el de ChromaDB (`hnsw:space="cosine"`) y las dos partes del TP son comparables número a número. |
| `write_index()` / `read_index()` | `persistir()` y `recargar()`. Además del `.index` se guarda `farmacia_ids.json` con el mapeo fila → `DOC-XXX`: FAISS almacena vectores, no identificadores, y sin ese mapeo el índice devuelve "fila 7" y nadie sabe qué documento es. |
| Búsqueda top-K sobre 3 consultas | Ver abajo. |

> **Nota de portabilidad.** `write_index()` y `read_index()` son C++ y abren el archivo con `fopen()`, que en Windows interpreta la ruta con la codepage ANSI. Si el repositorio está clonado en una carpeta con acentos o símbolos (el caso de esta máquina: *"5° Cuatrimestre"*), la ruta absoluta se corrompe al cruzar a C++ y FAISS falla con `could not open ... No such file or directory` aunque el directorio exista. El script entra al directorio del índice y le pasa un nombre relativo y ASCII, así que funciona en cualquier máquina y con cualquier ruta.

#### Las tres consultas de prueba

Están escritas como las escribiría un cliente por WhatsApp: **con jerga y sin las palabras exactas del documento** que tienen que recuperar. Si funcionaran solo con las palabras textuales, un `LIKE` alcanzaría.

```
[1] "che, me lo pueden acercar en moto hasta el bajo de Vicente López?"
    1. DOC-002  similitud=0.7405  distancia=0.2595  (logistica)   <- sucursal Norte
    2. DOC-001  similitud=0.7044  distancia=0.2956  (logistica)
    3. DOC-003  similitud=0.6560  distancia=0.3440  (logistica)

[2] "soy jubilado, los remedios de la presión los tengo que pagar?"
    1. DOC-010  similitud=0.7594  distancia=0.2406  (cobertura)   <- PAMI
    2. DOC-011  similitud=0.6602  distancia=0.3398  (cobertura)
    3. DOC-009  similitud=0.6527  distancia=0.3473  (cobertura)

[3] "puedo abonar escaneando con la billetera del celular?"
    1. DOC-007  similitud=0.6600  distancia=0.3400  (pagos)       <- medios de pago
    2. DOC-005  similitud=0.5889  distancia=0.4111  (logistica)
    3. DOC-012  similitud=0.5831  distancia=0.4169  (cobertura)   <- dado de baja
```

Las tres aciertan el documento correcto en el puesto 1. Vale la pena mirar el ruido: la consulta 1 trae las otras dos sucursales en los puestos 2 y 3 — semánticamente son casi lo mismo ("zona de reparto") y solo un metadato puede separarlas. Y la consulta 3 mete en el top-3 a **DOC-012, que está dado de baja**: FAISS no tiene forma de filtrarlo. Los dos problemas son el argumento de la Parte B.

---

### A.5 — Prueba destructiva: volatilidad de la RAM

Log real de la corrida (`pipeline_vectorial.py` con los flags `--borrar`, `--volatil` y sin flags).

#### Paso 1 — construir **sin** `write_index()`

```
$ python pipeline_vectorial.py --borrar
[A.5] indice_faiss/ borrado. El índice ya no existe en disco.

$ python pipeline_vectorial.py --volatil
[build] --volatil: se construye en RAM y NO se persiste.
  Vectorizando 20 descripciones con gemini-embedding-001...
  Índice construido: 20 vectores de 768 dimensiones.
  [A.5] write_index() OMITIDO a propósito: cuando este proceso termine,
        el índice muere con la RAM y hay que volver a pagar los tokens.
real    0m5.054s
```

#### Paso 2 — reiniciar el entorno (proceso nuevo)

```
$ ls indice_faiss/
   (indice_faiss/ no existe: el indice se perdio)
```

#### Paso 3 — el índice hay que **regenerar**, y se paga de nuevo

```
$ python pipeline_vectorial.py --volatil
[build] --volatil: se construye en RAM y NO se persiste.
  Vectorizando 20 descripciones con gemini-embedding-001...
  Índice construido: 20 vectores de 768 dimensiones.
real    0m4.351s
```

Segunda llamada a la API, segundo pago de los mismos 3.095 tokens, para obtener **exactamente los mismos 20 vectores**.

#### Paso 4 — ahora **con** `write_index()`

```
$ python pipeline_vectorial.py
[build] no hay índice en disco.
  Vectorizando 20 descripciones con gemini-embedding-001...
  Índice construido: 20 vectores de 768 dimensiones.
  Persistido en indice_faiss\farmacia.index (60.0 KB).
real    0m4.381s
```

#### Paso 5 — proceso nuevo: se recarga de disco, **sin consumir tokens**

```
$ python pipeline_vectorial.py
[read_index] Índice recargado desde disco: 20 vectores.
             0 llamadas a la API, 0 tokens consumidos.

[1] "che, me lo pueden acercar en moto hasta el bajo de Vicente López?"
    1. DOC-002  similitud=0.7405  distancia=0.2595  (logistica)
    2. DOC-001  similitud=0.7044  distancia=0.2956  (logistica)
    3. DOC-003  similitud=0.6560  distancia=0.3440  (logistica)
...
real    0m3.385s
```

```
$ ls -la indice_faiss/
-rw-r--r-- 61485 farmacia.index
-rw-r--r--   414 farmacia_ids.json
```

**La evidencia:** los scores del paso 5 (`0.7405 / 0.7044 / 0.6560`) son **idénticos** a los del paso 1, dígito por dígito, pero sin una sola llamada a la API. El índice recargado no es una aproximación del original: es el original. Y el tiempo baja de ~4,4 s a ~3,4 s — el segundo entero que se ahorra es la llamada de red que no se hizo.

#### Reflexión

En producción, si el servidor se reinicia sin persistencia, el sistema queda ciego hasta terminar de re-vectorizar la base entera: con 20 documentos son 2 segundos, pero con el vademécum completo de la farmacia son minutos de cold start en los que el agente no puede responder nada — y se paga la factura de embeddings de nuevo en cada deploy, cada crash y cada escalado automático. Con dos servidores el problema cambia de forma: cada uno construye su índice por su cuenta y quedan **dos bases divergentes** respondiendo distinto a la misma pregunta según a qué instancia caiga el cliente, y un documento actualizado en uno no existe en el otro. La persistencia en disco no es una optimización de costo: es lo que convierte el índice en una **fuente de verdad única y compartida**, que es justamente lo que el `PersistentClient` de ChromaDB formaliza en B.1.

---

## Parte B — ChromaDB, filtrado híbrido y ETL

### B.1 — Migración a ChromaDB

**Script:** [`vector_db.py`](vector_db.py) · correr con `python vector_db.py --ingesta`

```python
cliente = chromadb.PersistentClient(path=str(DIR_CHROMA))   # NO Client()
coleccion = cliente.get_or_create_collection(
    name="politicas_farmacia",
    embedding_function=EmbeddingFarmacia(),
    metadata={"hnsw:space": "cosine"},
)
coleccion.upsert(ids=..., documents=..., metadatas=...)      # NO add()
```

Salida real:

```
[B.1] Colección 'politicas_farmacia' en chroma_db/ (PersistentClient, hnsw:space=cosine)
      upsert de 20 documentos: 0 -> 20 registros.
```

Correrlo por segunda vez:

```
      upsert de 20 documentos: 20 -> 20 registros.
```

Las tres decisiones, y por qué:

* **`PersistentClient` y no `Client()`.** `Client()` es in-memory: nos devolvería exactamente al problema que acabamos de demostrar en A.5. La colección vive en `chroma_db/` y sobrevive al reinicio del proceso.
* **`hnsw:space="cosine"` y no L2.** Las descripciones tienen largos distintos (de 600 a 900 caracteres) y con distancia euclídea el documento más largo arrastra la distancia por magnitud y no por contenido. Además es la misma métrica del `IndexFlatIP` normalizado de A.4, así que los números de la Parte A y los de la Parte B son directamente comparables.
* **`upsert` y no `add`.** `add` revienta con `IDExistsError` en la segunda corrida y obliga a limpiar la colección a mano antes de cada ingesta. Con `upsert` el script es **idempotente**: reconstruir la base entera desde `base_conocimiento.json` es siempre seguro, que es la propiedad que pide el enunciado cuando dice que la base se reconstruye desde el JSON.

> **Un detalle de implementación.** ChromaDB solo acepta `str`, `int`, `float` o `bool` en los metadatos: una lista rompe el `upsert`. `tags_regionales` se guarda como string separado por comas (`aplanar_metadatos()`) y el JSON canónico conserva la lista. Es el único lugar donde el esquema de la base y el de la colección difieren, y está aislado en una sola función.

#### La función de embedding: por qué no alcanza la que viene de fábrica

ChromaDB trae `GoogleGenaiEmbeddingFunction` lista para usar, y la usamos — pero heredando de ella para cambiar **una sola cosa**: que las consultas se vectoricen con `task_type="RETRIEVAL_QUERY"` en vez de `"RETRIEVAL_DOCUMENT"`.

```python
class EmbeddingFarmacia(GoogleGenaiEmbeddingFunction):
    def __init__(self):
        super().__init__(model_name=MODELO, dimension=DIMENSION,
                         task_type="RETRIEVAL_DOCUMENT")
        self._como_consulta = GoogleGenaiEmbeddingFunction(
            model_name=MODELO, dimension=DIMENSION, task_type="RETRIEVAL_QUERY"
        )

    def embed_query(self, input):
        return self._como_consulta(input)
```

No es un adorno: una política de 900 caracteres y una pregunta de WhatsApp de diez palabras no son el mismo tipo de texto. Medimos las dos variantes sobre las mismas 24 consultas con las que calibramos el umbral en C.2:

| Cómo se vectoriza la consulta | Peor caso **dentro** del catálogo | Mejor caso **fuera** | Brecha |
| :--- | ---: | ---: | ---: |
| Como documento (la función tal cual viene) | 0,2798 | 0,2696 | **−0,0102** |
| Como consulta (`RETRIEVAL_QUERY`) | 0,3400 | 0,3631 | **+0,0231** |

Con la función de fábrica la brecha es **negativa**: la peor consulta legítima queda más lejos que la mejor consulta fuera de catálogo, y **no existe ningún umbral que las separe**. El sistema, o contesta cualquier cosa, o rechaza consultas válidas. Las seis líneas de la subclase son lo que hace posible el umbral de aceptación de C.2.

---

### B.2 — Los tres límites de FAISS que ChromaDB resuelve

| Límite de FAISS | Cómo se manifiesta en nuestro dominio | Cómo lo resuelve ChromaDB |
| :--- | :--- | :--- |
| **Sin persistencia transaccional / atomicidad** | `write_index()` serializa el índice entero de una vez. Si la farmacia actualiza la política de reparto de una sucursal y el proceso se corta a mitad del `write`, el archivo queda truncado y **se pierden los 20 documentos**, no el que se estaba tocando. Peor: el `.index` y su sidecar `farmacia_ids.json` se escriben por separado, así que un corte entre las dos escrituras deja el mapeo fila → `DOC-XXX` desfasado y el sistema empieza a devolver la política de la sucursal equivocada con toda confianza. Y no hay forma de escribir un documento sin reescribir el archivo completo. | El `PersistentClient` escribe sobre SQLite con un WAL: cada `upsert` es una transacción que se confirma o no ocurre, y el documento, sus metadatos y su vector se guardan juntos en la misma operación. No existe el estado intermedio de "vector actualizado pero metadato viejo". |
| **Sin filtrado híbrido nativo** | Se ve literalmente en el log de A.4: la consulta *"puedo abonar escaneando con la billetera del celular?"* trae a **DOC-012 (convenio IOMA suspendido, `vigente: false`) en el puesto 3**, y la consulta 1 trae las tres sucursales mezcladas. FAISS solo sabe de vectores: no conoce `vigente` ni `sucursal`. La única salida sería traer top-20 y descartar en Python — el post-filtering que el enunciado prohíbe, y que además puede dejar menos de *k* resultados sin que nadie se entere. | El `where` con operadores nativos (`$and`, `$eq`, `$in`) se evalúa **dentro** del motor, antes de rankear. `{"vigente": {"$eq": True}}` hace que el régimen derogado ni siquiera sea candidato, y `{"sucursal": {"$in": ["SUC-002", "TODAS"]}}` deja las políticas de la sucursal consultada más las de red. Es B.4. |
| **CRUD ineficiente / sin concurrencia** | `IndexFlatIP` no tiene borrado real: `remove_ids` reindexa y desplaza todas las filas posteriores, lo que **invalida el sidecar** entero. Actualizar el documento del corte de reparto (DOC-004) obliga a reconstruir y reescribir el índice completo. Y el archivo no está pensado para escritores concurrentes: con el proceso de atención leyendo y un ETL nocturno escribiendo, no hay bloqueo que los coordine. En una red de tres sucursales que tocan sus propias políticas, eso es una condición de carrera esperando. | `upsert` y `delete` operan por `id` sobre el registro puntual, sin tocar el resto de la colección; el índice HNSW se actualiza incremental. SQLite serializa los escritores y deja leer en paralelo, así que la consulta de un cliente y la actualización de una política pueden convivir. Es lo que hace viable el evento en caliente de B.3. |

---

### B.3 — Evento de negocio en caliente

**Correr con:** `python vector_db.py --evento`

El evento elegido es el que A.1 identificó como el dato más volátil del dominio: **la ventana de corte del reparto se adelanta por un alerta meteorológico**, mientras hay clientes con la conversación abierta preguntando si les llega hoy.

```
[B.3] Estado ANTES del evento:
      metadatos: {'tags_regionales': 'horario de corte, para hoy, urgente, en el día, llega hoy',
                  'categoria': 'logistica', 'sucursal': 'TODAS', 'vigente': True,
                  'intencion_relacionada': 'consulta_envio'}
      documento: La ventana de corte del reparto define si un pedido llega en el día o pasa a la
                 jornada siguiente. Para que una compra s...

[B.3] Estado DESPUÉS del upsert (verificado con get):
      metadatos: {'intencion_relacionada': 'consulta_envio', 'sucursal': 'TODAS',
                  'tags_regionales': 'contingencia, corte adelantado, no llega hoy, alerta',
                  'vigente': True, 'categoria': 'logistica'}
      documento: CONTINGENCIA VIGENTE HOY: por un alerta meteorológico la mensajería suspendió la
                 ronda de la tarde y la ventana de corte...
      total de registros en la colección: 20 (no se duplicó nada)
```

El cambio impacta de inmediato en la búsqueda: en la corrida de B.4, la consulta *"me lo acercan hasta Olivos en el día?"* ahora devuelve el documento de contingencia en el puesto 2.

**Por qué `upsert` y no `add` ni `update`.** `add` falla si el id ya existe y `update` falla si no existe: las dos obligan a consultar antes para saber cuál usar, y esa consulta previa es una condición de carrera con cualquier otro proceso escribiendo. `upsert` resuelve el caso real —"quiero que DOC-004 diga esto, exista o no"— en una sola operación atómica, y hace que el script sea re-ejecutable sin duplicar ni romperse.

---

### B.5 — ETL y purga semántica

**Script:** [`etl_purga.py`](etl_purga.py) · **Lote sucio:** [`ingesta_cruda.json`](ingesta_cruda.json)

`base_conocimiento.json` se mantiene limpio: es la fuente de verdad desde la que se reconstruye toda la base. La suciedad se agregó a mano en un archivo aparte, `ingesta_cruda.json`, que simula un lote cargado por la sucursal Norte en su propia planilla. El ETL toma los dos, reporta todo lo que tocó y deja la colección de ChromaDB sincronizada: upsert de lo que sobrevive y `delete` por id de los duplicados.

Lo que se agregó a mano:

| Registro | Qué problema tiene |
| :--- | :--- |
| `DOC-101` | Casi-duplicado de DOC-002 (zona de reparto norte) con otra redacción y otra jerga |
| `DOC-102` | Casi-duplicado de DOC-013 (psicotrópicos) en lenguaje de mostrador |
| `DOC-103` | Casi-duplicado de DOC-007 (medios de pago), escrito en segunda persona |
| `DOC-104` | **Clave mal nombrada**: `descripcion` en vez de `descripcion_semantica`, y `metadata` en vez de `metadatos` |
| `DOC-105` | **Booleano como string** (`"vigente": "true"`) y `tags_regionales` como string con comas en vez de lista |
| `DOC-010` | **Colisión de id**: reusa un id que la base ya tiene, con contenido completamente distinto (dermocosmética vs. PAMI) |

#### El umbral de duplicado, justificado

Antes de fijar el número medimos todos los pares de la base (20 canónicos + 6 entrantes):

| Par | Distancia coseno | |
| :--- | ---: | :--- |
| DOC-007 ↔ DOC-103 | **0,0398** | casi-duplicado real |
| DOC-013 ↔ DOC-102 | **0,0773** | casi-duplicado real |
| DOC-002 ↔ DOC-101 | **0,1115** | casi-duplicado real |
| **DOC-013 ↔ DOC-020** | **0,1248** | **par legítimo más cercano — NO debe fusionarse** |
| DOC-009 ↔ DOC-011 | 0,1580 | par legítimo (dos convenios distintos) |

El umbral tiene que entrar en la ventana `(0,1115 – 0,1248)`: **0,12**. El margen es finito a propósito y el par que lo aprieta es el más interesante de la base — DOC-013 (régimen de psicotrópicos **vigente**) y DOC-020 (el mismo régimen **derogado**) hablan del mismo tema con el mismo vocabulario y se diferencian solo por la vigencia. Cualquier umbral más laxo fusionaría la norma vigente con la derogada, que es exactamente el desastre que la Killer Query 2 pone a prueba.

Por eso rige además una **regla de seguridad**: la purga solo puede eliminar registros que vienen del lote crudo. Un documento canónico nunca se borra automáticamente, por más cerca que caiga de otro.

#### Salida real del ETL

```
$ python etl_purga.py

============ ETL — Normalización de claves y tipos ============
  base canónica :  20 registros
  lote entrante :   6 registros

  [CLAVE] DOC-104: 'descripcion' -> 'descripcion_semantica'
  [CLAVE] DOC-104: 'metadata' -> 'metadatos'
  [TIPO] DOC-105: 'vigente' venía como string 'true' -> bool True
  [TIPO] DOC-105: 'tags_regionales' venía como string -> lista de 5 tags

============ ETL — Colisión de IDs ============
  [ID] DOC-010: el id ya existe en la base con OTRO contenido -> reasignado a DOC-021

============ Purga semántica — distancia coseno <= 0.12 ============

  ELIMINADO DOC-101  (distancia 0.1115 a DOC-002)
    se conserva : DOC-002  "La sucursal Norte cubre el corredor de la zona norte del Gran Buenos Aires..."
    se descarta : DOC-101  "Desde el local de Vicente López mandamos pedidos a toda la zona: Olivos..."

  ELIMINADO DOC-102  (distancia 0.0773 a DOC-013)
    se conserva : DOC-013  "Los psicotrópicos y estupefacientes de las listas reguladas —clonazepam..."
    se descarta : DOC-102  "Las pastillas de la lista controlada, tipo clonazepam, alprazolam o..."

  ELIMINADO DOC-103  (distancia 0.0398 a DOC-007)
    se conserva : DOC-007  "Los medios de pago aceptados en el canal digital son transferencia..."
    se descarta : DOC-103  "Podés abonar por transferencia al CBU o al alias, con tarjeta de..."

============ Resultado ============
  20 canónicos + 6 entrantes - 3 duplicados = 23 documentos
  colección sincronizada: 23 registros
```

**Qué se eliminó:** DOC-101, DOC-102 y DOC-103 — los tres casi-duplicados. Sobrevivieron DOC-104 (la campaña de vacunación), DOC-105 (puericultura) y DOC-021 (dermocosmética, ex DOC-010): son documentos nuevos y legítimos, solo estaban mal cargados. Y sobrevivieron **DOC-013 y DOC-020 por separado**, que es la prueba de que el umbral está bien puesto.

#### Por qué un `SELECT DISTINCT` no habría encontrado nada de esto

`DISTINCT` compara bytes. Los tres pares purgados no comparten ni el `id`, ni el texto, ni una sola oración completa: dicen lo mismo con otras palabras — *"empresa de mensajería"* contra *"servicio de cadetes contratado"*, *"receta oficial archivada"* contra *"la receta que queda en la farmacia"*, *"tarjetas procesadas por la pasarela"* contra *"tarjeta de crédito o débito de cualquier banco a través de la pasarela"*. Para SQL son seis filas perfectamente distintas, y el `DISTINCT` las deja pasar todas. Solo el espacio vectorial las pone a 0,04–0,11 de distancia y las delata como el mismo documento escrito dos veces. Al revés también falla: `DISTINCT` **sí** habría colapsado dos filas idénticas byte a byte que fueran documentos legítimamente repetidos, y jamás habría detectado la colisión de `DOC-010`, donde el id coincide pero el contenido es otro.

---

