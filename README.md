# Automatización de Farmacia con IA

## 📌 Dominio

**Gestión automatizada de stock, consultas y pedidos para farmacias.**
Un sistema con IA que automatiza las consultas de los clientes por WhatsApp, verifica stock y precios en tiempo real contra el ERP de la farmacia y genera las órdenes de despacho, eliminando la carga manual de los empleados.

---

## 👥 Integrantes

* Camila Belén Capua
* Franco Salama
* Juan Pablo Ruax Debruijn
* Lorena Wajchman
* Tomas Rossi
* Miguel Ángel Tadakuma

---

## 📂 Contenido del repositorio

### Entrega 1 — Diagnóstico, arquitectura y pipeline de extracción

| Archivo | Contenido |
| :--- | :--- |
| `informe.md` | Partes A y B completas + narrativa de C.4 y C.5 |
| `schemas.py` | Contrato Pydantic V2 de la extracción (C.1) |
| `app.py` | Pipeline con la API de Gemini y Structured Outputs (C.2) |
| `resultados_lote.md` | Tabla de resultados del lote de prueba (C.3) |

### Entrega 2 — Base de Conocimiento vectorial

| Archivo | Contenido |
| :--- | :--- |
| `base_conocimiento.json` | Los 20 documentos de A.3 — **la fuente de verdad**: toda la base se reconstruye desde acá |
| `pipeline_vectorial.py` | Índice FAISS persistido y prueba de volatilidad (A.4, A.5) |
| `vector_db.py` | ChromaDB persistente, evento en caliente, búsqueda híbrida y killer queries (B.1–B.4, B.6) |
| `ingesta_cruda.json` | Lote sucio cargado a mano: casi-duplicados e inconsistencias estructurales (B.5) |
| `etl_purga.py` | ETL de normalización + purga semántica (B.5) |
| `informe_entrega2.md` | Partes A y B completas + Parte C |
| `resultados_killer_queries.md` | Las 3 killer queries con resultados reales (B.6) |

### Comunes

| Archivo | Contenido |
| :--- | :--- |
| `.env.example` | Variables de entorno, sin valores reales |
| `requirements.txt` | Dependencias con versión |

> **Lo que NO se commitea:** `.env`, `venv/`, el índice binario de FAISS (`indice_faiss/`) y la base de ChromaDB (`chroma_db/`). Los dos últimos se reconstruyen desde `base_conocimiento.json` con los scripts del repo.

---

## 🛠️ Puesta en marcha

### 1. Clonar el repositorio y crear el entorno

```bash
git clone https://github.com/juandebruijn/ai-pharmacy-automation.git
cd ai-pharmacy-automation

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

Requiere **Python 3.10+**.

### 3. Configurar las credenciales

```bash
cp .env.example .env      # en Windows: copy .env.example .env
```

Editar el `.env` recién creado y completar:

| Variable | Obligatoria | Descripción |
| :--- | :---: | :--- |
| `GEMINI_API_KEY` | Sí | Clave de la API de Google Gemini ([aistudio.google.com/apikey](https://aistudio.google.com/apikey)). La usan tanto el pipeline de la Entrega 1 como los embeddings de la Entrega 2. |
| `GEMINI_MODEL` | No | Modelo de chat para `app.py`. Si no se define, usa `gemini-3.6-flash`. |

> ⚠️ El archivo `.env` está en `.gitignore` y nunca se commitea. Solo se versiona `.env.example`, sin valores.

---

## ▶️ Cómo correr la Entrega 2

### A.4 / A.5 — Índice FAISS y prueba de volatilidad

```bash
python pipeline_vectorial.py             # carga de disco si existe; si no, construye y persiste
python pipeline_vectorial.py --volatil   # A.5: construye en RAM y NO persiste
python pipeline_vectorial.py --borrar    # A.5: borra el índice para simular el reinicio
```

La secuencia de la prueba destructiva de A.5, tal cual está en el informe:

```bash
python pipeline_vectorial.py --borrar    # servidor limpio
python pipeline_vectorial.py --volatil   # construye sin write_index()
python pipeline_vectorial.py --volatil   # proceso nuevo: se perdió, hay que pagar de nuevo
python pipeline_vectorial.py             # ahora con write_index()
python pipeline_vectorial.py             # proceso nuevo: read_index(), 0 tokens
```

### B.1 – B.4 y B.6 — ChromaDB, búsqueda híbrida y killer queries

```bash
python vector_db.py --ingesta                       # B.1: crea/actualiza la colección
python vector_db.py --evento                        # B.3: evento en caliente + verificación con get()
python vector_db.py --buscar "me llega hoy?"        # B.4: búsqueda híbrida
python vector_db.py --buscar "..." --categoria logistica --n 5
python vector_db.py --buscar "..." --incluir-no-vigentes
python vector_db.py --killer                        # B.6: las tres killer queries
```

`--ingesta` va primero: los demás comandos consultan la colección.

### B.5 — ETL y purga semántica

```bash
python etl_purga.py
```

Normaliza el lote de `ingesta_cruda.json`, resuelve la colisión de IDs, purga los casi-duplicados y deja la colección de ChromaDB sincronizada.

---

## ▶️ Cómo correr la Entrega 1

```bash
# Corre el input de ejemplo del dominio
python app.py

# O con un input propio
python app.py "Hola, tenés Amoxidal 500 x14? lo necesito hoy, tengo Swiss Medical"
```

**Salida esperada:** los campos extraídos y validados, más la acción de backend y el nivel de riesgo que le corresponde a la intención detectada.

### Códigos de salida

| Código | Significado |
| :---: | :--- |
| `0` | La extracción validó contra el contrato de `schemas.py` |
| `1` | `ValidationError` — el modelo respondió, pero violó el contrato |
| `2` | Error de red o de la API — el pipeline no llegó a validar |

> Los modelos `flash` devuelven `503 UNAVAILABLE` (alta demanda) con cierta frecuencia. El script reintenta hasta 3 veces con backoff exponencial antes de devolver el código `2`.
