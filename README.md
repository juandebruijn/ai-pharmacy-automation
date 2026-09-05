# AI Pharmacy Automation

## 📌 Domain

**Automated Stock, Inquiry, and Order Management for Pharmacies.**
An AI-driven system designed to automate customer inquiries via WhatsApp, check real-time stock and pricing against the pharmacy's ERP system, and generate dispatch orders—eliminating manual overhead for pharmacy staff.

---

## 👥 Team Members

* Camila Belén Capua
* Franco Salama
* Juan Pablo Ruax Debruijn
* Lorena Wajchman
* Tomas Rossi
* Miguel Ángel Tadakuma

---

## 📂 Contenido del repositorio

| Archivo | Contenido |
| :--- | :--- |
| `informe.md` | Partes A y B completas + narrativa de C.4 y C.5 |
| `schemas.py` | Contrato Pydantic V2 de la extracción (C.1) |
| `app.py` | Pipeline con la API de Gemini y Structured Outputs (C.2) |
| `resultados_lote.md` | Tabla de resultados del lote de prueba (C.3) |
| `.env.example` | Variables de entorno, sin valores reales |
| `requirements.txt` | Dependencias del script |

---

## 🛠️ Cómo correr el script

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

Requiere **Python 3.10+**. Las dependencias son `pydantic>=2.9`, `google-genai>=1.0` y `python-dotenv>=1.0`.

### 3. Configurar las credenciales

```bash
cp .env.example .env      # en Windows: copy .env.example .env
```

Editar el `.env` recién creado y completar:

| Variable | Obligatoria | Descripción |
| :--- | :---: | :--- |
| `GEMINI_API_KEY` | Sí | Clave de la API de Google Gemini. Se obtiene en [aistudio.google.com/apikey](https://aistudio.google.com/apikey). |
| `GEMINI_MODEL` | No | Modelo a usar. Si no se define, el script usa `gemini-3.6-flash`. |

> ⚠️ El archivo `.env` está en `.gitignore` y nunca se commitea. Solo se versiona `.env.example`, sin valores.

### 4. Ejecutar

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
