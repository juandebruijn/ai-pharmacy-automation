"""
Entrega 3 — Parte C: evaluación del pipeline RAG con RAGAS.

C.1  Golden dataset: 10 preguntas del dominio con su respuesta esperada.
C.2  Baseline RAGAS sobre el RAG básico: Faithfulness, Answer Relevancy,
     Context Precision y Context Recall.
C.3  Misma evaluación sobre el RAG avanzado (chunking + reranking) y tabla
     comparativa con el delta de cada métrica.

Uso (desde la raíz del repo):
    python entrega_3/evaluacion_ragas.py --pipeline basico      # C.2
    python entrega_3/evaluacion_ragas.py --pipeline avanzado    # C.3
    python entrega_3/evaluacion_ragas.py --comparar             # C.3: tabla básico vs. avanzado

Cada caso medido se guarda en entrega_3/resultados_ragas/<pipeline>.json apenas
termina: si la cuota gratuita de Gemini corta la corrida, se vuelve a correr el
mismo comando y sigue desde el primer caso pendiente (--reiniciar empieza de cero).
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

# RAGAS manda telemetría anónima por defecto; se apaga antes de importarlo.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
# instructor (lo usa RAGAS por dentro) loguea cada 429 completo; ya lo manejamos acá.
logging.getLogger("instructor").setLevel(logging.CRITICAL)

from google import genai  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402
from ragas.embeddings import GoogleEmbeddings  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics.collections import (  # noqa: E402
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from rag_pipeline import (  # noqa: E402
    FRASE_ESCAPE,
    construir_rag_avanzado,
    construir_rag_basico,
)

AQUI = Path(__file__).resolve().parent
DIR_RESULTADOS = AQUI / "resultados_ragas"
METRICAS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# El juez no es el modelo que genera las respuestas: cada modelo tiene su propia
# cuota diaria en la capa gratuita, y el juez hace muchas más llamadas que el RAG.
MODELO_JUEZ = "gemini-3.5-flash-lite"
EMBEDDINGS_JUEZ = "gemini-embedding-001"
PAUSA_ENTRE_CASOS = 4  # segundos, para no chocar el límite por minuto

# --------------------------------------------------------------------------- #
# C.1 — Golden dataset
# --------------------------------------------------------------------------- #
# Las respuestas de referencia están escritas a mano a partir de los documentos
# de base_conocimiento.json (y de los que sumó el ETL de B.5 en la Entrega 2).
# "docs" es solo documentación: RAGAS no lo usa.
GOLDEN_SET = [
    # --- Simples: la respuesta está en un solo chunk ---------------------------
    {
        "id": "simple-01", "tipo": "simple", "docs": ["DOC-007"],
        "pregunta": "¿Aceptan cheques o dólares como forma de pago?",
        "ground_truth": "No. No se aceptan cheques ni pagos en moneda extranjera. Los medios "
                        "aceptados son transferencia bancaria por CBU o alias, tarjetas de crédito "
                        "y débito, billeteras virtuales con código QR y efectivo solo contra entrega.",
    },
    {
        "id": "simple-02", "tipo": "simple", "docs": ["DOC-004"],
        "pregunta": "¿Hasta qué hora tengo que confirmar el pedido un día de semana para que me llegue en el día?",
        "ground_truth": "De lunes a viernes el pedido tiene que estar pagado y con la receta validada "
                        "antes de las dieciséis horas para salir en la última ronda de reparto; si se "
                        "confirma después, pasa a la jornada siguiente.",
    },
    {
        "id": "simple-03", "tipo": "simple", "docs": ["DOC-014"],
        "pregunta": "¿Puedo usar la misma receta electrónica en dos sucursales distintas?",
        "ground_truth": "No. Cada receta electrónica se marca como consumida al dispensarla, lo que "
                        "impide que el mismo código se use dos veces en dos sucursales de la red.",
    },
    # --- Complejas: hay que combinar chunks -----------------------------------
    {
        "id": "compleja-01", "tipo": "compleja", "docs": ["DOC-008", "DOC-009"],
        "pregunta": "Tengo OSDE y quiero pagar con la promo del banco, ¿se me suman los dos descuentos?",
        "ground_truth": "No. Los descuentos bancarios nunca se acumulan con el descuento de la obra "
                        "social: el sistema aplica automáticamente el beneficio que más le conviene al "
                        "cliente. Las promociones bancarias se limitan a perfumería, dermocosmética, "
                        "cuidado personal y accesorios. Para el descuento de OSDE hace falta la "
                        "credencial digital o el número de afiliado y una receta vigente a nombre del titular.",
    },
    {
        "id": "compleja-02", "tipo": "compleja", "docs": ["DOC-019", "DOC-003"],
        "pregunta": "Si no tienen el remedio, ¿me lo encargan y me lo mandan a Ramos Mejía el sábado?",
        "ground_truth": "Se puede encargar a la droguería con un plazo estimado de cuarenta y ocho a "
                        "setenta y dos horas hábiles, y no se cobra hasta que la mercadería entra al "
                        "depósito. Pero Ramos Mejía la cubre la sucursal Oeste, que reparte de lunes a "
                        "viernes con una única salida a las quince horas: no hay reparto los sábados.",
    },
    {
        "id": "compleja-03", "tipo": "compleja", "docs": ["DOC-013"],
        "pregunta": "¿Puedo comprar clonazepam por WhatsApp y que me lo manden a casa?",
        "ground_truth": "No. El clonazepam es un psicotrópico: no se vende por WhatsApp ni se despacha "
                        "a domicilio. Solo se dispensa en forma presencial contra receta oficial "
                        "archivada, con troquel, firma y sello del profesional, que el farmacéutico "
                        "verifica en papel contra el documento de quien retira. La receta queda "
                        "archivada en la farmacia.",
    },
    # --- Escape: el sistema tiene que decir que no sabe -----------------------
    {
        "id": "escape-01", "tipo": "escape", "docs": [],
        "pregunta": "¿Me pueden tomar la presión en la farmacia?",
        "ground_truth": FRASE_ESCAPE,
    },
    {
        "id": "escape-02", "tipo": "escape", "docs": [],
        "pregunta": "¿Cuánto sale la caja de ibuprofeno de 400?",
        "ground_truth": FRASE_ESCAPE,
    },
    # --- Lenguaje ambiguo o informal ------------------------------------------
    {
        "id": "informal-01", "tipo": "informal", "docs": ["DOC-006"],
        "pregunta": "che si me mandan la insu no se me corta lo del frio??",
        "ground_truth": "La insulina viaja en conservadora con gel refrigerante y registro de "
                        "temperatura, y solo se despacha en la primera ronda de reparto de la mañana. "
                        "La recibe el titular o alguien autorizado. Si la temperatura se sale de rango "
                        "en el trayecto, el producto se da de baja y se repone sin cargo.",
    },
    {
        "id": "informal-02", "tipo": "informal", "docs": ["DOC-010"],
        "pregunta": "mi vieja tiene pami, le sale gratis lo de la presion o q onda",
        "ground_truth": "Sí, si el medicamento para la presión está en el listado de medicamentos "
                        "gratuitos de PAMI para tratamientos crónicos, no abona nada. Necesita una "
                        "receta de un médico de cartilla con sello y matrícula y la credencial de "
                        "afiliada; no alcanza con el documento.",
    },
]

# --------------------------------------------------------------------------- #
# Juez: Gemini + las 4 métricas core de RAGAS
# --------------------------------------------------------------------------- #
def crear_metricas() -> dict:
    """
    El LLM juez va por el endpoint de Gemini compatible con OpenAI: RAGAS 0.4 (vía
    instructor) tiene problemas con el cliente async de google-genai. Sigue siendo
    Gemini con la misma GEMINI_API_KEY.
    """
    clave = os.environ["GEMINI_API_KEY"]
    juez = llm_factory(MODELO_JUEZ, client=AsyncOpenAI(
        api_key=clave, base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    ))
    embeddings = GoogleEmbeddings(client=genai.Client(api_key=clave), model=EMBEDDINGS_JUEZ)
    return {
        "faithfulness": Faithfulness(llm=juez),
        # strictness = cuántas preguntas genera el juez a partir de la respuesta. Con 1
        # se gasta un tercio de la cuota.
        "answer_relevancy": AnswerRelevancy(llm=juez, embeddings=embeddings, strictness=1),
        "context_precision": ContextPrecision(llm=juez),
        "context_recall": ContextRecall(llm=juez),
    }


def con_reintentos(fn, intentos: int = 4):
    """La capa gratuita corta con 429 (límite por minuto) o 503 (modelo saturado):
    se espera y se reintenta. Si es el límite diario no tiene sentido esperar."""
    for intento in range(intentos):
        try:
            return fn()
        except Exception as e:
            if "PerDay" in str(e):
                raise SystemExit("Cuota diaria de Gemini agotada. Lo medido quedó guardado: "
                                 "volvé a correr el mismo comando mañana y sigue desde ahí.")
            reintentable = any(c in str(e) for c in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"))
            if not reintentable or intento == intentos - 1:
                raise
            print("    (Gemini pidió esperar: rate limit o saturación, espero 30 s...)")
            time.sleep(30)


def medir(metricas: dict, pregunta: str, respuesta: str, contextos: list[str], referencia: str) -> dict:
    """Cada métrica recibe solo las columnas que necesita."""
    return {
        "faithfulness": con_reintentos(lambda: metricas["faithfulness"].score(
            user_input=pregunta, response=respuesta, retrieved_contexts=contextos)).value,
        "answer_relevancy": con_reintentos(lambda: metricas["answer_relevancy"].score(
            user_input=pregunta, response=respuesta)).value,
        "context_precision": con_reintentos(lambda: metricas["context_precision"].score(
            user_input=pregunta, reference=referencia, retrieved_contexts=contextos)).value,
        "context_recall": con_reintentos(lambda: metricas["context_recall"].score(
            user_input=pregunta, reference=referencia, retrieved_contexts=contextos)).value,
    }


def se_abstuvo(respuesta: str) -> bool:
    # Se busca en todo el texto y no solo al principio: ante un precio, el prompt
    # (regla 2) hace que el modelo primero aclare que no lo puede confirmar. Una
    # pregunta con respuesta que trae la frase de escape a mitad de camino también
    # cuenta: es una abstención parcial donde el sistema debería haber respondido.
    return FRASE_ESCAPE.split(".")[0].lower() in respuesta.lower()


# --------------------------------------------------------------------------- #
# Ejecución
# --------------------------------------------------------------------------- #
def evaluar(nombre: str, reiniciar: bool = False) -> None:
    rag = construir_rag_avanzado() if nombre == "avanzado" else construir_rag_basico()
    metricas = crear_metricas()

    DIR_RESULTADOS.mkdir(exist_ok=True)
    ruta = DIR_RESULTADOS / f"{nombre}.json"
    filas = [] if reiniciar or not ruta.exists() else json.loads(ruta.read_text(encoding="utf-8"))
    hechos = {f["id"] for f in filas}
    if hechos:
        print(f"Retomando: {len(hechos)} casos ya medidos en {ruta.name}")

    for caso in GOLDEN_SET:
        if caso["id"] in hechos:
            continue
        print(f"\n[{caso['id']}] {caso['pregunta']}")
        salida = con_reintentos(lambda: rag.invoke(caso["pregunta"]))
        # Para RAGAS el contexto es lo que efectivamente llegó al LLM: en el
        # avanzado, los chunks que eligió el juez, en el orden en que los dejó.
        contextos = [d.page_content for d in salida["documentos"]]
        print(f"  respuesta: {salida['respuesta'][:110]}")

        abstencion_ok = se_abstuvo(salida["respuesta"]) == (caso["tipo"] == "escape")
        if caso["tipo"] == "escape":
            # Sin ground truth útil las cuatro métricas no aplican (NaN): lo que se
            # mide en estas dos preguntas es si el sistema se abstuvo.
            puntajes = {m: float("nan") for m in METRICAS}
        else:
            puntajes = medir(metricas, caso["pregunta"], salida["respuesta"], contextos,
                             caso["ground_truth"])
        print("  " + " | ".join(f"{m}={v:.2f}" for m, v in puntajes.items())
              + f" | abstencion_ok={abstencion_ok}")

        filas.append({
            **caso,
            "respuesta": salida["respuesta"],
            "fuentes": [d.id for d in salida["documentos"]],
            "contextos": contextos,
            "abstencion_ok": abstencion_ok,
            **puntajes,
        })
        ruta.write_text(json.dumps(filas, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(PAUSA_ENTRE_CASOS)

    imprimir_resumen(nombre, filas)


def promedio(valores: list[float]) -> float:
    validos = [v for v in valores if not math.isnan(v)]  # NaN = la métrica no aplica
    return sum(validos) / len(validos) if validos else float("nan")


def imprimir_resumen(nombre: str, filas: list[dict]) -> None:
    print(f"\n{'=' * 78}\nRAG {nombre.upper()} — promedio sobre los casos donde la métrica aplica")
    for m in METRICAS:
        aplicables = [f for f in filas if not math.isnan(f[m])]
        peor = min(aplicables, key=lambda f: f[m])
        print(f"  {m:<18} {promedio([f[m] for f in filas]):.3f}   peor: {peor['id']} ({peor[m]:.2f})")
    print(f"  {'abstención ok':<18} {sum(f['abstencion_ok'] for f in filas)}/{len(filas)}")

    print("\nPor tipo de pregunta:")
    for tipo in ("simple", "compleja", "informal"):
        del_tipo = [f for f in filas if f["tipo"] == tipo]
        print(f"  {tipo:<9}" + "  ".join(f"{m}={promedio([f[m] for f in del_tipo]):.2f}"
                                          for m in METRICAS))


# --------------------------------------------------------------------------- #
# C.3 — Tabla comparativa
# --------------------------------------------------------------------------- #
def comparar() -> None:
    corridas = {}
    for nombre in ("basico", "avanzado"):
        ruta = DIR_RESULTADOS / f"{nombre}.json"
        if not ruta.exists():
            raise SystemExit(f"Falta {ruta.name}: corré primero --pipeline {nombre}.")
        corridas[nombre] = json.loads(ruta.read_text(encoding="utf-8"))

    print("| Métrica | RAG básico | RAG avanzado | Delta |")
    print("| :--- | :---: | :---: | :---: |")
    for m in METRICAS:
        basico = promedio([f[m] for f in corridas["basico"]])
        avanzado = promedio([f[m] for f in corridas["avanzado"]])
        print(f"| {m} | {basico:.3f} | {avanzado:.3f} | {avanzado - basico:+.3f} |")
    print("| abstención correcta | "
          + " | ".join(f"{sum(f['abstencion_ok'] for f in corridas[n])}/{len(corridas[n])}"
                       for n in ("basico", "avanzado")) + " | |")

    print("\n| Caso | " + " | ".join(f"{m} (B → A)" for m in METRICAS) + " |")
    print("| :--- |" + " :---: |" * len(METRICAS))
    avanzado_por_id = {f["id"]: f for f in corridas["avanzado"]}
    for b in corridas["basico"]:
        a = avanzado_por_id[b["id"]]
        celdas = ["n/a" if math.isnan(b[m]) else f"{b[m]:.2f} → {a[m]:.2f}" for m in METRICAS]
        print(f"| {b['id']} | " + " | ".join(celdas) + " |")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluación RAGAS del pipeline RAG (Entrega 3, Parte C).")
    parser.add_argument("--pipeline", choices=["basico", "avanzado"], help="pipeline a evaluar.")
    parser.add_argument("--reiniciar", action="store_true", help="descarta lo medido y empieza de cero.")
    parser.add_argument("--comparar", action="store_true", help="C.3: tabla básico vs. avanzado.")
    args = parser.parse_args()

    if args.pipeline:
        evaluar(args.pipeline, args.reiniciar)
    if args.comparar:
        comparar()
    if not (args.pipeline or args.comparar):
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
