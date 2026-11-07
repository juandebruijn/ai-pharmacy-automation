"""C.3 — Análisis de costos a partir de las trazas de LangSmith.

Lee las corridas raíz del agente (run_name="agente_farmacia") del proyecto de
LangSmith, y por cada una suma tokens, cuenta llamadas al LLM y herramientas, y
estima el costo con la tarifa de observabilidad.py. Imprime una tabla y la
guarda en entrega_final/resultados/costos_langsmith.md.

Uso (desde la raíz del repo, con LANGSMITH_API_KEY en el .env):
    python -m entrega_final.costos_langsmith            # últimas 50 corridas
    python -m entrega_final.costos_langsmith --n 200
"""

import argparse
import os
from collections import Counter, defaultdict

from entrega_final.config import DIR_FINAL, MODELO_AGENTE
from entrega_final.observabilidad import (
    PRECIO_ENTRADA_USD_MTOK, PRECIO_SALIDA_USD_MTOK, costo_usd, percentil,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Costos del agente desde LangSmith.")
    parser.add_argument("--n", type=int, default=50, help="cantidad de corridas a analizar")
    args = parser.parse_args()

    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit("Falta LANGSMITH_API_KEY en el .env (ver .env.example).")
    from langsmith import Client

    cliente = Client()
    proyecto = os.getenv("LANGSMITH_PROJECT", "default")
    raices = list(cliente.list_runs(project_name=proyecto, is_root=True,
                                    filter='eq(name, "agente_farmacia")', limit=args.n))
    if not raices:
        raise SystemExit(f"No hay corridas de 'agente_farmacia' en el proyecto {proyecto!r}.")

    filas = []
    hijos = defaultdict(list)
    for raiz in raices:
        if raiz.end_time is None:
            continue
        # Las corridas hijas de la traza: cada llamada al LLM y cada herramienta.
        hijos[raiz.trace_id] = list(cliente.list_runs(
            project_name=proyecto, trace_id=raiz.trace_id,
            filter='or(eq(run_type, "llm"), eq(run_type, "tool"))'))
        llms = [r for r in hijos[raiz.trace_id] if r.run_type == "llm"]
        tools = Counter(r.name for r in hijos[raiz.trace_id] if r.run_type == "tool")
        entrada, salida = raiz.prompt_tokens or 0, raiz.completion_tokens or 0
        consulta = ""
        mensajes = (raiz.inputs or {}).get("messages") or []
        if mensajes:
            ultimo = mensajes[-1]
            consulta = ultimo.get("content", "") if isinstance(ultimo, dict) else str(ultimo)
        filas.append({
            "consulta": str(consulta)[:70],
            "llamadas_llm": len(llms),
            "herramientas": ", ".join(f"{k}×{v}" if v > 1 else k for k, v in tools.items()) or "—",
            "entrada": entrada,
            "salida": salida,
            "latencia_s": (raiz.end_time - raiz.start_time).total_seconds(),
            "costo": costo_usd(entrada, salida),
        })

    n = len(filas)
    total_in = sum(f["entrada"] for f in filas)
    total_out = sum(f["salida"] for f in filas)
    costo_total = costo_usd(total_in, total_out)
    por_consulta = costo_total / n
    lineas = [
        f"# Costos del agente según LangSmith (proyecto `{proyecto}`)",
        "",
        f"Modelo: `{MODELO_AGENTE}` · tarifa de referencia: US$ {PRECIO_ENTRADA_USD_MTOK} / M tokens de "
        f"entrada y US$ {PRECIO_SALIDA_USD_MTOK} / M de salida · {n} corridas analizadas.",
        "",
        "| Consulta | Llamadas LLM | Herramientas | Tokens entrada | Tokens salida | Latencia (s) | Costo (US$) |",
        "| :--- | :---: | :--- | ---: | ---: | ---: | ---: |",
    ]
    for f in filas:
        lineas.append(f"| {f['consulta']} | {f['llamadas_llm']} | {f['herramientas']} | {f['entrada']:,} | "
                      f"{f['salida']:,} | {f['latencia_s']:.2f} | {f['costo']:.6f} |")
    latencias = [f["latencia_s"] for f in filas]
    lineas += [
        "",
        "| Resumen | Valor |",
        "| :--- | ---: |",
        f"| Tokens promedio por consulta | {(total_in + total_out) / n:,.0f} |",
        f"| Proporción entrada / salida | {total_in / max(total_out, 1):.0f} : 1 |",
        f"| Llamadas al LLM promedio por consulta | {sum(f['llamadas_llm'] for f in filas) / n:.2f} |",
        f"| Latencia promedio / p95 (s) | {sum(latencias) / n:.2f} / {percentil(latencias, 95):.2f} |",
        f"| Costo promedio por consulta (US$) | {por_consulta:.6f} |",
        f"| Costo proyectado 1.000 consultas/día × 30 días (US$) | {por_consulta * 30_000:.2f} |",
    ]
    texto = "\n".join(lineas)
    print(texto)
    destino = DIR_FINAL / "resultados" / "costos_langsmith.md"
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(texto + "\n", encoding="utf-8")
    print(f"\nGuardado en {destino.relative_to(DIR_FINAL.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
