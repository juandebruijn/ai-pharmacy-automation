# Costos del agente según LangSmith (proyecto `ai-pharmacy-automation`)

Modelo: `gemini-3.1-flash-lite` · tarifa de referencia: US$ 0.25 / M tokens de entrada y US$ 1.5 / M de salida · 34 corridas analizadas.

| Consulta | Llamadas LLM | Herramientas | Tokens entrada | Tokens salida | Latencia (s) | Costo (US$) |
| :--- | :---: | :--- | ---: | ---: | ---: | ---: |
| ¿Cómo me mandan la insulina, se corta la cadena de frío? | 2 | buscar_politicas | 4,126 | 193 | 3.91 | 0.001321 |
| ¿Qué descuento tengo con OSDE en el Losacor y cuánto sale en el Norte? | 2 | consultar_stock, buscar_politicas | 4,257 | 170 | 2.03 | 0.001319 |
| ¿Tienen omeprazol? Si no hay, ¿me lo pueden encargar? | 3 | buscar_politicas, consultar_stock | 6,802 | 147 | 25.87 | 0.001921 |
| Reservame un Rivotril en el Centro | 3 | buscar_politicas, consultar_stock | 6,399 | 143 | 3.23 | 0.001814 |
| ¿Me venden clonazepam con una receta común? | 2 | buscar_politicas | 4,150 | 126 | 2.24 | 0.001226 |
| Vivo en Olivos, ¿me mandan el pedido a casa? | 2 | buscar_politicas | 4,238 | 90 | 4.15 | 0.001195 |
| ¿Hay Amoxidal 500 en el Oeste? Tengo la receta. | 3 | buscar_politicas, consultar_stock | 6,373 | 150 | 63.58 | 0.001818 |
| ¿Aceptan Mercado Pago con QR? | 2 | buscar_politicas | 4,118 | 52 | 14.27 | 0.001107 |
| Mi DNI es 30.123.456 y mi celular 11 5555-1234, repetímelos para confi | 1 | — | 1,878 | 53 | 12.13 | 0.000549 |
| ¿Me recordás qué te había reservado y hasta cuándo lo tengo que retira | 2 | consultar_reserva | 4,823 | 91 | 9.62 | 0.001342 |
| Hola, ¿tienen Buscapina en la sucursal Oeste? | 2 | consultar_stock | 3,859 | 96 | 1.85 | 0.001109 |
| ¿Me recomendás una película para ver esta noche? | 1 | — | 1,846 | 26 | 0.92 | 0.000500 |
| Reservame 2 cajas de Tafirol 500 en la sucursal Centro, paso a retirar | 3 | reservar_producto, consultar_stock | 6,069 | 116 | 17.19 | 0.001691 |
| ¿Tienen Ibupirac 600 en la sucursal Norte? ¿Cuánto sale? | 2 | consultar_stock | 3,985 | 117 | 4.11 | 0.001172 |
| ¿Hasta qué hora tengo que hacer el pedido para que me llegue hoy a cas | 2 | buscar_politicas | 4,192 | 92 | 27.97 | 0.001186 |
| ¿Tienen omeprazol? Si no hay, ¿me lo pueden encargar? | 3 | buscar_politicas, consultar_stock | 6,802 | 147 | 36.50 | 0.001921 |
| Reservame un Rivotril en el Centro | 3 | buscar_politicas, consultar_stock | 6,399 | 143 | 95.71 | 0.001814 |
| ¿Cómo me mandan la insulina, se corta la cadena de frío? | 2 | buscar_politicas | 4,044 | 167 | 2.08 | 0.001262 |
| ¿Qué descuento tengo con OSDE en el Losacor y cuánto sale en el Norte? | 2 | buscar_politicas, consultar_stock | 4,175 | 201 | 3.39 | 0.001345 |
| ¿Tienen omeprazol? Si no hay, ¿me lo pueden encargar? | 2 | consultar_stock | 3,984 | 57 | 4.29 | 0.001081 |
| Reservame un Rivotril en el Centro | 2 | consultar_stock | 3,758 | 112 | 1.88 | 0.001107 |
| ¿Me venden clonazepam con una receta común? | 2 | buscar_politicas | 4,079 | 126 | 5.02 | 0.001209 |
| Vivo en Olivos, ¿me mandan el pedido a casa? | 2 | buscar_politicas | 4,156 | 86 | 41.37 | 0.001168 |
| ¿Hay Amoxidal 500 en el Oeste? Tengo la receta. | 2 | consultar_stock | 3,781 | 75 | 1.88 | 0.001058 |
| ¿Aceptan Mercado Pago con QR? | 2 | buscar_politicas | 4,036 | 52 | 4.45 | 0.001087 |
| Mi DNI es 30.123.456 y mi celular 11 5555-1234, repetímelos para confi | 1 | — | 1,837 | 53 | 0.93 | 0.000539 |
| ¿Me recordás qué te había reservado y hasta cuándo lo tengo que retira | 2 | consultar_reserva | 4,701 | 92 | 2.61 | 0.001313 |
| Hola, ¿tienen Buscapina en la sucursal Oeste? | 2 | consultar_stock | 3,777 | 89 | 5.74 | 0.001078 |
| ¿Me recordás qué te había reservado y hasta cuándo lo tengo que retira | 2 | consultar_reserva | 4,705 | 92 | 1.88 | 0.001314 |
| Hola, ¿tienen Buscapina en la sucursal Oeste? | 2 | consultar_stock | 3,777 | 89 | 2.58 | 0.001078 |
| ¿Me recomendás una película para ver esta noche? | 1 | — | 1,805 | 26 | 0.73 | 0.000490 |
| Reservame 2 cajas de Tafirol 500 en la sucursal Centro, paso a retirar | 3 | reservar_producto, consultar_stock | 5,948 | 116 | 2.36 | 0.001661 |
| ¿Tienen Ibupirac 600 en la sucursal Norte? ¿Cuánto sale? | 2 | consultar_stock | 3,903 | 111 | 1.79 | 0.001142 |
| ¿Hasta qué hora tengo que hacer el pedido para que me llegue hoy a cas | 2 | buscar_politicas | 4,110 | 122 | 3.91 | 0.001210 |

| Resumen | Valor |
| :--- | ---: |
| Tokens promedio por consulta | 4,427 |
| Proporción entrada / salida | 41 : 1 |
| Llamadas al LLM promedio por consulta | 2.09 |
| Latencia promedio / p95 (s) | 12.12 / 63.58 |
| Costo promedio por consulta (US$) | 0.001240 |
| Costo proyectado 1.000 consultas/día × 30 días (US$) | 37.19 |
