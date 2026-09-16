# B.6 — Killer Queries

Tres consultas trampa contra la colección `politicas_farmacia` (20 documentos).

**Reproducible con:** `python vector_db.py --killer`
**Modelo:** `gemini-embedding-001`, 768 dimensiones
**Métrica:** distancia coseno (`hnsw:space="cosine"`) — más chica = más parecido
**Umbral de aceptación:** distancia ≤ 0,35 (justificado en `informe_entrega2.md`, C.2)

> **Sobre reproducir estos números.** La API de embeddings es un servicio hospedado y
> devuelve vectores con variación de milésimas entre corridas, así que las distancias se
> mueven en el tercer decimal. Donde eso importa es en la Killer Query 3: los tres
> candidatos están a 0,369–0,379, tan pegados que el orden del top-3 puede cambiar de una
> corrida a otra. Lo que **no** cambia es el veredicto, que es lo único que la prueba
> evalúa: los tres siguen por encima del umbral y el sistema responde "no tengo esa
> información". En las Killer Queries 1 y 2 el margen es amplio y el ganador es estable.

---

## Tabla de resultados

| # | Consulta | Qué pone a prueba | Resultado esperado | Resultado real | ¿Pasó? |
| :-: | :--- | :--- | :--- | :--- | :-: |
| **1** | *"mi vieja tiene 80 años y toma pastillas para el corazón todos los días, ¿le sale algo o no paga nada?"*<br><br>`filtro_categoria="cobertura"` | **Poder semántico: jerga sin ninguna palabra del documento.** La consulta no dice *PAMI*, ni *jubilado*, ni *cobertura*, ni *crónico*, ni *vademécum*, ni *presión arterial*, ni *afiliado*. Ni una de las palabras clave del documento que la responde. Un `LIKE '%...%'` devuelve **cero filas**. | DOC-010 (convenio PAMI: jubilados, crónicos de presión, medicamentos gratuitos) en el puesto 1, por debajo del umbral. | **DOC-010 · 0,2929 · ACEPTADO**<br>Los puestos 2 y 3 (DOC-011 a 0,3524 y DOC-009 a 0,3618) quedan **por encima del umbral** y se descartan: no solo acierta el primero, además reconoce que los otros dos convenios no responden esta pregunta. | ✅ |
| **2** | *"el médico me hizo la receta en su papel con membrete, ¿sirve para el alprazolam? ¿me lo pueden enviar a domicilio?"*<br><br>`solo_vigentes` en `False` vs. `True` | **El metadato salva el día.** DOC-020 es el régimen de psicotrópicos **derogado** (`vigente: false`), que permitía receta manuscrita y envío a domicilio. Describe *exactamente* lo que el cliente pregunta, así que la semántica cruda lo pone **primero**. Si el orquestador se lo pasa al LLM, el sistema habilita una dispensa que hoy es una **infracción sanitaria**. | **(a) sin filtro:** DOC-020 gana → desastre.<br>**(b) con `{"vigente": {"$eq": True}}` en el `where`:** DOC-020 ni siquiera es candidato y gana DOC-013, el régimen vigente. | **(a) SIN filtro:**<br>1. **DOC-020 · 0,2485 · `vigente=False`** ← el derogado gana<br>2. DOC-013 · 0,2736<br>3. DOC-014 · 0,3060<br><br>**(b) CON el filtro en el `where`:**<br>1. **DOC-013 · 0,2736 · `vigente=True`** ← el vigente<br>2. DOC-014 · 0,3060<br>3. DOC-001 · 0,3332 | ✅ |
| **3** | *"¿hacen análisis de sangre y electrocardiogramas en la farmacia?"* | **Prueba de estrés: consulta fuera del catálogo.** La farmacia no presta servicios de diagnóstico y no hay ningún documento sobre el tema. Pero el índice **nunca devuelve vacío**: siempre existe un vecino más cercano. Lo que decide que el sistema no invente es el umbral, no la búsqueda. | Los tres resultados por encima de 0,35 → `hay_respuesta: False` → *"no tengo esa información"*. | **Ningún resultado supera el umbral:**<br>1. DOC-011 · 0,3695 · descartado<br>2. DOC-018 · 0,3779 · descartado<br>3. DOC-010 · 0,3790 · descartado<br><br>Respuesta emitida:<br>*"No tengo esa información en mi base de conocimiento. Te derivo con un empleado de la farmacia."* | ✅ |

---

## Salida literal de la corrida

```
$ python vector_db.py --killer

[B.1] Colección 'politicas_farmacia' en chroma_db/ (PersistentClient, hnsw:space=cosine)
      upsert de 20 documentos: 23 -> 23 registros.

==============================================================================
KILLER QUERY 1 — Poder semántico: jerga sin ninguna palabra del documento
==============================================================================
Consulta: "mi vieja tiene 80 años y toma pastillas para el corazón todos los días, ¿le sale algo o no paga nada?"
where:    {"$and": [{"vigente": {"$eq": true}}, {"categoria": {"$eq": "cobertura"}}]}
umbral:   distancia <= 0.35
  1. DOC-010  distancia=0.2929  [ACEPTADO]
     categoria=cobertura  vigente=True
     Los jubilados y pensionados afiliados al PAMI acceden al vademécum del instituto con cobertura preferencial y,...
  2. DOC-011  distancia=0.3524  [descartado por umbral]
     categoria=cobertura  vigente=True
     Los convenios con Swiss Medical, OMINT y Medifé funcionan bajo el mismo circuito de validación en línea y comp...
  3. DOC-009  distancia=0.3618  [descartado por umbral]
     categoria=cobertura  vigente=True
     El convenio con OSDE cubre a los afiliados de todos los planes con un porcentaje de descuento que varía según ...

==============================================================================
KILLER QUERY 2 — El metadato salva el día
==============================================================================
DOC-020 es el régimen de psicotrópicos DEROGADO (vigente=False) y describe
exactamente lo que el cliente pregunta, así que la semántica cruda lo pone primero.

--- (a) SIN el filtro de vigencia ---
Consulta: "el médico me hizo la receta en su papel con membrete, ¿sirve para el alprazolam? ¿me lo pueden enviar a domicilio?"
where:    null
umbral:   distancia <= 0.35
  1. DOC-020  distancia=0.2485  [ACEPTADO]
     categoria=normativa  vigente=False
     Régimen anterior de dispensa de psicotrópicos, derogado y fuera de vigencia. Bajo este régimen alcanzaba con u...
  2. DOC-013  distancia=0.2736  [ACEPTADO]
     categoria=normativa  vigente=True
     Los psicotrópicos y estupefacientes de las listas reguladas —clonazepam, alprazolam, zolpidem, metilfenidato y...
  3. DOC-014  distancia=0.3060  [ACEPTADO]
     categoria=normativa  vigente=True
     La receta electrónica se acepta en todas las sucursales siempre que traiga el código de validación o el QR emi...

--- (b) CON el filtro de vigencia en el where ---
Consulta: "el médico me hizo la receta en su papel con membrete, ¿sirve para el alprazolam? ¿me lo pueden enviar a domicilio?"
where:    {"vigente": {"$eq": true}}
umbral:   distancia <= 0.35
  1. DOC-013  distancia=0.2736  [ACEPTADO]
     categoria=normativa  vigente=True
     Los psicotrópicos y estupefacientes de las listas reguladas —clonazepam, alprazolam, zolpidem, metilfenidato y...
  2. DOC-014  distancia=0.3060  [ACEPTADO]
     categoria=normativa  vigente=True
     La receta electrónica se acepta en todas las sucursales siempre que traiga el código de validación o el QR emi...
  3. DOC-001  distancia=0.3332  [ACEPTADO]
     categoria=logistica  vigente=True
     La sucursal Centro despacha pedidos a domicilio dentro del microcentro porteño y los barrios linderos: San Nic...

==============================================================================
KILLER QUERY 3 — Prueba de estrés: consulta fuera del catálogo
==============================================================================
Consulta: "¿hacen análisis de sangre y electrocardiogramas en la farmacia?"
where:    {"vigente": {"$eq": true}}
umbral:   distancia <= 0.35
  1. DOC-011  distancia=0.3695  [descartado por umbral]
     categoria=cobertura  vigente=True
     Los convenios con Swiss Medical, OMINT y Medifé funcionan bajo el mismo circuito de validación en línea y comp...
  2. DOC-018  distancia=0.3779  [descartado por umbral]
     categoria=atencion  vigente=True
     Los medicamentos no admiten cambio ni devolución una vez que salieron del mostrador, porque la trazabilidad y ...
  3. DOC-010  distancia=0.3790  [descartado por umbral]
     categoria=cobertura  vigente=True
     Los jubilados y pensionados afiliados al PAMI acceden al vademécum del instituto con cobertura preferencial y,...

  >> Sin coincidencias sobre el umbral. El sistema responde:
     "No tengo esa información en mi base de conocimiento. Te derivo con un empleado de la farmacia."
```

---

## Qué muestra cada una

**KQ1 — la búsqueda semántica hace lo que ningún `LIKE` puede.** El puente entre *"mi vieja de 80 años"* y *"jubilados y pensionados afiliados al PAMI"*, y entre *"pastillas para el corazón todos los días"* y *"tratamientos crónicos de presión arterial... incluidos en el listado de medicamentos gratuitos"*, lo cruza el embedding solo, sin diccionario de sinónimos y sin que nadie haya anticipado esa forma de preguntar.

**KQ2 — la semántica sola es peligrosa, no solo insuficiente.** El caso es incómodo justamente porque el modelo *no se equivoca*: DOC-020 **es** el documento más parecido a lo que el cliente preguntó. La semántica funcionó perfecto y el resultado es una infracción sanitaria. Lo que separa la respuesta correcta de la incorrecta no es un mejor modelo de embeddings ni un prompt más largo: es un `bool` en los metadatos y un `$eq` en el `where`. Esto es lo que la Regla del Arquitecto quiere decir cuando dice que todo lo que se filtra duro va como metadato — y el motivo por el que el filtro tiene que ir **dentro** del `where` y no en un `if` posterior: con post-filtering, el motor igual habría calculado la similitud contra el documento derogado y, con `n_results=3`, nos habríamos quedado con dos resultados sin darnos cuenta.

**KQ3 — el umbral es la única defensa contra la alucinación por proximidad.** La búsqueda vectorial no tiene concepto de "no encontré nada": devuelve los *k* más cercanos aunque los *k* sean basura. Acá el más cercano es el convenio con Swiss Medical a 0,3695 — un documento sobre credenciales de prepaga contra una pregunta sobre electrocardiogramas. Sin umbral, el orquestador se lo pasaría al LLM y el LLM redactaría, con toda la seguridad del mundo, una respuesta sobre análisis clínicos apoyada en un texto que habla de otra cosa. Es la alucinación de A.2 de la Entrega 1, ahora con fuente citable.
