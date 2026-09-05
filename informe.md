## Parte A — Diagnóstico y Arquitectura

---

### A.1 — El caso

* **Empresa:** Farmacia de barrio / red comercial farmacéutica.
* **Proceso a automatizar:** La atención de consultas por WhatsApp, la verificación manual de disponibilidad de stock/precios y la carga del pedido para despacho. 
* **Situación actual:** Los empleados deben interrumpir la atención presencial para consultar el sistema o ir al depósito, lo que genera cuellos de botella, pérdida de clientes por demoras y carga manual duplicada.
* **Solución propuesta:** El sistema de IA automatizará la recepción del pedido, la consulta en tiempo real al ERP (Sistema de Planificación de Recursos Empresariales) para verificar stock y la generación automática de la orden de despacho.

---

### A.2 — Evidencia de la necesidad ("El Proveedor Enojado")

#### 💬 Prompt enviado al modelo (Gemini)
> *"Actuá como el sistema de atención por WhatsApp de Farmacia San Juan. Un usuario pregunta: 'Hola! Tienen stock de Ibupirac 600mg por 20 comprimidos, cuánto sale con descuento de OSDE y me lo pueden mandar hoy a la tarde?' Respondé como lo haría el sistema."*

#### 🤖 Respuesta obtenida del modelo
> *"¡Hola! Sí, contamos con stock disponible de Ibupirac 600mg x 20 comprimidos. El precio de lista es de $4.500, y con la cobertura del 40% de OSDE te queda en $2.700. Podemos realizar el envío a tu domicilio hoy entre las 15:00 y las 18:00 hs. ¿Te gustaría que te lo reservemos?"*

#### 🔍 Documentación del fallo

* **Qué inventó (Alucinación):**
  * **Stock disponible:** Afirmó contar con unidades en la sucursal sin consultar inventarios.
  * **Precios y cobertura:** Inventó el precio de lista y el porcentaje de descuento de la obra social sin validar el vademécum ni la credencial/plan del afiliado.
  * **Logística:** Asignó rangos horarios de entrega y factibilidad de delivery sin consultar la disponibilidad del servicio.
  * **Nivel de confianza:** *Extremo*. El modelo respondió de forma directa, afirmativa y profesional, sin advertir que carecía de acceso a la base de datos de la farmacia.

* **Qué le faltó al modelo para responder correctamente:**
  * Conexión en tiempo real a la base de datos (SQL) del ERP/sistema de gestión farmacéutico para verificar stock por lote/vencimiento.
  * Integración con el vademécum y el validador online de obras sociales y prepagas.
  * Acceso a la matriz de reglas de negocio para delivery y normativas de venta (Venta Libre vs. Venta Bajo Receta).

> ⚠️ **Conclusión de la prueba:** El modelo no falla por ser "malo", sino por la ausencia de una **Base de Conocimiento** integrada. La alucinación es la consecuencia directa de forzar al LLM a responder sobre datos dinámicos privados sin la arquitectura de integración correspondiente.

---

### A.3 — PEAS extendido (5 pilares)

| Pilar | Pregunta que responde | Aplicado al sistema (Farmacia) |
| :--- | :--- | :--- |
| *Performance* | ¿Cómo medimos el éxito? | • Tasa de resolución/automatización de consultas frecuentes de stock y precio (>85%).<br>• Reducción del tiempo de respuesta por WhatsApp (<15 segundos).<br>• Tasa de error en precios, descuentos y stock informado (0%).<br>• Precisión en la creación del pedido/borrador de despacho en el sistema (>98%). |
| *Environment* | ¿Dónde opera? ¿Sistemas legados? | • Canal de cara al cliente: WhatsApp Business API.<br>• Sistema legado de gestión farmacéutica (base de datos relacional para inventarios, precios y recetas).<br>• Servidor/Middleware de orquestación del agente de IA. |
| *Actuators* | ¿Qué puede hacer? | • Enviar mensajes de texto y opciones vía WhatsApp.<br>• Generar una solicitud/borrador de despacho de pedido en el ERP.<br>• Bloquear/reservar stock temporalmente en la base de datos.<br>• Solicitar al cliente la foto/imagen de la receta o credencial médica. |
| *Sensors* | ¿Qué recibe y por qué canal? | • Mensajes de texto, imágenes de recetas médicas/credencial de OS por WhatsApp Webhook.<br>• Número de teléfono / ID del cliente.<br>• Datos de confirmación de stock, precios y validez de obras sociales enviados por las APIs internas del sistema. |
| *Base de Conocimiento* | ¿Qué sabe el sistema? | • Base de datos estructurada SQL (catálogo de medicamentos/perfumería, stock en tiempo real, precios de lista).<br>• Vademécum institucional y convenios de obras sociales/prepagas.<br>• Base vectorial (RAG) con políticas de entrega a domicilio, zonas de cobertura, medios de pago y requisitos normativos para medicamentos bajo receta. |

---

### A.4 — Anatomía del token

#### 💻 Código ejecutado en Python

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

consulta_es = "Hola! Tienen stock de Ibupirac 600mg por 20 comprimidos, cuánto sale con descuento de OSDE y me lo pueden mandar hoy a la tarde?"
consulta_en = "Hi! Do you have stock of Ibupirac 600mg 20 tablets, how much is it with OSDE discount and can you send it today in the afternoon?"

tokens_es = len(enc.encode(consulta_es))
tokens_en = len(enc.encode(consulta_en))

print(f"ES: {tokens_es} tokens")
print(f"EN: {tokens_en} tokens")

📊 Resultado del conteo
ES: 36 tokens
EN: 36 tokens

💡 Reflexión técnica
Las frases a procesar son principalmente en español, lo cual en modelos anteriores a GPT-4o eran penalizados frente a frases en ingles. A partir de este modelo la cantidad de tokens para procesar frases en español son similares a las de ingles, esto sumado a que ciertas palabras de las frases a procesar son técnicas (nombres de medicamentos, nombre de obra social etc), podemos concluir que procesar frases en español en textos cortos como un chat no va a encarecer el costo frente a procesar las mismas frases en ingles.
\