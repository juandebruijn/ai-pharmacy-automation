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