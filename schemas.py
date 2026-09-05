"""
C.1 — El contrato en código.

Traduce el System Prompt y el JSON de B.5 a un modelo Pydantic V2.
Este archivo es la ÚNICA fuente de verdad del contrato: el mismo conjunto de
intenciones aparece en la Matriz de B.3, en el CHECK de la tabla `interacciones`
(B.5.b) y en la tabla de resultados de C.3.
"""

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

# --- Las 6 intenciones cerradas. Mismos valores que el CHECK de `interacciones`. ---
Intencion = Literal[
    "consulta_stock",
    "consulta_precio_cobertura",
    "consulta_envio",
    "validar_receta",
    "crear_pedido",
    "fuera_de_alcance",
]

# Coberturas con convenio vigente. El LLM puede escribirlas de cualquier forma;
# el validador las normaliza contra este catálogo o rechaza la extracción.
COBERTURAS_CONOCIDAS = {
    "OSDE",
    "SWISS_MEDICAL",
    "GALENO",
    "OMINT",
    "MEDIFE",
    "PAMI",
    "IOMA",
    "OSECAC",
    "PARTICULAR",
}

# Cómo la escribe la gente en WhatsApp -> cómo la guarda la base.
ALIAS_COBERTURAS = {
    "SWISSMEDICAL": "SWISS_MEDICAL",
    "SWISS": "SWISS_MEDICAL",
    "SM": "SWISS_MEDICAL",
    "OSDE BINARIO": "OSDE",
    "GALENO ARGENTINA": "GALENO",
    "MEDIFE ARGENTINA": "MEDIFE",
    "INSSJP": "PAMI",
    "SIN OBRA SOCIAL": "PARTICULAR",
    "NINGUNA": "PARTICULAR",
    "NO TENGO": "PARTICULAR",
}

# Regla de negocio: WhatsApp es canal minorista. Un pedido mayorista va por
# el circuito comercial, no por el bot.
MAX_UNIDADES_MINORISTA = 20

# Límite de la columna productos.nombre_comercial / productos.presentacion (B.5.b).
MAX_LARGO_TEXTO = 120


class ConsultaFarmacia(BaseModel):
    """Salida estructurada que el LLM debe producir a partir de `texto_libre`."""

    intencion: Intencion = Field(
        description="Una de las 6 intenciones permitidas de la Matriz de B.3."
    )
    producto: Optional[str] = Field(
        default=None,
        description="Nombre comercial o droga tal como lo escribió el cliente. null si no lo menciona.",
    )
    presentacion: Optional[str] = Field(
        default=None,
        description="Dosis y/o cantidad de unidades del envase. null si no la menciona.",
    )
    cantidad: Optional[int] = Field(
        default=None,
        description="Cantidad de envases pedidos. null si no la menciona.",
    )
    obra_social: Optional[str] = Field(
        default=None,
        description="Cobertura mencionada por el cliente. null si no la menciona.",
    )
    requiere_envio: Optional[bool] = Field(
        default=None,
        description="true si pide envío, false si retira en sucursal, null si no lo menciona.",
    )
    urgencia: Optional[Literal["alta", "normal"]] = Field(
        default=None,
        description="'alta' solo si el cliente expresa que lo necesita hoy o urgente.",
    )
    confianza: float = Field(
        description="Qué tan claro fue el mensaje, entre 0.0 y 1.0.",
    )

    # ------------------------------------------------------------------ #
    # Validadores
    # ------------------------------------------------------------------ #

    @field_validator("producto", "presentacion", mode="before")
    @classmethod
    def limpiar_texto_libre(cls, v: object, info: ValidationInfo) -> Optional[str]:
        """Normaliza el texto que después va a una columna VARCHAR(120) de la base.

        El modelo devuelve cosas como "  Ibupirac   600 ", la cadena vacía o el
        string literal "null" en vez de null. Todo eso se colapsa a None para que
        el resto del pipeline tenga un solo valor que signifique "no lo sé".
        """
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError(f"{info.field_name} debe ser texto o null, llegó {type(v).__name__}")

        limpio = re.sub(r"\s+", " ", v).strip(" .,;-")
        if limpio.lower() in {"", "null", "none", "n/a", "no especifica", "no especificado"}:
            return None
        if len(limpio) > MAX_LARGO_TEXTO:
            raise ValueError(
                f"{info.field_name} excede {MAX_LARGO_TEXTO} caracteres "
                f"({len(limpio)}): el modelo devolvió texto libre en vez de un dato"
            )
        return limpio

    @field_validator("obra_social", mode="before")
    @classmethod
    def normalizar_cobertura(cls, v: object) -> Optional[str]:
        """Normaliza la cobertura y la valida contra el catálogo de convenios.

        Pasa de "osde 210" / "Swiss Medical" / "swissmedical" a la clave canónica
        que espera la columna clientes.obra_social. Si la cobertura no tiene
        convenio, se rechaza la extracción entera: informar un descuento de una
        obra social inexistente es el error financiero que documentamos en A.2.
        """
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError(f"obra_social debe ser texto o null, llegó {type(v).__name__}")

        texto = re.sub(r"\s+", " ", v).strip()
        if texto.lower() in {"", "null", "none", "n/a"}:
            return None

        # "OSDE 210", "Galeno plan 330" -> se descarta el número de plan.
        texto = re.sub(r"\b(plan|nivel)\b", " ", texto, flags=re.IGNORECASE)
        texto = re.sub(r"\d+", " ", texto)
        clave = re.sub(r"\s+", " ", texto).strip().upper()
        clave = ALIAS_COBERTURAS.get(clave, clave.replace(" ", "_").replace("-", "_"))

        if clave not in COBERTURAS_CONOCIDAS:
            raise ValueError(
                f"cobertura sin convenio vigente: '{v}'. "
                f"Válidas: {', '.join(sorted(COBERTURAS_CONOCIDAS))}"
            )
        return clave

    @field_validator("cantidad")
    @classmethod
    def validar_rango_minorista(cls, v: Optional[int]) -> Optional[int]:
        """Rechaza cantidades fuera del rango de venta minorista."""
        if v is None:
            return None
        if v < 1:
            raise ValueError(f"cantidad debe ser >= 1, llegó {v}")
        if v > MAX_UNIDADES_MINORISTA:
            raise ValueError(
                f"cantidad {v} supera el máximo minorista de {MAX_UNIDADES_MINORISTA} "
                "envases: el pedido debe derivarse al circuito mayorista"
            )
        return v

    @field_validator("confianza")
    @classmethod
    def validar_confianza(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confianza debe estar entre 0.0 y 1.0, llegó {v}")
        return round(v, 2)

    @model_validator(mode="after")
    def coherencia_intencion_parametros(self) -> "ConsultaFarmacia":
        """Reglas cruzadas que el JSON Schema por sí solo no puede expresar.

        `crear_pedido` escribe en la tabla `pedidos` (riesgo ALTO en B.3): no se
        puede disparar sin saber QUÉ producto. Y `fuera_de_alcance` no puede
        traer parámetros: si el modelo clasificó el mensaje como hostil o ajeno
        al dominio y aun así extrajo datos, la clasificación no es confiable.
        """
        if self.intencion == "crear_pedido" and self.producto is None:
            raise ValueError(
                "crear_pedido requiere 'producto': no se puede generar una orden "
                "de despacho sin saber qué medicamento pidió el cliente"
            )

        if self.intencion == "fuera_de_alcance":
            con_datos = [
                nombre
                for nombre in ("producto", "presentacion", "cantidad", "obra_social")
                if getattr(self, nombre) is not None
            ]
            if con_datos:
                raise ValueError(
                    "fuera_de_alcance no admite parámetros extraídos, "
                    f"pero llegaron: {', '.join(con_datos)}"
                )
        return self
