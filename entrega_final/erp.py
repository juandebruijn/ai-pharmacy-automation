"""ERP simulado de la farmacia sobre SQLite.

Es la tabla `productos` de la Entrega 1 (B.5.b) más una tabla `reservas`, que es
el actuador "bloquear/reservar stock temporalmente" del PEAS (A.3, Entrega 1).

El LLM nunca ve este módulo: lo usan las herramientas de herramientas.py. Las
reglas duras (stock disponible, psicotrópicos, tope minorista, vencimiento de
48 h) viven acá, en código, y no en el prompt: la IA interpreta, el SQL decide.
"""

import json
import random
import sqlite3
import string
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timedelta

from entrega_final.config import (
    CATALOGO_JSON, DB_ERP, HORAS_RESERVA, MAX_UNIDADES_RESERVA, SUCURSALES,
)

ESQUEMA = """
CREATE TABLE IF NOT EXISTS productos (
    producto_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    sucursal_id       TEXT    NOT NULL,
    codigo_barras     TEXT    NOT NULL,
    droga             TEXT    NOT NULL,
    nombre_comercial  TEXT    NOT NULL,
    presentacion      TEXT    NOT NULL,
    precio_lista      REAL    NOT NULL CHECK (precio_lista >= 0),
    stock_disponible  INTEGER NOT NULL DEFAULT 0 CHECK (stock_disponible >= 0),
    stock_reservado   INTEGER NOT NULL DEFAULT 0 CHECK (stock_reservado >= 0),
    requiere_receta   INTEGER NOT NULL DEFAULT 0,
    psicotropico      INTEGER NOT NULL DEFAULT 0,
    cadena_frio       INTEGER NOT NULL DEFAULT 0,
    UNIQUE (sucursal_id, codigo_barras),
    CHECK (stock_reservado <= stock_disponible)
);
CREATE TABLE IF NOT EXISTS reservas (
    codigo        TEXT    PRIMARY KEY,
    thread_id     TEXT    NOT NULL,
    producto_id   INTEGER NOT NULL REFERENCES productos(producto_id),
    cantidad      INTEGER NOT NULL CHECK (cantidad > 0),
    precio_total  REAL    NOT NULL,
    estado        TEXT    NOT NULL CHECK (estado IN
                    ('confirmada', 'pendiente_receta', 'vencida', 'cancelada')),
    creado_en     TEXT    NOT NULL,
    vence_en      TEXT    NOT NULL
);
"""

ESTADOS_ACTIVOS = ("confirmada", "pendiente_receta")


@contextmanager
def conexion():
    """Una conexión por operación: FastAPI atiende cada request en un hilo distinto."""
    DB_ERP.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_ERP)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def inicializar(forzar: bool = False) -> int:
    """Crea las tablas y carga el catálogo si la base está vacía. Devuelve la cantidad de filas."""
    with conexion() as con:
        if forzar:
            con.executescript("DROP TABLE IF EXISTS reservas; DROP TABLE IF EXISTS productos;")
        con.executescript(ESQUEMA)
        if con.execute("SELECT COUNT(*) FROM productos").fetchone()[0] == 0:
            catalogo = json.loads(CATALOGO_JSON.read_text(encoding="utf-8"))["productos"]
            for p in catalogo:
                for sucursal in SUCURSALES:
                    con.execute(
                        "INSERT INTO productos (sucursal_id, codigo_barras, droga, nombre_comercial, "
                        "presentacion, precio_lista, stock_disponible, requiere_receta, psicotropico, "
                        "cadena_frio) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (sucursal, p["codigo_barras"], p["droga"], p["nombre_comercial"],
                         p["presentacion"], p["precio_lista"][sucursal], p["stock"][sucursal],
                         p["requiere_receta"], p["psicotropico"], p["cadena_frio"]),
                    )
        return con.execute("SELECT COUNT(*) FROM productos").fetchone()[0]


def _normalizar(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sin_tildes.lower()


def _fila_a_dict(fila: sqlite3.Row) -> dict:
    return {
        "producto_id": fila["producto_id"],
        "sucursal_id": fila["sucursal_id"],
        "sucursal": SUCURSALES[fila["sucursal_id"]],
        "nombre_comercial": fila["nombre_comercial"],
        "droga": fila["droga"],
        "presentacion": fila["presentacion"],
        "precio_lista_ars": fila["precio_lista"],
        "unidades_disponibles": fila["stock_disponible"] - fila["stock_reservado"],
        "requiere_receta": bool(fila["requiere_receta"]),
        "psicotropico": bool(fila["psicotropico"]),
        "cadena_frio": bool(fila["cadena_frio"]),
    }


def buscar_productos(texto: str, sucursal_id: str | None = None) -> list[dict]:
    """
    Busca por nombre comercial o droga, sin importar mayúsculas ni tildes. Basta
    con que coincida una palabra de 4+ letras: "ibupirac 600" encuentra Ibupirac.
    """
    palabras = [p for p in _normalizar(texto).replace(",", " ").split() if len(p) >= 4]
    with conexion() as con:
        filas = con.execute(
            "SELECT * FROM productos " + ("WHERE sucursal_id = ? " if sucursal_id else "")
            + "ORDER BY nombre_comercial, presentacion, sucursal_id",
            (sucursal_id,) if sucursal_id else (),
        ).fetchall()
    resultado = []
    for fila in filas:
        nombre = _normalizar(f"{fila['nombre_comercial']} {fila['droga']}")
        if any(p in nombre for p in palabras):
            resultado.append(_fila_a_dict(fila))
    return resultado


def _codigo_reserva() -> str:
    return "RES-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def crear_reserva(producto_id: int, cantidad: int, thread_id: str) -> dict:
    """
    Reserva stock por 48 h (DOC-005). Devuelve {"ok": True, ...} o {"ok": False, "motivo": ...}.

    El descuento de stock es un UPDATE condicional: si entre la consulta y la
    reserva otro cliente se llevó las unidades, el UPDATE no toca ninguna fila y
    la reserva no se crea. No hay forma de reservar más de lo que hay.
    """
    if not 1 <= cantidad <= MAX_UNIDADES_RESERVA:
        return {"ok": False, "motivo": f"La cantidad debe estar entre 1 y {MAX_UNIDADES_RESERVA} "
                                       "unidades: WhatsApp es un canal minorista."}
    with conexion() as con:
        fila = con.execute("SELECT * FROM productos WHERE producto_id = ?", (producto_id,)).fetchone()
        if fila is None:
            return {"ok": False, "motivo": f"No existe el producto_id {producto_id}. "
                                           "Usá consultar_stock para obtener ids válidos."}
        producto = _fila_a_dict(fila)
        if producto["psicotropico"]:
            # DOC-013: los psicotrópicos no se venden por WhatsApp ni se reservan.
            return {"ok": False, "producto": producto,
                    "motivo": "Es un psicotrópico de lista regulada: no se reserva ni se vende por "
                              "WhatsApp. La dispensa es exclusivamente presencial con receta oficial "
                              "[DOC-013]."}
        actualizado = con.execute(
            "UPDATE productos SET stock_reservado = stock_reservado + ? "
            "WHERE producto_id = ? AND stock_disponible - stock_reservado >= ?",
            (cantidad, producto_id, cantidad),
        ).rowcount
        if not actualizado:
            return {"ok": False, "producto": producto,
                    "motivo": f"Stock insuficiente en la sucursal {producto['sucursal']}: hay "
                              f"{producto['unidades_disponibles']} unidades disponibles."}

        ahora = datetime.now().replace(microsecond=0)
        reserva = {
            "codigo": _codigo_reserva(),
            "thread_id": thread_id,
            "producto_id": producto_id,
            "cantidad": cantidad,
            "precio_total": round(producto["precio_lista_ars"] * cantidad, 2),
            # Con receta, la reserva queda retenida hasta que el farmacéutico la valide.
            "estado": "pendiente_receta" if producto["requiere_receta"] else "confirmada",
            "creado_en": ahora.isoformat(),
            "vence_en": (ahora + timedelta(hours=HORAS_RESERVA)).isoformat(),
        }
        con.execute(
            "INSERT INTO reservas VALUES (:codigo, :thread_id, :producto_id, :cantidad, "
            ":precio_total, :estado, :creado_en, :vence_en)",
            reserva,
        )
    return {"ok": True, "reserva": {**reserva, "producto": producto["nombre_comercial"],
                                    "presentacion": producto["presentacion"],
                                    "sucursal": producto["sucursal"]}}


def obtener_reservas(thread_id: str, codigo: str | None = None) -> list[dict]:
    """
    Reservas de una conversación. Si alguna superó las 48 h, se marca vencida y
    se libera el stock (DOC-005: "el stock se libera automáticamente").
    """
    ahora = datetime.now().isoformat()
    with conexion() as con:
        vencidas = con.execute(
            "SELECT codigo, producto_id, cantidad FROM reservas "
            "WHERE estado IN (?, ?) AND vence_en < ?", (*ESTADOS_ACTIVOS, ahora),
        ).fetchall()
        for r in vencidas:
            con.execute("UPDATE reservas SET estado = 'vencida' WHERE codigo = ?", (r["codigo"],))
            con.execute("UPDATE productos SET stock_reservado = stock_reservado - ? "
                        "WHERE producto_id = ?", (r["cantidad"], r["producto_id"]))

        consulta = ("SELECT r.*, p.nombre_comercial, p.presentacion, p.sucursal_id FROM reservas r "
                    "JOIN productos p USING (producto_id) WHERE r.thread_id = ?")
        parametros: tuple = (thread_id,)
        if codigo:
            consulta += " AND r.codigo = ?"
            parametros += (codigo.strip().upper(),)
        filas = con.execute(consulta + " ORDER BY r.creado_en", parametros).fetchall()
    return [
        {"codigo": f["codigo"], "producto": f["nombre_comercial"], "presentacion": f["presentacion"],
         "sucursal": SUCURSALES[f["sucursal_id"]], "cantidad": f["cantidad"],
         "precio_total": f["precio_total"], "estado": f["estado"], "vence_en": f["vence_en"]}
        for f in filas
    ]


if __name__ == "__main__":
    print(f"ERP inicializado en {DB_ERP}: {inicializar(forzar=True)} filas en productos.")
