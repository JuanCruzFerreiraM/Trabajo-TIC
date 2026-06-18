"""
Utilidades de fuente para el Ejercicio 3 (codificación de fuente / Huffman).

Convención de bits: False/negro → 0, True/blanco → 1 (vía astype(int)).
Fuente extendida de orden n: se agrupan n bits consecutivos sin solapamiento;
cada bloque se mapea a un símbolo entero 0 … 2^n-1 (bits MSB primero).
"""

from pathlib import Path

import numpy as np
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOGO_FI = DATA_DIR / "logo FI.tif"
ALTO = 434
ANCHO = 432


def cargar_imagen_bits(ruta):
    """
    Lee una imagen y devuelve sus píxeles como secuencia 1D de bits 0/1.

    Recorre la imagen fila por fila (orden C de NumPy). Si hay varios canales
    (p. ej. RGB), usa solo el primero.
    """
    arr = np.array(Image.open(ruta))
    if arr.ndim > 2:
        arr = arr[:, :, 0]
    return arr.astype(int).flatten()


def bloques_extendidos(bits, n):
    """
    Agrupa la secuencia en bloques de longitud n (fuente extendida).

    Descarta los últimos len(bits) % n bits si hiciera falta.
    Cada bloque [b0, b1, …, b_{n-1}] se convierte al entero
    b0·2^(n-1) + … + b_{n-1}  (equivalente a m0=00…, m1=01… en las filminas).

    Parámetros
    ----------
    bits : secuencia de 0/1
    n : orden de la fuente extendida (2 o 3 en el TP)

    Retorna
    -------
    ndarray de enteros en [0, 2^n - 1], un símbolo por bloque.
    """
    bits = np.asarray(bits, dtype=int).flatten()
    bits = bits[: (len(bits) // n) * n]
    bloques = bits.reshape(-1, n)
    pesos = 2 ** np.arange(n - 1, -1, -1)
    return bloques @ pesos


def estimar_probabilidades(simbolos, n=None):
    """
    Frecuencias relativas de cada símbolo (estimación máxima verosímil).

    Parámetros
    ----------
    simbolos : secuencia de enteros (salida de bloques_extendidos)
    n : orden de la fuente; si se pasa, incluye todos los 2^n símbolos
        posibles con probabilidad 0 si no aparecen.

    Retorna
    -------
    dict {símbolo: probabilidad} que suma 1.
    """
    simbolos = np.asarray(simbolos, dtype=int)
    total = len(simbolos)
    if total == 0:
        raise ValueError("No hay símbolos para estimar probabilidades.")

    max_simbolo = int(2**n) if n is not None else int(simbolos.max()) + 1
    conteos = np.bincount(simbolos, minlength=max_simbolo)
    probs = conteos[:max_simbolo] / total
    return {i: float(probs[i]) for i in range(max_simbolo)}


def simbolos_a_bits(simbolos, n):
    simbolos = np.asarray(simbolos, dtype=int).flatten()
    shifts = np.arange(n - 1, -1, -1)
    return ((simbolos[:, None] >> shifts) & 1).astype(int).ravel()


def reconstruir_imagen(bits, alto=ALTO, ancho=ANCHO):
    bits = np.asarray(bits, dtype=int).flatten()
    total = alto * ancho
    img = np.zeros(total, dtype=int)
    n = min(len(bits), total)
    img[:n] = bits[:n]
    return img.reshape(alto, ancho)


def guardar_imagen_bits(bits, ruta, alto=ALTO, ancho=ANCHO):
    arr = (reconstruir_imagen(bits, alto, ancho) * 255).astype(np.uint8)
    Image.fromarray(arr).save(ruta)


def _tests():
    # --- cargar_imagen_bits ---
    assert LOGO_FI.exists(), f"No se encontró {LOGO_FI}"
    bits = cargar_imagen_bits(LOGO_FI)
    assert bits.ndim == 1
    assert set(np.unique(bits)).issubset({0, 1})
    assert len(bits) == 434 * 432
    assert 0.45 < bits.mean() < 0.55, "blanco/negro deberían ser ~equiprobables"

    # --- bloques_extendidos: caso manual orden 2 ---
    bits_test = np.array([0, 0, 1, 1, 0, 1])
    sim = bloques_extendidos(bits_test, 2)
    # 00→0, 11→3, 01→1
    assert np.array_equal(sim, [0, 3, 1])

    # --- bloques_extendidos: orden 3 ---
    bits3 = np.array([0, 0, 0, 0, 0, 1])
    assert bloques_extendidos(bits3, 3).tolist() == [0, 1]

    # --- simbolos_a_bits: inverso de bloques_extendidos ---
    sim_rev = bloques_extendidos(bits_test, 2)
    assert np.array_equal(simbolos_a_bits(sim_rev, 2), bits_test)

    # --- logo: cantidad de bloques ---
    sim2 = bloques_extendidos(bits, 2)
    sim3 = bloques_extendidos(bits, 3)
    assert len(sim2) == len(bits) // 2
    assert len(sim3) == len(bits) // 3

    # --- estimar_probabilidades ---
    probs2 = estimar_probabilidades(sim2, n=2)
    assert len(probs2) == 4
    assert abs(sum(probs2.values()) - 1.0) < 1e-12
    assert all(p >= 0 for p in probs2.values())

    probs3 = estimar_probabilidades(sim3, n=3)
    assert len(probs3) == 8
    assert abs(sum(probs3.values()) - 1.0) < 1e-12

    # orden 2: con equiprobables bits NO todos los símbolos son 0.25
    assert not np.allclose(list(probs2.values()), 0.25)

    print("fuente OK: todas las pruebas pasaron")
    print(f"  bits totales: {len(bits)}, P(1)={bits.mean():.4f}")
    print(f"  orden 2: {probs2}")
    print(f"  orden 3 (no cero): { {k: v for k, v in probs3.items() if v > 0} }")


if __name__ == "__main__":
    _tests()
