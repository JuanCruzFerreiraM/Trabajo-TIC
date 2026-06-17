import numpy as np


def transmitir_codificado(V, EbfN0, n, k, A=1.0):
    """
    Simula el canal AWGN con BPSK antipodal y detección dura para un sistema
    **con codificación de canal**.

    Teoría
    ------
    Cada fila de V es una palabra de n bits de canal. Se modula en BPSK antipodal:
    bit 0 -> -A, bit 1 -> +A. El ruido es AWGN; con detección dura el canal
    equivale a un CSB con probabilidad de error por bit de canal:

        p = Q( sqrt(2 * Es / N0) ) = Q( sqrt(2 * k/n * Ebf/N0) )

    Energías (enunciado / filminas):
        Es  = A^2                     (energía de símbolo de canal)
        Ebf = Es * n / k              (energía de bit de fuente)
        N0  = Ebf / EbfN0

    El modelo del enunciado usa ruido complejo con varianza N0/2 por dimensión
    y decide el bit con umbral 0 sobre la parte real.

    Consideraciones de implementación
    ---------------------------------
    - V debe ser int 0/1, forma (num_bloques, n).
    - EbfN0 es lineal (no dB): Ebf/N0.
    - Vectorizar sobre toda la matriz (sin for por palabra).
    - Salida R misma forma que V, dtype int, valores solo 0 y 1.
    - Usar np.random.randn con ruido complejo (1j) como el MATLAB del TP.
    - A=1 fijo es suficiente; solo escala energías, no cambia las curvas.

    Parámetros
    ----------
    V : array (num_bloques, n)
        Palabras de código a transmitir.
    EbfN0 : float
        Cociente Ebf/N0 deseado (lineal).
    n, k : int
        Largo de palabra de código y de fuente ((14, 10) en este TP).
    A : float
        Amplitud BPSK.

    Retorna
    -------
    R : array (num_bloques, n)
        Bits recibidos tras canal y detección dura.
    """
    raise NotImplementedError


def transmitir_sin_codificar(U, EbfN0, A=1.0):
    """
    Simula el canal AWGN con BPSK antipodal y detección dura para el sistema
    **sin codificación** (referencia del Ejercicio 1).

    Teoría
    ------
    Cada bit de fuente se transmite directamente, sin bits de paridad.
    La energía por bit de fuente es Ebf = Es = A^2 (un bit por símbolo).
    La probabilidad de error de bit de fuente es la del enunciado:

        Peb_sc = Q( sqrt(2 * Ebf / N0) )

    Con N0 = Ebf / EbfN0.

    Consideraciones de implementación
    ---------------------------------
    - U forma (num_bloques, k); salida misma forma.
    - **No** aplicar el factor n/k en la energía: cada bit de fuente lleva Ebf.
    - Misma modulación, ruido y detección dura que transmitir_codificado.
    - Reutilizar la misma lógica interna si querés (DRY), cambiando solo Ebf.

    Parámetros
    ----------
    U : array (num_bloques, k)
        Bits de fuente.
    EbfN0 : float
        Cociente Ebf/N0 (lineal).
    A : float
        Amplitud BPSK.

    Retorna
    -------
    R : array (num_bloques, k)
        Bits recibidos.
    """
    raise NotImplementedError


def _Q(x):
    """Función Q de Gauss: P(Z > x), Z ~ N(0,1)."""
    from scipy.stats import norm

    return norm.sf(x)


def _tasa_error_bits(recibido, original):
    """Fracción de bits distintos entre dos matrices de igual forma."""
    return np.mean(recibido != original)


def _tests():
    from src.matrices import G, K, N

    rng = np.random.default_rng(0)
    num_bloques = 5000
    U = rng.integers(0, 2, size=(num_bloques, K))
    V = (U @ G) % 2

    EbfN0_alto = 100.0  # SNR alto → casi sin errores
    EbfN0_medio = 4.0   # para comparar con p teórico

    # --- transmitir_codificado: forma y tipo ---
    R = transmitir_codificado(V, EbfN0_alto, N, K)
    assert R.shape == V.shape, "salida codificada debe tener forma de V"
    assert R.dtype == int or np.issubdtype(R.dtype, np.integer)
    assert set(np.unique(R)).issubset({0, 1})

    # --- transmitir_sin_codificar: forma y tipo ---
    R_sc = transmitir_sin_codificar(U, EbfN0_alto)
    assert R_sc.shape == U.shape, "salida sin codificar debe tener forma de U"
    assert set(np.unique(R_sc)).issubset({0, 1})

    # --- SNR alto: muy pocos errores ---
    assert _tasa_error_bits(R, V) < 0.01, "EbfN0 alto → p_error bit canal baja"
    assert _tasa_error_bits(R_sc, U) < 0.01, "EbfN0 alto → p_error bit fuente baja"

    # --- tasa empírica vs teórica (codificado, bits de canal) ---
    R = transmitir_codificado(V, EbfN0_medio, N, K)
    p_emp = _tasa_error_bits(R, V)
    p_teo = _Q(np.sqrt(2 * (K / N) * EbfN0_medio))
    assert np.isclose(p_emp, p_teo, rtol=0.25), (
        f"p empírico canal {p_emp:.4f} vs teórico {p_teo:.4f}"
    )

    # --- tasa empírica vs teórica (sin codificar, bits de fuente) ---
    R_sc = transmitir_sin_codificar(U, EbfN0_medio)
    p_emp_sc = _tasa_error_bits(R_sc, U)
    p_teo_sc = _Q(np.sqrt(2 * EbfN0_medio))
    assert np.isclose(p_emp_sc, p_teo_sc, rtol=0.25), (
        f"p empírico fuente {p_emp_sc:.4f} vs teórico {p_teo_sc:.4f}"
    )

    # --- a igual EbfN0, sin codificar erra más (más Eb efectivo por bit) ---
    assert p_emp_sc > p_emp, "sin codificar debe tener mayor p a igual Ebf/N0"


if __name__ == "__main__":
    _tests()
    print("canal OK: todas las pruebas pasaron")
