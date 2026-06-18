import numpy as np
from scipy.stats import norm


def transmitir_codificado(V, EbfN0, n, k, A=1.0):
    V = np.asarray(V, dtype=int)
    Es = A ** 2
    Ebf = Es * n / k
    S = (2*V - 1) * A
    N0 = Ebf / EbfN0
    form = V.shape
    noise = np.sqrt(N0/2)*(np.random.randn(*form) + 1 * 1j*np.random.randn(*form))
    R = S + noise
    VR = 1 * (np.real(R) > 0)
    return VR.astype(int)


def transmitir_sin_codificar(U, EbfN0, A=1.0):
    U = np.asarray(U, dtype=int)
    Es = A ** 2
    Ebf = Es 
    S = (2*U - 1) * A
    N0 = Ebf / EbfN0
    form = U.shape
    noise = np.sqrt(N0/2)*(np.random.randn(*form) + 1 * 1j*np.random.randn(*form))
    R = S + noise
    VR = 1 * (np.real(R) > 0)
    return VR.astype(int)


def _Q(x):
    """Función Q de Gauss: P(Z > x), Z ~ N(0,1)."""
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

    # --- a igual EbfN0, el canal codificado erra más por bit de canal
    # (cada bit de canal tiene menos energía: Es = k/n * Ebf) ---
    assert p_emp > p_emp_sc, "codificado debe tener mayor p de bit de canal a igual Ebf/N0"


if __name__ == "__main__":
    _tests()
    print("canal OK: todas las pruebas pasaron")
