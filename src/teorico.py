import numpy as np
from scipy.stats import norm
from scipy.special import comb


def Q(x):
    """Función Q de Gauss: P(Z > x), Z ~ N(0,1). Acepta escalar o array."""
    return norm.sf(x)


def p_canal(EbfN0, n, k):
    """Probabilidad de error por bit de canal (CSB) antes del decodificador."""
    return Q(np.sqrt((2 * k / n) * EbfN0))


def peb_sin_cod(EbfN0):
    """Probabilidad de error de bit de fuente sin codificar (BPSK)."""
    return Q(np.sqrt(2 * EbfN0))


def pep_corrector(p, n, tc):
    """Probabilidad de error de palabra con código corrector de tc errores."""
    pep = 0.0
    for i in range(tc + 1, n + 1):
        pep += comb(n, i, exact=False) * (p**i) * ((1 - p) ** (n - i))
    return pep


def peb_corrector(p, n, tc):
    """Aproximación de Peb de fuente post-decodificación (p << 1)."""
    return ((2 * tc + 1) / n) * comb(n, tc + 1, exact=False) * (p ** (tc + 1))


def _tests():
    from src.matrices import K, N

    TC = 1

    # --- Q básica ---
    assert np.isclose(Q(0), 0.5)
    assert Q(10) < 1e-10

    # --- consistencia con fórmulas del informe ---
    EbfN0 = 4.0
    assert np.isclose(p_canal(EbfN0, N, K), Q(np.sqrt(2 * K / N * EbfN0)))
    assert np.isclose(peb_sin_cod(EbfN0), Q(np.sqrt(2 * EbfN0)))

    # --- vectorización ---
    ebf_vec = np.array([1.0, 4.0, 10.0])
    p_vec = p_canal(ebf_vec, N, K)
    assert p_vec.shape == ebf_vec.shape
    assert np.all(p_vec > 0) and np.all(p_vec < 1)

    # --- orden esperado a igual EbfN0 ---
    p = p_canal(EbfN0, N, K)
    assert p > peb_sin_cod(EbfN0), "p canal > Peb sin codificar a igual Ebf/N0"

    # --- pep y peb corrector ---
    pep = pep_corrector(p, N, TC)
    peb = peb_corrector(p, N, TC)
    assert 0 < pep < 1
    assert 0 < peb < 1
    assert pep >= peb, "Pep debe ser >= Peb aproximado"

    # --- caso conocido n=14, tc=1: Peb = (3/14) * C(14,2) * p^2 ---
    peb_esperado = (3 / 14) * comb(14, 2, exact=False) * p**2
    assert np.isclose(peb, peb_esperado)

    # --- límites ---
    assert pep_corrector(0.0, N, TC) == 0.0
    assert pep_corrector(1.0, N, TC) == 1.0

    # --- SNR alto: probabilidades muy bajas ---
    p_bajo = p_canal(100.0, N, K)
    assert pep_corrector(p_bajo, N, TC) < 1e-6
    assert peb_corrector(p_bajo, N, TC) < 1e-6


if __name__ == "__main__":
    _tests()
    print("teorico OK: todas las pruebas pasaron")
