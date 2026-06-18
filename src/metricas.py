import numpy as np


def estimar_pep(U, U_hat):
    U = np.asarray(U, dtype=int)
    U_hat = np.asarray(U_hat, dtype=int)
    return (U != U_hat).any(axis=1).mean()


def estimar_peb(U, U_hat):
    U = np.asarray(U, dtype=int)
    U_hat = np.asarray(U_hat, dtype=int)
    return (U != U_hat).mean()


def estimar_pep_detector(U, U_hat, mask):
    U = np.asarray(U, dtype=int)
    U_hat = np.asarray(U_hat, dtype=int)
    mask = np.asarray(mask, dtype=bool)
    num_total = U.shape[0]
    num_err = (~mask).sum() + (U[mask] != U_hat).any(axis=1).sum()
    return num_err / num_total


def estimar_peb_detector(U, U_hat, mask, K):
    U = np.asarray(U, dtype=int)
    U_hat = np.asarray(U_hat, dtype=int)
    mask = np.asarray(mask, dtype=bool)
    bits_err = (~mask).sum() * K + (U[mask] != U_hat).sum()
    bits_total = U.shape[0] * K
    return bits_err / bits_total


def _tests():
    from src.matrices import K

    rng = np.random.default_rng(0)
    num_bloques = 100
    U = rng.integers(0, 2, size=(num_bloques, K))

    # --- corrector: sin errores ---
    assert estimar_pep(U, U) == 0.0
    assert estimar_peb(U, U) == 0.0

    # --- corrector: un bit mal en una fila ---
    U_hat = U.copy()
    U_hat[0, 3] ^= 1
    assert estimar_pep(U, U_hat) == 0.01
    assert estimar_peb(U, U_hat) == 0.001

    # --- corrector: fila entera distinta ---
    U_hat = U.copy()
    U_hat[5] = 1 - U_hat[5]
    assert estimar_pep(U, U_hat) == 0.01
    assert estimar_peb(U, U_hat) == K / (num_bloques * K)

    # --- detector: sin descartes ni errores ---
    mask = np.ones(num_bloques, dtype=bool)
    assert estimar_pep_detector(U, U, mask) == 0.0
    assert estimar_peb_detector(U, U, mask, K) == 0.0

    # --- detector: una palabra descartada ---
    mask = np.ones(num_bloques, dtype=bool)
    mask[3] = False
    U_hat = U[mask]
    assert estimar_pep_detector(U, U_hat, mask) == 0.01
    assert estimar_peb_detector(U, U_hat, mask, K) == K / (num_bloques * K)

    # --- detector: aceptada con un bit mal ---
    mask = np.ones(num_bloques, dtype=bool)
    U_hat = U.copy()
    U_hat[7, 2] ^= 1
    assert estimar_pep_detector(U, U_hat, mask) == 0.01
    assert estimar_peb_detector(U, U_hat, mask, K) == 0.001

    # --- detector: coincide con corrector si mask todo True ---
    U_hat = U.copy()
    U_hat[0, 1] ^= 1
    mask = np.ones(num_bloques, dtype=bool)
    assert estimar_pep_detector(U, U_hat, mask) == estimar_pep(U, U_hat)
    assert estimar_peb_detector(U, U_hat, mask, K) == estimar_peb(U, U_hat)


if __name__ == "__main__":
    _tests()
    print("metricas OK: todas las pruebas pasaron")
