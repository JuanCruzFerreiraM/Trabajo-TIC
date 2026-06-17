import numpy as np
from src.matrices import N,H


SINDROME_A_POS = {
    tuple(H.T[i]): i
    for i in range(N)
} #diccionario que nos permite relacionar los sindromes con sus posiciones

def codificar (U, G):
    U = np.asarray(U, dtype=int)
    return (U @ G) % 2

def decodificar(V,K):
    return V[:,:K].copy() #Al usar forma sistemática el mensaje es las primeras K columnas de V

def sindrome(R,H):
    R = np.asarray(R, dtype=int)
    return (R @ H.T) % 2

def corregir(R,H):
    R = np.asarray(R, dtype=int)
    R_corr = R.copy()
    S = sindrome(R,H)
    for j in range(R_corr.shape[0]):
        if not any(S[j]):
            continue
        
        s = tuple(S[j])
        
        pos = SINDROME_A_POS.get(s)
        if pos is not None:
            R_corr[j,pos] ^= 1
    return R_corr

def detectar(R,H):
    R = np.asarray(R, dtype=int)
    S = sindrome(R,H)
    mask = np.all(S == 0, axis=1)
    return (R[mask], mask)


def _tests():
    from src.matrices import G, K

    rng = np.random.default_rng(42)
    num_bloques = 50
    U = rng.integers(0, 2, size=(num_bloques, K))

    # --- codificar / decodificar (sistemático) ---
    V = codificar(U, G)
    assert V.shape == (num_bloques, N)
    assert np.array_equal(V[:, :K], U), "forma sistemática: primeras K columnas = mensaje"
    assert np.array_equal(decodificar(V, K), U), "decodificar(codificar(U)) == U"

    # --- diccionario de síndromes ---
    assert len(SINDROME_A_POS) == N, "cada posición debe tener un síndrome único"

    # --- síndrome en palabras válidas ---
    S = sindrome(V, H)
    assert np.all(S == 0), "palabras de código válidas → síndrome cero"

    # --- corregir sin errores ---
    assert np.array_equal(corregir(V, H), V), "sin errores, corregir no modifica"

    # --- corregir con 1 error en cada posición ---
    for pos in range(N):
        R = V.copy()
        R[0, pos] ^= 1
        V_rec = corregir(R, H)
        assert np.array_equal(V_rec[0], V[0]), f"corregir falló con 1 error en posición {pos}"

    # --- detectar sin errores ---
    R_ok, mask = detectar(V, H)
    assert np.all(mask), "sin errores, todas las palabras aceptadas"
    assert np.array_equal(R_ok, V), "sin errores, R_ok == R"

    # --- detectar con 1 error descarta esa palabra ---
    R = V.copy()
    R[3, 5] ^= 1
    R_ok, mask = detectar(R, H)
    assert not mask[3], "1 error → palabra descartada"
    assert mask.sum() == num_bloques - 1
    assert R_ok.shape[0] == num_bloques - 1
    assert np.array_equal(R_ok, V[np.arange(num_bloques) != 3])


if __name__ == "__main__":
    _tests()
    print("codec OK: todas las pruebas pasaron")