import numpy as np

# Parámetros 
N = 14
K = 10
Q = N - K

# Matriz generadora G
G = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    ],
    dtype=int,
)

# Matriz de control de paridad H 
H = np.array(
    [
        [1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1],
    ],
    dtype=int,
)


# Esto corresponde a un test, no afecta el informe y no es necesario correrlo para el mismo
if __name__ == "__main__":
    assert G.shape == (K, N)
    assert H.shape == (Q, N)
    assert np.all((G @ H.T) % 2 == 0), "G @ H.T debe ser cero en GF(2)"
    print("G y H OK:", G.shape, H.shape)
