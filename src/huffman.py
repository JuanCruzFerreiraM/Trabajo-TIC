import heapq
from itertools import count

import numpy as np


def _asignar_codigos(nodo, prefijo="", codigos=None):
    if codigos is None:
        codigos = {}
    if isinstance(nodo, int):
        codigos[nodo] = prefijo or "0"
        return codigos
    izq, der = nodo
    _asignar_codigos(izq, prefijo + "0", codigos)
    _asignar_codigos(der, prefijo + "1", codigos)
    return codigos


def construir_huffman(probs):
    heap = []
    tie = count()
    for simbolo, p in probs.items():
        if p > 0:
            heapq.heappush(heap, (p, next(tie), simbolo))

    if not heap:
        raise ValueError("No hay símbolos con probabilidad positiva.")
    if len(heap) == 1:
        return {heap[0][2]: "0"}

    while len(heap) > 1:
        p1, _, n1 = heapq.heappop(heap)
        p2, _, n2 = heapq.heappop(heap)
        heapq.heappush(heap, (p1 + p2, next(tie), (n1, n2)))

    return _asignar_codigos(heap[0][2])


def codificar(simbolos, codigos):
    simbolos = np.asarray(simbolos, dtype=int)
    return "".join(codigos[int(s)] for s in simbolos)


def decodificar(bits, codigos):
    inv = {c: s for s, c in codigos.items()}
    prefijos = set()
    for c in codigos.values():
        for i in range(1, len(c) + 1):
            prefijos.add(c[:i])

    simbolos = []
    buffer = ""
    for b in bits:
        buffer += b
        if buffer in inv:
            simbolos.append(inv[buffer])
            buffer = ""
        elif buffer not in prefijos:
            raise ValueError(f"Secuencia inválida cerca de '{buffer}'")
    if buffer:
        raise ValueError("Secuencia de bits incompleta al final.")
    return simbolos


def largo_promedio_teorico(probs, codigos, n=1):
    # bits por símbolo Huffman, normalizado a bits por bit de fuente
    l_por_simbolo = sum(p * len(codigos[k]) for k, p in probs.items() if p > 0)
    return l_por_simbolo / n


def largo_promedio_empirico(simbolos, codigos, n):
    simbolos = np.asarray(simbolos, dtype=int)
    bits_cod = codificar(simbolos, codigos)
    return len(bits_cod) / (len(simbolos) * n)


def entropia(probs):
    return -sum(p * np.log2(p) for p in probs.values() if p > 0)


def tasa_compresion(l_bar):
    return 1.0 / l_bar


def _es_prefijo(codigos):
    vals = list(codigos.values())
    for i, ci in enumerate(vals):
        for j, cj in enumerate(vals):
            if i != j and cj.startswith(ci):
                return False
    return True


def _tests():
    probs_clasico = {0: 0.5, 1: 0.25, 2: 0.125, 3: 0.125}
    codigos_clasico = construir_huffman(probs_clasico)
    assert np.isclose(largo_promedio_teorico(probs_clasico, codigos_clasico), 1.75)
    assert _es_prefijo(codigos_clasico)
    assert sum(2 ** -len(c) for c in codigos_clasico.values()) <= 1.0 + 1e-12

    msg = [0, 1, 2, 3, 0]
    bits = codificar(msg, codigos_clasico)
    assert decodificar(bits, codigos_clasico) == msg

    from src.fuente import LOGO_FI, bloques_extendidos, cargar_imagen_bits, estimar_probabilidades

    bits_img = cargar_imagen_bits(LOGO_FI)
    for n in (2, 3):
        simbolos = bloques_extendidos(bits_img, n)
        probs = estimar_probabilidades(simbolos, n=n)
        codigos = construir_huffman(probs)
        assert _es_prefijo(codigos)
        L_teo = largo_promedio_teorico(probs, codigos, n)
        L_emp = largo_promedio_empirico(simbolos, codigos, n)
        H = entropia(probs)
        L_sym = L_teo * n
        assert L_teo < 1.0
        assert abs(L_teo - L_emp) < 1e-9
        assert H <= L_sym < H + 1
        assert decodificar(codificar(simbolos, codigos), codigos) == simbolos.tolist()

    print("huffman OK: todas las pruebas pasaron")
    probs2 = estimar_probabilidades(bloques_extendidos(bits_img, 2), n=2)
    cod2 = construir_huffman(probs2)
    L2 = largo_promedio_empirico(bloques_extendidos(bits_img, 2), cod2, 2)
    print(f"  orden 2: L={L2:.4f}, tasa={tasa_compresion(L2):.4f}, codigos={cod2}")


if __name__ == "__main__":
    _tests()
