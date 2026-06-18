from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.canal import transmitir_codificado, transmitir_sin_codificar
from src.codec import codificar, decodificar, detectar
from src.matrices import G, H, K, N
from src.teorico import peb_sin_cod

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

LOTE_BLOQUES = 100_000
MAX_BLOQUES = 3_000_000
MIN_ERRORES_PALABRA = 30
MIN_ERRORES_BIT = 50


def _detector_lote(EbfN0, num_bloques):
    U = np.random.randint(0, 2, size=(num_bloques, K))
    V = codificar(U, G)
    R = transmitir_codificado(V, EbfN0, N, K)
    R_ok, mask = detectar(R, H)
    U_hat = decodificar(R_ok, K)
    pal_err = int((~mask).sum() + (U[mask] != U_hat).any(axis=1).sum())
    bit_err = int((~mask).sum() * K + (U[mask] != U_hat).sum())
    return pal_err, bit_err, num_bloques


def simular_detector(EbfN0):
    objetivo = MIN_ERRORES_PALABRA

    pal_err = bit_err = n_total = 0
    while n_total < MAX_BLOQUES:
        n_lote = min(LOTE_BLOQUES, MAX_BLOQUES - n_total)
        pe, be, n = _detector_lote(EbfN0, n_lote)
        pal_err += pe
        bit_err += be
        n_total += n
        if pal_err >= objetivo:
            break

    pep = pal_err / n_total
    peb = bit_err / (n_total * K)
    return pep, peb, n_total


def _sin_cod_lote(EbfN0, num_bloques):
    U = np.random.randint(0, 2, size=(num_bloques, K))
    R = transmitir_sin_codificar(U, EbfN0)
    bit_err = int(np.sum(U != R))
    return bit_err, num_bloques * K


def simular_sin_codificar(EbfN0):
    peb_teo = peb_sin_cod(EbfN0)
    objetivo = MIN_ERRORES_BIT if peb_teo >= 1e-5 else max(10, MIN_ERRORES_BIT // 4)
    max_bits = MAX_BLOQUES * K

    bit_err = n_bits = 0
    while n_bits < max_bits:
        n_lote = min(LOTE_BLOQUES, (max_bits - n_bits) // K)
        if n_lote <= 0:
            break
        be, nb = _sin_cod_lote(EbfN0, n_lote)
        bit_err += be
        n_bits += nb
        if bit_err >= objetivo:
            break

    return bit_err / n_bits if n_bits else 0.0, n_bits


def _para_semilogy(y):
    y = np.asarray(y, dtype=float)
    return np.where(y > 0, y, np.nan)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    ebfN0_dB = np.arange(0, 12, 1)
    ebfN0 = 10 ** (ebfN0_dB / 10)

    pep_sim = np.zeros_like(ebfN0, dtype=float)
    peb_sim = np.zeros_like(ebfN0, dtype=float)
    peb_sc_sim = np.zeros_like(ebfN0, dtype=float)
    peb_sc_teo = np.zeros_like(ebfN0, dtype=float)

    for i, ebf in enumerate(ebfN0):
        pep_sim[i], peb_sim[i], n_bloques = simular_detector(ebf)
        peb_sc_sim[i], n_bits = simular_sin_codificar(ebf)
        peb_sc_teo[i] = peb_sin_cod(ebf)

        print(
            f"Ebf/N0 = {ebfN0_dB[i]:2d} dB | bloques={n_bloques:7d} | "
            f"Pep det={pep_sim[i]:.2e} | Peb det={peb_sim[i]:.2e} | "
            f"Peb(sc) sim={peb_sc_sim[i]:.2e} teo={peb_sc_teo[i]:.2e} "
            f"[bits={n_bits}]"
        )

    pep_plot = _para_semilogy(pep_sim)
    peb_plot = _para_semilogy(peb_sim)
    peb_sc_plot = _para_semilogy(peb_sc_sim)

    plt.figure(figsize=(8, 5))
    plt.semilogy(ebfN0_dB, pep_plot, "o-", label="Detector (simulación)")
    plt.xlabel(r"$E_{bf}/N_0$ (dB)")
    plt.ylabel(r"$P_{ep}$")
    plt.title("Ejercicio 2 - Error de palabra (código detector)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ej2_pep.png", dpi=150)

    plt.figure(figsize=(8, 5))
    plt.semilogy(ebfN0_dB, peb_plot, "o-", label="Detector (simulación)")
    plt.semilogy(ebfN0_dB, peb_sc_plot, "s-", label="Sin codificar (simulación)")
    plt.semilogy(ebfN0_dB, peb_sc_teo, ":", label="Sin codificar (teórico)")
    plt.xlabel(r"$E_{bf}/N_0$ (dB)")
    plt.ylabel(r"$P_{eb}$")
    plt.title("Ejercicio 2 - Error de bit de fuente")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ej2_peb.png", dpi=150)

    plt.close("all")
    print(f"\nGráficos guardados en {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
