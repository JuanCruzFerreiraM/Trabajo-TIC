from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.fuente import (
    LOGO_FI,
    bloques_extendidos,
    cargar_imagen_bits,
    estimar_probabilidades,
    guardar_imagen_bits,
    simbolos_a_bits,
)
from src.huffman import (
    codificar,
    construir_huffman,
    decodificar,
    entropia,
    largo_promedio_empirico,
    largo_promedio_teorico,
    tasa_compresion,
)
from src.canal import transmitir_codificado
from src.codec import codificar as codificar_canal, corregir, decodificar as decodificar_canal
from src.matrices import G, H, K, N

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
N_MAX_OPCIONAL = 8
ORDEN_OPCIONAL2 = 2
EBFN0_OPCIONAL2_DB = (6, 8, 10, 12)


def _simbolo_a_bits(simbolo, n):
    return format(int(simbolo), f"0{n}b")


def analizar_orden(bits, n):
    simbolos = bloques_extendidos(bits, n)
    probs = estimar_probabilidades(simbolos, n=n)
    codigos = construir_huffman(probs)
    bits_cod = codificar(simbolos, codigos)

    L_teo = largo_promedio_teorico(probs, codigos, n)
    L_emp = largo_promedio_empirico(simbolos, codigos, n)
    H = entropia(probs)
    bits_fuente = len(simbolos) * n

    assert decodificar(bits_cod, codigos) == simbolos.tolist()

    return {
        "n": n,
        "simbolos": simbolos,
        "probs": probs,
        "codigos": codigos,
        "bits_cod": bits_cod,
        "bits_fuente": bits_fuente,
        "bits_codificados": len(bits_cod),
        "L_teo": L_teo,
        "L_emp": L_emp,
        "H": H,
        "tasa": tasa_compresion(L_emp),
    }


def imprimir_resultado(res):
    n = res["n"]
    print(f"\n{'=' * 60}")
    print(f"Fuente extendida orden {n}")
    print(f"{'=' * 60}")
    print(f"{'Simbolo':>8} {'Bloque':>8} {'p_k':>10} {'Codigo':>8} {'l_k':>5}")
    print("-" * 45)
    for k in sorted(res["probs"]):
        if res["probs"][k] <= 0:
            continue
        cod = res["codigos"][k]
        print(
            f"{k:8d} {_simbolo_a_bits(k, n):>8} "
            f"{res['probs'][k]:10.6f} {cod:>8} {len(cod):5d}"
        )

    print(f"\nEntropia H(S^{n})     = {res['H']:.6f} bits/simbolo")
    print(f"H / n (por bit fuente)= {res['H'] / n:.6f} bits/bit")
    print(f"L_bar teorico         = {res['L_teo']:.6f} bits/bit fuente")
    print(f"L_bar empirico        = {res['L_emp']:.6f} bits/bit fuente")
    print(f"Tasa de compresion    = {res['tasa']:.4f}")
    print(f"Bits fuente           = {res['bits_fuente']}")
    print(f"Bits codificados      = {res['bits_codificados']}")


def graficar_largo_vs_orden(bits, n_max=N_MAX_OPCIONAL):
    ordenes = list(range(1, n_max + 1))
    L_emps = []
    L_teos = []
    H_norm = []

    for n in ordenes:
        res = analizar_orden(bits, n)
        L_emps.append(res["L_emp"])
        L_teos.append(res["L_teo"])
        H_norm.append(res["H"] / n)

    plt.figure(figsize=(8, 5))
    plt.plot(ordenes, L_emps, "o-", label="L_bar empirico")
    plt.plot(ordenes, L_teos, "s--", label="L_bar teorico")
    plt.plot(ordenes, H_norm, "^:", label=r"$H(S^n)/n$")
    plt.xlabel("Orden de la fuente extendida n")
    plt.ylabel("bits / bit de fuente")
    plt.title("Ejercicio 3 (opcional) - Largo promedio vs orden n")
    plt.xticks(ordenes)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ej3_lbar_vs_n.png", dpi=150)
    plt.close()


def _empaquetar_bits(bits_str, k=K):
    bits = np.array([int(b) for b in bits_str], dtype=int)
    n_bits = len(bits)
    pad = (-n_bits) % k
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=int)])
    return bits.reshape(-1, k), n_bits


def _bits_a_str(bits):
    return "".join(str(int(b)) for b in np.asarray(bits, dtype=int).flatten())


def transmitir_comprimido(bits_cod, EbfN0):
    """Transmite la secuencia Huffman con el codigo corrector del Ej. 1."""
    U, n_bits = _empaquetar_bits(bits_cod)
    V = codificar_canal(U, G)
    R = transmitir_codificado(V, EbfN0, N, K)
    U_hat = decodificar_canal(corregir(R, H), K)
    return _bits_a_str(U_hat.flatten()[:n_bits])


def recuperar_imagen(bits_cod_rec, codigos, n, num_bits_fuente):
    simbolos_rec = decodificar(bits_cod_rec, codigos)
    bits_rec = simbolos_a_bits(simbolos_rec, n)
    if len(bits_rec) < num_bits_fuente:
        faltan = num_bits_fuente - len(bits_rec)
        bits_rec = np.concatenate([bits_rec, np.zeros(faltan, dtype=int)])
    return bits_rec[:num_bits_fuente]


def _error_pixeles(bits_ref, bits_rec):
    bits_ref = np.asarray(bits_ref, dtype=int)
    bits_rec = np.asarray(bits_rec, dtype=int)
    n = len(bits_ref)
    if len(bits_rec) < n:
        aciertos = np.sum(bits_ref[: len(bits_rec)] == bits_rec)
        return 1.0 - aciertos / n
    return np.mean(bits_ref != bits_rec[:n])


def opcional2(bits, n=ORDEN_OPCIONAL2, ebfN0_dB_list=EBFN0_OPCIONAL2_DB):
    res = analizar_orden(bits, n)
    bits_cod = res["bits_cod"]
    bits_fuente = simbolos_a_bits(res["simbolos"], n)
    codigos = res["codigos"]

    guardar_imagen_bits(bits_fuente, OUTPUT_DIR / "ej3_original.png")

    print(f"\n{'=' * 60}")
    print(f"Opcional 2 - Transmision del mensaje comprimido (orden {n})")
    print(f"{'=' * 60}")
    print(f"Bits comprimidos: {len(bits_cod)} | Bloques fuente (K={K}): {_empaquetar_bits(bits_cod)[0].shape[0]}")

    for db in ebfN0_dB_list:
        EbfN0 = 10 ** (db / 10)
        bits_canal = transmitir_comprimido(bits_cod, EbfN0)
        tx = np.array([int(b) for b in bits_cod])
        rx = np.array([int(b) for b in bits_canal])
        err_comp = np.mean(tx != rx)

        try:
            bits_rec = recuperar_imagen(bits_canal, codigos, n, len(bits_fuente))
            err_pixel = _error_pixeles(bits_fuente, bits_rec)
            ok = True
            msg_err = ""
        except ValueError as exc:
            bits_rec = None
            err_pixel = 1.0
            ok = False
            msg_err = str(exc)

        if ok and err_pixel == 0:
            estado = "OK"
        elif ok:
            estado = "degradada"
        else:
            estado = "fallo decode"
        print(
            f"Ebf/N0 = {db:2d} dB | err. stream comprimido = {err_comp:.2e} | "
            f"err. pixeles = {err_pixel:.4f} | {estado}"
        )
        if not ok:
            print(f"  -> {msg_err}")
        else:
            guardar_imagen_bits(bits_rec, OUTPUT_DIR / f"ej3_recuperada_{db}dB.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bits = cargar_imagen_bits(LOGO_FI)
    print(f"Imagen: {LOGO_FI.name} | bits fuente = {len(bits)} | P(1) = {bits.mean():.4f}")

    resultados = []
    for n in (2, 3):
        res = analizar_orden(bits, n)
        resultados.append(res)
        imprimir_resultado(res)

    print(f"\n{'=' * 60}")
    print("Resumen obligatorio (item i)")
    print(f"{'=' * 60}")
    for res in resultados:
        print(
            f"Orden {res['n']}: L_bar = {res['L_emp']:.4f} bits/bit, "
            f"tasa = {res['tasa']:.4f}x "
            f"({res['bits_fuente']} -> {res['bits_codificados']} bits)"
        )

    # Opcional 1
    print(f"\nGenerando curva L_bar vs n (n = 1..{N_MAX_OPCIONAL})...")
    graficar_largo_vs_orden(bits)
    print(f"Gráfico guardado en {OUTPUT_DIR / 'ej3_lbar_vs_n.png'}")

    opcional2(bits)


if __name__ == "__main__":
    main()
