#!/usr/bin/env python3
"""
Q19 Quantization — tehnika: Amplitude Quantization (b-bit preciznost state-prep ulaza)
(čisto kvantno, bez klasičnog treninga i bez hibrida).

Koncept (kvantni analog klasične LLM-quantization FP32→INT8/INT4):
  - „Model“ je kvantno stanje |ψ_CSV⟩ = amplitude-encoding freq_vector-a CELOG CSV-a.
  - Originalne amp-ove (float64) zaokružujemo na b-bitnu preciznost
    (2^b - 1 diskretnih nivoa), pa L2-renormalizujemo:
        amp_q = round(amp · (2^b - 1)) / (2^b - 1)
        amp_q ← amp_q / ‖amp_q‖
  - Kvantizirano stanje se priprema preko StatePreparation na nq qubit-a.
  - Pokazuje direktnu zavisnost kvaliteta predikcije od preciznosti
    amplitudne reprezentacije (kompresija vs. gubitak informacije).

Kolo (nq qubit-a):
  StatePreparation(amp_q) → egzaktan Statevector → |ψ|² → bias_39 → NEXT (TOP-7).

Sve deterministički: seed=39; amp iz CELOG CSV-a, zaokruživanje deterministično.
Deterministička grid-optimizacija (nq, b) po cos(bias_39, freq_csv).

Okruženje: Python 3.11.13, qiskit 1.4.4, qiskit-machine-learning 0.8.3, macOS M1 (vidi README.md).
"""

from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

# =========================
# Seed
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass

# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4600_k31.csv")
N_NUMBERS = 7
N_MAX = 39

GRID_NQ = (5, 6)
GRID_B = (1, 2, 3, 4, 8)
B_MAIN = 8


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def freq_vector(H: np.ndarray) -> np.ndarray:
    c = np.zeros(N_MAX, dtype=np.float64)
    for v in H.ravel():
        if 1 <= v <= N_MAX:
            c[int(v) - 1] += 1.0
    return c


def amp_from_freq(f: np.ndarray, nq: int) -> np.ndarray:
    dim = 2 ** nq
    edges = np.linspace(0, N_MAX, dim + 1, dtype=int)
    amp = np.array(
        [float(f[edges[i] : edges[i + 1]].mean()) if edges[i + 1] > edges[i] else 0.0 for i in range(dim)],
        dtype=np.float64,
    )
    amp = np.maximum(amp, 0.0)
    n2 = float(np.linalg.norm(amp))
    if n2 < 1e-18:
        amp = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    else:
        amp = amp / n2
    return amp


# =========================
# b-bit amplitude quantization
# =========================
def quantize_amp(amp: np.ndarray, b: int) -> np.ndarray:
    """Zaokruživanje na 2^b − 1 diskretnih nivoa, pa L2-renorm."""
    dim = int(amp.shape[0])
    levels = (2 ** int(b)) - 1
    if levels < 1:
        return amp.copy()

    a = np.maximum(amp, 0.0).astype(np.float64)
    m = float(a.max())
    a_scaled = a / m if m > 1e-18 else a
    q = np.round(a_scaled * levels) / float(levels)
    q = np.maximum(q, 0.0)

    n2 = float(np.linalg.norm(q))
    if n2 < 1e-18:
        return np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    return q / n2


# =========================
# Quantization kolo
# =========================
def build_quantized_state(H: np.ndarray, nq: int, b: int) -> Statevector:
    amp_full = amp_from_freq(freq_vector(H), nq)
    amp_q = quantize_amp(amp_full, b)
    qc = QuantumCircuit(nq, name="q_amp")
    qc.append(StatePreparation(amp_q.tolist()), range(nq))
    return Statevector(qc)


def quantized_state_probs(H: np.ndarray, nq: int, b: int) -> np.ndarray:
    sv = build_quantized_state(H, nq, b)
    p = np.abs(sv.data) ** 2
    s = float(p.sum())
    return p / s if s > 0 else p


# =========================
# Readout
# =========================
def bias_39(probs: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    b = np.zeros(n_max, dtype=np.float64)
    for idx, p in enumerate(probs):
        b[idx % n_max] += float(p)
    s = float(b.sum())
    return b / s if s > 0 else b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pick_next_combination(probs: np.ndarray, k: int = N_NUMBERS, n_max: int = N_MAX) -> Tuple[int, ...]:
    b = bias_39(probs, n_max)
    order = np.argsort(-b, kind="stable")
    return tuple(sorted(int(o + 1) for o in order[:k]))


def l2_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# =========================
# Determ. grid-optimizacija: samo nq (b se evaluira pri B_MAIN —
# najvernija reprezentacija; degenerisano b=1 bi veštački pobedilo)
# =========================
def optimize_hparams(H: np.ndarray):
    f_csv = freq_vector(H)
    s = float(f_csv.sum())
    f_csv_n = f_csv / s if s > 0 else np.ones(N_MAX) / N_MAX
    best = None
    for nq in GRID_NQ:
        try:
            p = quantized_state_probs(H, nq, int(B_MAIN))
            bi = bias_39(p)
            score = cosine(bi, f_csv_n)
        except Exception:
            continue
        key = (score, nq)
        if best is None or key > best[0]:
            best = (key, dict(nq=nq, b=int(B_MAIN), score=float(score)))
    return best[1] if best else None


def main() -> int:
    H = load_rows(CSV_PATH)
    if H.shape[0] < 1:
        print("premalo redova")
        return 1

    print("Q19 Quantization (b-bit amplitude preciznost): CSV:", CSV_PATH)
    print("redova:", H.shape[0], "| seed:", SEED)

    best = optimize_hparams(H)
    if best is None:
        print("grid optimizacija nije uspela")
        return 2
    print(
        "BEST hparam:",
        "nq=", best["nq"],
        "| b (bit-a):", best["b"],
        "| cos(bias, freq_csv):", round(float(best["score"]), 6),
    )

    nq_best = int(best["nq"])
    f_csv = freq_vector(H)
    s = float(f_csv.sum())
    f_csv_n = f_csv / s if s > 0 else np.ones(N_MAX) / N_MAX

    amp_full = amp_from_freq(freq_vector(H), nq_best)
    print("--- demonstracija efekta kompresije (isti nq, različito b) ---")
    for b in GRID_B:
        amp_q = quantize_amp(amp_full, int(b))
        p = quantized_state_probs(H, nq_best, int(b))
        pred = pick_next_combination(p)
        cos_b = cosine(bias_39(p), f_csv_n)
        err = l2_diff(amp_q, amp_full)
        print(f"b={b:d}  |amp_q−amp_full|₂={err:.6f}  cos(bias, freq_csv)={cos_b:.6f}  NEXT={pred}")

    p_main = quantized_state_probs(H, nq_best, best["b"])
    pred_main = pick_next_combination(p_main)
    print("--- glavna predikcija ---")
    print("predikcija NEXT:", pred_main)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Q19 Quantization (b-bit amplitude preciznost): CSV: /data/loto7hh_4600_k31.csv
redova: 4600 | seed: 39
BEST hparam: nq= 5 | b (bit-a): 8 | cos(bias, freq_csv): 0.900311
--- demonstracija efekta kompresije (isti nq, različito b) ---
b=1  |amp_q-amp_full|₂=0.036003  cos(bias, freq_csv)=0.902043  NEXT=(1, 2, 3, 4, 5, 6, 7)
b=2  |amp_q-amp_full|₂=0.036003  cos(bias, freq_csv)=0.902043  NEXT=(1, 2, 3, 4, 5, 6, 7)
b=3  |amp_q-amp_full|₂=0.052896  cos(bias, freq_csv)=0.892492  NEXT=(9, 24, 27, 28, 29, 31, 32)
b=4  |amp_q-amp_full|₂=0.019033  cos(bias, freq_csv)=0.899572  NEXT=(3, 7, 19, 20, 27, 28, 29)
b=8  |amp_q-amp_full|₂=0.001242  cos(bias, freq_csv)=0.900311  NEXT=(7, 19, 22, 24, 27, 28, 31)
--- glavna predikcija ---
predikcija NEXT: (7, 19, 22, 24, 27, 28, 31)
"""



"""
Q19_Quantization_Amplitude.py — tehnika: Amplitude Quantization.

Pipeline:
  amp_full = amp_from_freq(freq_vector(CEO CSV))  (float64, L2-normalizovano)
  amp_q    = round(amp_full / max · (2^b - 1)) / (2^b - 1)  → L2-renorm
  |ψ_q⟩   = StatePreparation(amp_q)
  p        = |ψ_q|²  →  bias_39  →  TOP-7 = NEXT

Grid + demo:
  Grid bira samo nq po cos(bias, freq_csv) pri B_MAIN = 8 (najvernija
  reprezentacija). Degenerisano b=1 se namerno ne koristi kao pobednik
  jer kvazi-uniformna raspodela veštački povećava cos sa near-uniform
  freq_csv — to je artefakt, ne kvalitet.
  Demo sekcija prikazuje, za isti nq, efekat smanjenja b (1..8) na:
    · L2 razliku amp_q u odnosu na amp_full,
    · cos(bias, freq_csv),
    · NEXT predikciju.

Tehnike:
StatePreparation za amp-based enkodiranje (kao Q8/Q15/Q17).
Determinističko zaokruživanje amp-a na 2^b - 1 nivoa (bez klasičnog optimizera).
Egzaktan Statevector (bez uzorkovanja).
Deterministička grid-optimizacija (nq, b).

Prednosti:
Najdirektniji kvantni analog klasične LLM-kompresije (INT4/INT8).
Demonstrira preciznost-vs-kvalitet trade-off reprezentacije stanja.
Čisto kvantno: bez treninga, softmax-a, hibrida.

Nedostaci:
Za nisko b (npr. 1 ili 2) veliki deo amp-a se svede na istu (retku) vrednost,
što može dati izrazitu degeneraciju i „stable sort“ artefakte (prvih K indeksa
iste vrednosti).
mod-39 readout meša stanja (dim 2^nq ≠ 39).
„Quantization“ u kvantnom programiranju ima i drugi smisao (dubina/qubit redukcija);
ovde je fokus striktno na preciznost amplitude enkodera.
"""
