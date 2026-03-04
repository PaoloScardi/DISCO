# cfactor/elasticity.py
from __future__ import annotations

from typing import Tuple, List, Optional, Dict, Any
import math
import numpy as np

from .models import ContrastFactorInput
from .geometry import build_slip_frame, _direct_basis_from_cell


DEBUG_DEGEN_LIMIT = False          # set False to silence
DEBUG_DEGEN_LIMIT_DUMP_NPZ = False
DEBUG_DEGEN_LIMIT_PREFIX = "degen_limit_phi"

# in cfactor/elasticity.py, near the top
_EM_CACHE_VERSION = 5  # bump whenever Em logic changes

DEBUG_DUMP_FEM_NPZ = False
DEBUG_FEM_NPZ_PREFIX = "fem_phi"


# ---------------------------------------------------------------------------
# Small per-input cache (attached to ContrastFactorInput instance)
# ---------------------------------------------------------------------------
def _get_cf_cache(cf_input: ContrastFactorInput) -> dict:
    cache = getattr(cf_input, "_cfactor_cache", None)
    if cache is None:
        cache = {}
        setattr(cf_input, "_cfactor_cache", cache)
    return cache


def dump_fem_inputs(path: str, fem_dbg: dict, Nburger: float) -> None:
    np.savez(
        path,
        A=fem_dbg["A"],
        D=fem_dbg["D"],
        R=fem_dbg["R"],
        Nburger=np.array([float(Nburger)]),
        pref=np.array([float(fem_dbg.get("pref", np.nan))]),
        Ab=fem_dbg["Ab"],
        Aa=fem_dbg["Aa"],
        Db=fem_dbg["Db"],
        Da=fem_dbg["Da"],
        ReR=fem_dbg["ReR"],
        ImR=fem_dbg["ImR"],
        AbsR=fem_dbg["AbsR"],
        DC2=fem_dbg["DC2"],
        Em=fem_dbg["Em"],
    )


# ------------------------------------------------------------
# Voigt order (Mathematica style): {11,22,33,23,13,12}
# ------------------------------------------------------------
_VOIGT_PAIRS_MATH: List[tuple[int, int]] = [
    (0, 0),  # 11
    (1, 1),  # 22
    (2, 2),  # 33
    (1, 2),  # 23
    (0, 2),  # 13
    (0, 1),  # 12
]


def _voigt_pairs() -> List[tuple[int, int]]:
    return _VOIGT_PAIRS_MATH


def _chop(A: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    A = np.asarray(A, dtype=float).copy()
    A[np.abs(A) < tol] = 0.0
    # kill -0.0
    A[A == 0.0] = 0.0
    return A


def _K_quadrants_from_P_rows(P_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reproduce Mathematica package quadrants exactly (0-based with %3).
    """
    t = np.asarray(P_rows, dtype=float).reshape(3, 3)

    K1 = np.zeros((3, 3), dtype=float)
    K2 = np.zeros((3, 3), dtype=float)
    K3 = np.zeros((3, 3), dtype=float)
    K4 = np.zeros((3, 3), dtype=float)

    for i in range(3):
        for j in range(3):
            K1[i, j] = t[i, j] ** 2

            j1 = (j + 1) % 3
            j2 = (j + 2) % 3
            K2[i, j] = t[i, j1] * t[i, j2]

            i1 = (i + 1) % 3
            i2 = (i + 2) % 3
            K3[i, j] = t[i1, j] * t[i2, j]

            K4[i, j] = (
                t[i1, j1] * t[i2, j2]
                + t[i1, j2] * t[i2, j1]
            )

    return K1, K2, K3, K4


def _Kmatrix_from_P_rows(P_rows: np.ndarray) -> dict[str, np.ndarray]:
    """
    Reproduce Mathematica Kmatrix:
      t = {K1, 2*K2, K3, K4};
      Join[
        Table[Join[t[[1,i]], t[[2,i]]], {i,3}],
        Table[Join[t[[3,i]], t[[4,i]]], {i,3}]
      ]
    """
    K1, K2, K3, K4 = _K_quadrants_from_P_rows(P_rows)
    top = np.hstack([K1, 2.0 * K2])
    bot = np.hstack([K3, K4])
    K = np.vstack([top, bot])
    return {"K1": K1, "K2": K2, "K3": K3, "K4": K4, "K": _chop(K)}


# ------------------------------------------------------------
# Voigt conversion (NO Kelvin scaling)
# ------------------------------------------------------------
def _voigt_to_c4(Cv: np.ndarray) -> np.ndarray:
    """
    6x6 Voigt -> C_ijkl filling minor symmetries.
    Voigt order matches Mathematica.
    """
    pairs = _voigt_pairs()
    C4 = np.zeros((3, 3, 3, 3), dtype=float)
    for a, (i, j) in enumerate(pairs):
        for b, (k, l) in enumerate(pairs):
            v = float(Cv[a, b])
            C4[i, j, k, l] = v
            C4[j, i, k, l] = v
            C4[i, j, l, k] = v
            C4[j, i, l, k] = v
    return C4


def _c4_to_voigt(C4: np.ndarray) -> np.ndarray:
    """
    C_ijkl -> 6x6 Voigt using average over minor symmetries.
    Voigt order matches Mathematica.
    """
    pairs = _voigt_pairs()
    Cv = np.zeros((6, 6), dtype=float)
    for a, (i, j) in enumerate(pairs):
        for b, (k, l) in enumerate(pairs):
            v1 = float(C4[i, j, k, l])
            v2 = float(C4[j, i, k, l])
            v3 = float(C4[i, j, l, k])
            v4 = float(C4[j, i, l, k])
            Cv[a, b] = 0.25 * (v1 + v2 + v3 + v4)
    return Cv


def _rotate_c4(C: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Rotate 4th-order tensor:
      C'_{abcd} = P_{ia} P_{jb} P_{kc} P_{ld} C_{ijkl}
    P: 3x3, columns = {e1,e2,e3}.
    """
    Cprime = np.zeros_like(C)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    s = 0.0
                    for i in range(3):
                        Pia = P[i, a]
                        for j in range(3):
                            Pjb = P[j, b]
                            for k in range(3):
                                Pkc = P[k, c]
                                for l in range(3):
                                    Pld = P[l, d]
                                    s += Pia * Pjb * Pkc * Pld * C[i, j, k, l]
                    Cprime[a, b, c, d] = s
    return Cprime


# ------------------------------------------------------------
# Cubic C4 helper
# ------------------------------------------------------------
def _stiffness_cubic_4th(C11: float, C12: float, C44: float) -> np.ndarray:
    C = np.zeros((3, 3, 3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    if i == j == k == l:
                        C[i, j, k, l] = C11
                    elif i == j and k == l and i != k:
                        C[i, j, k, l] = C12
                    elif (i == k and j == l and i != j) or (j == k and i == l and i != j):
                        C[i, j, k, l] = C44
    return C


# ------------------------------------------------------------
# Stroh Q,R,T,N (Mathematica convention)
# ------------------------------------------------------------
def _stroh_denominators(A: np.ndarray, L: np.ndarray) -> np.ndarray:
    A = np.asarray(A, complex).reshape(3, 3)
    L = np.asarray(L, complex).reshape(3, 3)
    den = np.zeros(3, dtype=complex)
    for a in range(3):
        den[a] = np.dot(A[:, a], L[:, a])  # bilinear, no conjugation
    return den


def _is_bad_stroh_basis(A: np.ndarray, L: np.ndarray, tol: float = 1e-8) -> bool:
    den = _stroh_denominators(A, L)
    if np.any(np.abs(den) < tol):
        return True
    G = (np.asarray(A, complex).T @ np.asarray(L, complex))
    svals = np.linalg.svd(G, compute_uv=False)
    if np.min(svals) < tol * np.max(svals):
        return True
    return False


def _regularization_scale(CinSS: np.ndarray) -> float:
    return float(np.max(np.abs(CinSS))) if np.max(np.abs(CinSS)) > 0 else 1.0


def _symmetric_regularizer(kind: int = 0) -> np.ndarray:
    """
    Deterministic symmetric perturbations.

    kind=0: dense symmetric (recommended)
    kind=1: sparse symmetric (backup)
    kind=2: identity (last-resort)
    """
    if kind == 2:
        return np.eye(6, dtype=float)

    if kind == 1:
        S = np.zeros((6, 6), dtype=float)
        pairs = [(0, 1), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5), (1, 5), (2, 4)]
        for t, (i, j) in enumerate(pairs, start=1):
            val = 0.1 * t
            S[i, j] = val
            S[j, i] = val
        S += 0.5 * np.eye(6)
        S /= np.linalg.norm(S)
        return S

    S = np.zeros((6, 6), dtype=float)
    for i in range(6):
        for j in range(i, 6):
            v = (i + 1) + 0.37 * (j + 1) + 0.11 * (i + 1) * (j + 1)
            S[i, j] = v
            S[j, i] = v
    S /= np.linalg.norm(S)
    return S


def _QRT_from_Cvoigt(Cv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = Cv
    Q = np.array(
        [[t[0, 0], t[0, 5], t[0, 4]],
         [t[0, 5], t[5, 5], t[4, 5]],
         [t[0, 4], t[4, 5], t[4, 4]]],
        dtype=float,
    )
    R = np.array(
        [[t[0, 5], t[0, 1], t[0, 3]],
         [t[5, 5], t[1, 5], t[3, 5]],
         [t[4, 5], t[1, 4], t[3, 4]]],
        dtype=float,
    )
    T = np.array(
        [[t[5, 5], t[1, 5], t[3, 5]],
         [t[1, 5], t[1, 1], t[1, 3]],
         [t[3, 5], t[1, 3], t[3, 3]]],
        dtype=float,
    )
    return Q, R, T


def _stroh_N(Q: np.ndarray, R: np.ndarray, T: np.ndarray) -> np.ndarray:
    Ti = np.linalg.inv(T)
    A = -Ti @ R.T
    B = -Ti
    Cb = Q - R @ Ti @ R.T
    D = -R @ Ti
    return np.vstack((np.hstack((A, B)), np.hstack((Cb, D))))


def _select_three_im_positive_roots(w: np.ndarray) -> list[int]:
    """
    Deterministic selection:
      - sort by imag descending, then real ascending as tie-break
      - pick first 3 with imag>0
      - fallback: take first 3 of the sorted list
    """
    idx = list(range(len(w)))
    idx.sort(key=lambda i: (-float(w[i].imag), float(w[i].real)))
    idx_pos = [i for i in idx if w[i].imag > 0][:3]
    if len(idx_pos) != 3:
        idx_pos = idx[:3]
    return idx_pos


# ------------------------------------------------------------
# API: CinSS (stiffness in slip-system frame)
# ------------------------------------------------------------
def stiffness_in_slip_system(
    cf_input: ContrastFactorInput,
    verbose: bool = False,
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Match Mathematica Crystallography`Elastic` CinSS (Ting method).
    """
    C = np.asarray(cf_input.elastic.as_voigt_6x6(), dtype=float)

    e1, e2, e3 = build_slip_frame(cf_input)
    P_rows = np.vstack([e1, e2, e3])  # rows = e1,e2,e3

    Kparts = _Kmatrix_from_P_rows(P_rows)
    K = Kparts["K"]

    raw = (K @ C) @ K.T
    CinSS = _chop(raw)

    if not return_debug:
        return CinSS

    dbg = {
        "P_rows_cart": _chop(P_rows),
        "K1": _chop(Kparts["K1"]),
        "K2": _chop(Kparts["K2"]),
        "K3": _chop(Kparts["K3"]),
        "K4": _chop(Kparts["K4"]),
        "K": _chop(K),
        "C": _chop(C),
        "raw": _chop(raw),
        "CinSS": _chop(CinSS),
    }
    return CinSS, dbg


# ------------------------------------------------------------
# Canonicalization / snapping
# ------------------------------------------------------------
def canonicalize_complex(z: complex, tol: float = 1e-6) -> complex:
    zr = 0.0 if abs(z.real) < tol else float(z.real)
    zi = 0.0 if abs(z.imag) < tol else float(z.imag)
    return complex(zr, zi)


def canonicalize_roots(p: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=complex)
    return np.array([canonicalize_complex(pi, tol=tol) for pi in p], dtype=complex)


def _round_half_up(x: float) -> float:
    if x >= 0:
        return float(math.floor(x + 0.5))
    return float(math.ceil(x - 0.5))


def quantize_complex(z: complex, tol: float = 1e-6) -> complex:
    xr = z.real / tol
    xi = z.imag / tol
    zr = 0.0 if abs(z.real) < tol else _round_half_up(xr) * tol
    zi = 0.0 if abs(z.imag) < tol else _round_half_up(xi) * tol
    if zr == 0.0:
        zr = 0.0
    if zi == 0.0:
        zi = 0.0
    return complex(zr, zi)


def snap_roots(p: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=complex)
    return np.array([quantize_complex(pi, tol=tol) for pi in p], dtype=complex)


def is_degenerate_after_chop(p: np.ndarray, tol: float = 1e-6) -> bool:
    pq = [quantize_complex(pi, tol=tol) for pi in np.asarray(p, dtype=complex)]
    return (pq[0] == pq[1]) or (pq[0] == pq[2]) or (pq[1] == pq[2])


def is_near_degenerate(p: np.ndarray, tol: float = 1e-6) -> bool:
    p = np.asarray(p, dtype=complex).ravel()
    if p.size != 3:
        return False
    d01 = abs(p[0] - p[1])
    d02 = abs(p[0] - p[2])
    d12 = abs(p[1] - p[2])
    return min(d01, d02, d12) <= tol


# ------------------------------------------------------------
# Deterministic column normalization helpers
# ------------------------------------------------------------
def columnwise_bilinear_normalize(A: np.ndarray, L: np.ndarray, tol: float = 1e-14):
    A = np.asarray(A, complex).copy()
    L = np.asarray(L, complex).copy()
    for j in range(3):
        den = np.dot(A[:, j], L[:, j])  # bilinear, no conjugation
        if abs(den) < tol:
            continue
        L[:, j] /= den
    return A, L


def normalize_stroh_columns(
    A: np.ndarray,
    L: np.ndarray,
    eps: float = 1e-30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministic phase/sign fixing with tie-stable pivot selection.
    """
    A = np.asarray(A, dtype=complex).copy()
    L = np.asarray(L, dtype=complex).copy()

    for j in range(A.shape[1]):
        col = A[:, j]

        abscol = np.abs(col)
        m = float(np.max(abscol))
        if m < eps:
            continue

        # tie-stable: choose smallest index among near-max
        tol = 1e-12
        cand = np.where(abscol >= (1.0 - tol) * m)[0]
        k = int(cand[0])

        pivot = col[k]
        if abs(pivot) < eps:
            continue

        phase = pivot / abs(pivot)
        A[:, j] /= phase
        L[:, j] /= phase

        if A[k, j].real < 0:
            A[:, j] *= -1
            L[:, j] *= -1

        A.real[np.abs(A.real) < 1e-14] = 0.0
        A.imag[np.abs(A.imag) < 1e-14] = 0.0
        L.real[np.abs(L.real) < 1e-14] = 0.0
        L.imag[np.abs(L.imag) < 1e-14] = 0.0

    return A, L


def rescale_stroh_columns(
    A: np.ndarray,
    L: np.ndarray,
    target: float = 1.0,
    eps: float = 1e-30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep column magnitudes bounded WITHOUT changing A^T L.
    """
    A = np.asarray(A, dtype=complex).copy()
    L = np.asarray(L, dtype=complex).copy()

    for j in range(A.shape[1]):
        m = float(np.max(np.abs(A[:, j])))
        if m < eps:
            continue
        s = m / target
        A[:, j] /= s
        L[:, j] *= s

    return A, L


# ------------------------------------------------------------
# Reduced compliance SinSS (Mathematica)
# ------------------------------------------------------------
def _sinss_from_cinss(CinSS: np.ndarray) -> np.ndarray:
    t = np.linalg.inv(np.asarray(CinSS, dtype=float))
    s = np.zeros((6, 6), dtype=float)
    t33 = t[2, 2]
    for M in range(6):
        for N in range(6):
            s[M, N] = t[M, N] - (t[M, 2] * t[N, 2]) / t33
    return s


# ------------------------------------------------------------
# Root sorting to match Mathematica
# ------------------------------------------------------------
def _L2_L3_L4(p: complex, s: np.ndarray) -> tuple[complex, complex, complex]:
    s11, s12, s22 = s[0, 0], s[0, 1], s[1, 1]
    s14, s15, s16 = s[0, 3], s[0, 4], s[0, 5]
    s24, s25, s26 = s[1, 3], s[1, 4], s[1, 5]
    s44, s45, s55 = s[3, 3], s[3, 4], s[4, 4]
    s46, s56, s66 = s[3, 5], s[4, 5], s[5, 5]

    L2 = s55 * p**2 - 2.0 * s45 * p + s44
    L3 = s15 * p**3 - (s14 + s56) * p**2 + (s25 + s46) * p - s24
    L4 = s11 * p**4 - 2.0 * s16 * p**3 + (2.0 * s12 + s66) * p**2 - 2.0 * s26 * p + s22
    return L2, L3, L4


def _p_special_from_sinss(s: np.ndarray) -> complex:
    s44 = float(s[3, 3])
    s45 = float(s[3, 4])
    s55 = float(s[4, 4])

    disc = s44 * s55 - s45 * s45
    if disc < 0.0 and abs(disc) < 1e-14:
        disc = 0.0
    disc = max(disc, 0.0)

    return complex(s45 / s55, math.sqrt(disc) / s55)


def sort_roots_like_mathematica(
    p: np.ndarray,
    A: np.ndarray,
    L: np.ndarray,
    s: np.ndarray,
    tol_match: float = 1e-8,
    tol_L4: float = 1e-8,
    tol_pair_im: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Emulates Elastic.m SortRoots:
      1) move p_special to the end (stable)
      2) move L4==0 roots to the front (stable)
      3) if last two have same imag, order by Re ascending
    """
    p = np.asarray(p, dtype=complex)
    A = np.asarray(A, dtype=complex)
    L = np.asarray(L, dtype=complex)

    p_special = _p_special_from_sinss(s)

    def close(a: complex, b: complex) -> bool:
        return abs(a - b) <= tol_match

    idx = list(range(3))
    idx_not = [i for i in idx if not close(p[i], p_special)]
    idx_yes = [i for i in idx if close(p[i], p_special)]
    idx = idx_not + idx_yes

    def is_L4_zero(i: int) -> bool:
        _, _, L4 = _L2_L3_L4(p[i], s)
        return abs(L4) <= tol_L4

    idx_L4 = [i for i in idx if is_L4_zero(i)]
    idx_nonL4 = [i for i in idx if not is_L4_zero(i)]
    idx = idx_L4 + idx_nonL4

    if len(idx) == 3:
        i1, i2 = idx[1], idx[2]
        if abs(p[i1].imag - p[i2].imag) <= tol_pair_im:
            if p[i1].real > p[i2].real:
                idx[1], idx[2] = idx[2], idx[1]

    return p[idx], A[:, idx], L[:, idx]


# ------------------------------------------------------------
# Mathematica LL/AA building blocks
# ------------------------------------------------------------
def _complex_derivative(f, p0: complex, h: float = 1e-8) -> complex:
    hp = h * (1.0 + abs(p0))
    return (f(p0 + hp) - f(p0 - hp)) / (2.0 * hp)


def _ratio_limit_like_mathematica(
    num_f,
    den_f,
    p0: complex,
    *,
    tol: float = 1e-10,
    h: float = 1e-8,
) -> complex:
    num0 = num_f(p0)
    den0 = den_f(p0)

    if abs(den0) > tol:
        return num0 / den0

    if abs(num0) <= tol and abs(den0) <= tol:
        dnum = _complex_derivative(num_f, p0, h=h)
        dden = _complex_derivative(den_f, p0, h=h)

        if abs(dden) > tol:
            return dnum / dden

        d2num = _complex_derivative(lambda p: _complex_derivative(num_f, p, h=h), p0, h=h)
        d2den = _complex_derivative(lambda p: _complex_derivative(den_f, p, h=h), p0, h=h)
        if abs(d2den) > tol:
            return d2num / d2den

        return 0.0 + 0.0j

    hp = h * (1.0 + abs(p0))
    den1 = den_f(p0 + hp)
    if abs(den1) > tol:
        return num_f(p0 + hp) / den1

    return 0.0 + 0.0j


def _lambdas_from_roots(p: np.ndarray, CinSS: np.ndarray) -> np.ndarray:
    s = _sinss_from_cinss(CinSS)
    lam = np.zeros(3, dtype=complex)

    def L2_of(pp): return _L2_L3_L4(pp, s)[0]
    def L3_of(pp): return _L2_L3_L4(pp, s)[1]
    def L4_of(pp): return _L2_L3_L4(pp, s)[2]

    lam[0] = -_ratio_limit_like_mathematica(L3_of, L2_of, p[0], tol=1e-10, h=1e-8)
    lam[1] = -_ratio_limit_like_mathematica(L3_of, L2_of, p[1], tol=1e-10, h=1e-8)
    lam[2] = -_ratio_limit_like_mathematica(L3_of, L4_of, p[2], tol=1e-10, h=1e-8)
    return lam


def _LL_matrix(p: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    p1, p2, p3 = p
    l1, l2, l3 = lambdas
    return np.array(
        [[-p1, -p2, -p3 * l3],
         [1.0 + 0j, 1.0 + 0j, l3],
         [-l1, -l2, -1.0 + 0j]],
        dtype=complex,
    )


def _KKmatrix(ind: int, p: np.ndarray, CinSS: np.ndarray) -> np.ndarray:
    s = _sinss_from_cinss(CinSS)
    pi = p[ind]
    row1 = np.array([s[0, 5] - pi * s[0, 0], s[0, 1], s[0, 3] - pi * s[0, 4]], dtype=complex)
    row2 = np.array([(s[1, 5] - pi * s[1, 0]) / pi, s[1, 1] / pi, (s[1, 3] - pi * s[1, 4]) / pi], dtype=complex)
    row3 = np.array([s[4, 5] - pi * s[4, 0], s[4, 1], s[4, 3] - pi * s[4, 4]], dtype=complex)
    return np.vstack([row1, row2, row3])


def _A_matrix_from_KK_LL(p: np.ndarray, CinSS: np.ndarray) -> np.ndarray:
    lambdas = _lambdas_from_roots(p, CinSS)
    LL = _LL_matrix(p, lambdas)
    cols = []
    for i in range(3):
        Ki = _KKmatrix(i, p, CinSS)
        cols.append(Ki @ LL[:, i])
    return np.column_stack(cols)


# ------------------------------------------------------------
# D and FEM pieces
# ------------------------------------------------------------
def D_from_stroh_bilinear(
    A: np.ndarray,
    L: np.ndarray,
    phi_deg: float,
    Nburger: float,
) -> np.ndarray:
    A = np.asarray(A, dtype=complex).reshape(3, 3)
    L = np.asarray(L, dtype=complex).reshape(3, 3)

    phi = math.radians(float(phi_deg))
    bu = np.array(
        [float(Nburger) * math.sin(phi), 0.0, float(Nburger) * math.cos(phi)],
        dtype=complex,
    )

    D = np.zeros(3, dtype=complex)
    for a in range(3):
        La = L[:, a]
        Aa = A[:, a]
        num = -np.dot(La, bu)          # NO conjugation
        den = 2.0 * np.dot(Aa, La)     # NO conjugation
        D[a] = 0.0 if abs(den) < 1e-20 else (num / den)

    return D


def _safe_div_tan(num: float, den: float, eps: float) -> float:
    den = den if abs(den) > eps else (eps if den >= 0.0 else -eps)
    return math.atan2(num, den)


def _FEm_from_ADR(
    A: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    Nburger: float,
    verbose: bool = False,
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, Dict[str, Any]]:
    A = np.asarray(A, dtype=complex).reshape(3, 3)
    D = np.asarray(D, dtype=complex).reshape(3)
    R = np.asarray(R, dtype=complex).reshape(3)

    Ab = np.abs(A)
    Aa = np.angle(A)
    Db = np.abs(D)
    Da = np.angle(D)

    ReR = R.real
    ImR = R.imag
    AbsR = np.abs(R)

    eps = 1e-14

    def AbC(a_idx: int, ell: int) -> float:
        return 1.0 if ell == 1 else float(AbsR[a_idx])

    def AgC(a_idx: int, ell: int) -> float:
        return 0.0 if ell == 1 else float(np.angle(R[a_idx]))

    def G(k_idx: int, ell: int, a_idx: int) -> float:
        return float(Aa[k_idx, a_idx] + Da[a_idx] + AgC(a_idx, ell))

    def X(a_idx: int) -> float:
        return _safe_div_tan(float(ReR[a_idx]), float(ImR[a_idx]), eps)

    def Y(a1: int, a2: int) -> float:
        num = float(ReR[a1] - ReR[a2])
        den = float(ImR[a1] + ImR[a2])
        return _safe_div_tan(num, den, eps)

    def Z(a1: int, a2: int) -> float:
        X1 = X(a1)
        X2 = X(a2)
        num = float(ImR[a1] * ImR[a2] * (math.tan(X1) + math.tan(X2)))
        den = float(ImR[a1] * ImR[a2] - ReR[a1] * ReR[a2] + AbsR[a1] ** 2)
        return _safe_div_tan(num, den, eps)

    DC2 = np.zeros((3, 3), dtype=float)
    for a1 in range(3):
        for a2 in range(3):
            if a1 == a2:
                den_im = float(ImR[a1])
                den_im = den_im if abs(den_im) > eps else (eps if den_im >= 0.0 else -eps)
                DC2[a1, a2] = float(AbsR[a1] / (2.0 * den_im * den_im))
            else:
                num = float((ReR[a1] - ReR[a2]) ** 2 + (ImR[a1] - ImR[a2]) ** 2)
                denom = float(
                    (AbsR[a1] ** 2 - AbsR[a2] ** 2) ** 2
                    + 4.0
                    * (ReR[a1] - ReR[a2])
                    * (AbsR[a2] ** 2 * ReR[a1] - AbsR[a1] ** 2 * ReR[a2])
                )
                denom = denom if abs(denom) > eps else (eps if denom >= 0.0 else -eps)
                den_im = float(ImR[a1])
                den_im = den_im if abs(den_im) > eps else (eps if den_im >= 0.0 else -eps)
                DC2[a1, a2] = float(AbsR[a1] / den_im * math.sqrt(max(num / denom, 0.0)))

    pref = 0.0 if Nburger == 0.0 else (8.0 / (Nburger ** 2))

    E = np.zeros((3, 2, 3, 2), dtype=float)
    for i in range(3):
        for j in range(2):
            for k in range(3):
                for l in range(2):
                    s_val = 0.0
                    for a1 in range(3):
                        for a2 in range(3):
                            dc1 = (
                                Ab[k, a1]
                                * Ab[i, a2]
                                * Db[a1]
                                * Db[a2]
                                * AbC(a1, l + 1)
                                * AbC(a2, j + 1)
                            )
                            term = (
                                math.cos(G(k, l + 1, a1) + X(a1))
                                * math.cos(G(i, j + 1, a2) - Y(a1, a2))
                                + math.sin(G(k, l + 1, a1))
                                * math.sin(G(i, j + 1, a2) + Z(a1, a2))
                            )
                            s_val += dc1 * DC2[a1, a2] * term
                    E[i, j, k, l] = pref * s_val

    E[np.abs(E) < 1e-3] = 0.0

    Em = np.zeros((6, 6), dtype=float)
    for j in range(2):
        for i in range(3):
            row = 3 * j + i
            for l in range(2):
                for k in range(3):
                    col = 3 * l + k
                    Em[row, col] = E[i, j, k, l]

    if not return_debug:
        return Em

    dbg = {
        "A": A, "D": D, "R": R,
        "Ab": Ab, "Aa": Aa,
        "Db": Db, "Da": Da,
        "ReR": ReR, "ImR": ImR, "AbsR": AbsR,
        "DC2": DC2,
        "pref": pref,
        "Em": Em,
    }
    return Em, dbg


# ------------------------------------------------------------
# Brute-force Eq.(12) integration (used only for degenerate cases)
# ------------------------------------------------------------
def _Em_from_eq12_bruteforce(
    p: np.ndarray,
    A: np.ndarray,
    D: np.ndarray,
    Nburger: float,
    *,
    n_phi: int = 8192,
    denom_eps: float = 1e-14,
) -> np.ndarray:
    """Compute Em by numerically integrating Eq.(12) using Eq.(16).

    This is intended as a deterministic fallback when Stroh roots are
    *degenerate* (repeated or near-repeated), where the closed-form
    expressions are prone to branch/limit issues.

    Notes:
      - Uses a fixed midpoint grid for phi in [0, 2π) to be deterministic.
      - Returns the same 6x6 layout as _FEm_from_ADR:
            row = 3*j + i, col = 3*l + k  with i,k=0..2 and j,l=0..1.
      - Applies the same overall prefactor 8/|b|^2 as _FEm_from_ADR.
    """
    p = np.asarray(p, dtype=np.complex128).reshape(3)
    A = np.asarray(A, dtype=np.complex128).reshape(3, 3)
    D = np.asarray(D, dtype=np.complex128).reshape(3)

    if n_phi is None or int(n_phi) <= 0:
        n_phi = 8192
    n_phi = int(n_phi)

    # Deterministic midpoint rule grid
    k = np.arange(n_phi, dtype=np.float64)
    phi = (2.0 * np.pi) * (k + 0.5) / n_phi
    c = np.cos(phi)
    s = np.sin(phi)

    # denom[a, k] = cos(phi) + p[a] sin(phi)
    denom = c[None, :] + p[:, None] * s[None, :]
    # Avoid extremely tiny denominators (should not happen for Im(p)>0, but be safe)
    tiny = np.abs(denom) < denom_eps
    if np.any(tiny):
        denom = denom.copy()
        denom[tiny] = denom[tiny] + denom_eps

    # beta[i,j,k] where i=0..2 and j=0..1 (j corresponds to n=1,2 in the paper)
    beta = np.zeros((3, 2, n_phi), dtype=np.float64)

    # j=0 -> p^(0)=1 ; j=1 -> p^(1)=p
    p_pow = np.vstack([np.ones_like(p), p])  # shape (2,3)
    for j in (0, 1):
        num = (A * D[None, :]) * p_pow[j][None, :]  # (3,3)
        val = (num[:, :, None] / denom[None, :, :]).sum(axis=1)  # (3,n_phi)
        beta[:, j, :] = np.imag(val)

    # Eq.(12): E[i,j,k,l] = (1/pi) ∫ beta[i,j] * beta[k,l] dphi
    # Use simple midpoint weights (uniform)
    dphi = (2.0 * np.pi) / n_phi
    E = np.zeros((3, 2, 3, 2), dtype=np.float64)
    for i in range(3):
        for j in range(2):
            bij = beta[i, j, :]
            for k_ in range(3):
                for l in range(2):
                    E[i, j, k_, l] = (1.0 / np.pi) * float(np.sum(bij * beta[k_, l, :]) * dphi)

    pref = 0.0 if Nburger == 0.0 else (8.0 / (Nburger ** 2))
    E *= pref
    E[np.abs(E) < 1e-3] = 0.0

    Em = np.zeros((6, 6), dtype=np.float64)
    for j in range(2):
        for i in range(3):
            row = 3 * j + i
            for l in range(2):
                for k_ in range(3):
                    col = 3 * l + k_
                    Em[row, col] = E[i, j, k_, l]
    return Em


# ------------------------------------------------------------
# Degenerate-limit Em
# ------------------------------------------------------------
def _Em_degenerate_limit(
    CinSS: np.ndarray,
    s: np.ndarray,
    phi_deg: float,
    Nburger: float,
    *,
    eps_list=(1e-9, 3e-10, 1e-10, 3e-11, 1e-11),
) -> np.ndarray:
    CinSS = np.asarray(CinSS, float)
    scale = _regularization_scale(CinSS)

    def Em_from_C_using_LLAA(Ceps: np.ndarray, seps: np.ndarray, snap_tol: float) -> tuple[np.ndarray, dict]:
        Q, R, T = _QRT_from_Cvoigt(Ceps)
        detT = float(np.linalg.det(T))
        if abs(detT) < 1e-14:
            T = T + (1e-12 * np.eye(3))
        N = _stroh_N(Q, R, T)
        w, _v = np.linalg.eig(N)

        idx_pos = _select_three_im_positive_roots(w)

        p_raw = w[idx_pos]
        p_clean = canonicalize_roots(p_raw, tol=1e-12)
        p_snap = snap_roots(p_clean, tol=snap_tol)

        lambdas = _lambdas_from_roots(p_snap, Ceps)
        LL = _LL_matrix(p_snap, lambdas)
        AA = _A_matrix_from_KK_LL(p_snap, Ceps)

        # deterministic Mathematica-like ordering
        p_snap, AA, LL = sort_roots_like_mathematica(p_snap, AA, LL, seps)

        # deterministic phase/sign + safe normalization
        AA, LL = normalize_stroh_columns(AA, LL)
        AA, LL = columnwise_bilinear_normalize(AA, LL, tol=1e-14)
        AA, LL = normalize_stroh_columns(AA, LL)
        AA, LL = rescale_stroh_columns(AA, LL, target=1.0)

        p_use = p_snap
        D = D_from_stroh_bilinear(AA, LL, phi_deg, Nburger)
        Em = _FEm_from_ADR(AA, D, p_use, Nburger, verbose=False)

        dbg_local = {
            "detT": detT,
            "idx_pos": np.array(idx_pos, dtype=int),
            "p_raw": p_raw,
            "p_clean": p_clean,
            "p_snap": p_snap,
            "A": AA,
            "L": LL,
            "ATL": (AA.T @ LL),
            "D": D,
            "Em": Em,
        }
        return np.asarray(Em, float), dbg_local

    def try_one_regularizer(S: np.ndarray, kind: int) -> Optional[np.ndarray]:
        Ems: list[np.ndarray] = []
        xs: list[float] = []

        for eps in eps_list:
            Ceps = CinSS + (eps * scale) * S
            seps = _sinss_from_cinss(Ceps)

            Em, dbg_local = Em_from_C_using_LLAA(Ceps, seps, snap_tol=1e-12)

            A = dbg_local["A"]
            L = dbg_local["L"]
            if _is_bad_stroh_basis(A, L, tol=1e-8):
                if DEBUG_DEGEN_LIMIT:
                    print(f"[degen_limit] kind={kind} eps={eps:.1e}: bad basis (skip)")
                continue

            x = float(eps * scale)
            Ems.append(Em)
            xs.append(x)

            if DEBUG_DEGEN_LIMIT:
                print(
                    f"[degen_limit] kind={kind} eps={eps:.1e} x={x:.3e} | "
                    f"sha(Ceps)={_sha256_arr(np.asarray(Ceps,float))[:12]} "
                    f"sha(seps)={_sha256_arr(np.asarray(seps,float))[:12]} "
                    f"p_clean={_fmt_cplx_vec(dbg_local['p_clean'])} "
                    f"p_snap={_fmt_cplx_vec(dbg_local['p_snap'])} "
                    f"sha(A)={_sha256_arr(np.asarray(dbg_local['A']))[:12]} "
                    f"sha(L)={_sha256_arr(np.asarray(dbg_local['L']))[:12]} "
                    f"sha(D)={_sha256_arr(np.asarray(dbg_local['D']))[:12]} "
                    f"sha(Em)={_sha256_arr(np.asarray(dbg_local['Em']))[:12]}"
                )

            if DEBUG_DEGEN_LIMIT_DUMP_NPZ:
                phi = float(phi_deg)
                npz_path = f"{DEBUG_DEGEN_LIMIT_PREFIX}{phi:.0f}_k{kind}_eps{eps:.1e}.npz".replace("+", "")
                np.savez(
                    npz_path,
                    Ceps=np.asarray(Ceps, float),
                    seps=np.asarray(seps, float),
                    p_raw=np.asarray(dbg_local["p_raw"], complex),
                    p_clean=np.asarray(dbg_local["p_clean"], complex),
                    p_snap=np.asarray(dbg_local["p_snap"], complex),
                    A=np.asarray(dbg_local["A"], complex),
                    L=np.asarray(dbg_local["L"], complex),
                    ATL=np.asarray(dbg_local["ATL"], complex),
                    D=np.asarray(dbg_local["D"], complex),
                    Em=np.asarray(dbg_local["Em"], float),
                    x=np.array([x], float),
                    kind=np.array([kind], int),
                )

        if len(Ems) == 0:
            return None
        if len(Ems) == 1:
            return Ems[0]

        order = np.argsort(xs)
        Em1, Em2 = Ems[order[0]], Ems[order[1]]
        x1, x2 = xs[order[0]], xs[order[1]]
        Em0 = Em1 + (Em1 - Em2) * (x1 / (x2 - x1))

        if DEBUG_DEGEN_LIMIT:
            print(f"[degen_limit] kind={kind}: Richardson using x1={x1:.3e}, x2={x2:.3e}")
            print(f"[degen_limit] kind={kind}: sha(Em0)={_sha256_arr(np.asarray(Em0,float))[:12]}")

        return np.asarray(Em0, float)

    for kind in (0, 1, 2):
        S = _symmetric_regularizer(kind=kind)
        out = try_one_regularizer(S, kind)
        if out is not None:
            return out

    raise RuntimeError("Degenerate-limit fallback failed: no stable Em found for any epsilon/regularizer.")


# ------------------------------------------------------------
# Deterministic degenerate fallback (root-splitting + extrapolation)
#
# Rationale:
#   In true/near repeated-root Stroh cases, (A,L) and especially D can become
#   ill-conditioned, causing Em to blow up (even if Eq.(12) is evaluated).
#   A practical, stable workaround is to *deterministically* split the roots by
#   a small, fixed perturbation to CinSS, evaluate Em with the standard
#   non-degenerate closed-form machinery, and extrapolate eps -> 0.
#
#   This is intentionally used ONLY when 'degenerate' is detected.
# ------------------------------------------------------------
def _Em_degenerate_limit_deterministic(
    CinSS: np.ndarray,
    s: np.ndarray,
    phi_deg: float,
    Nburger: float,
    *,
    eps_list=(1e-6, 3e-6, 1e-5, 3e-5, 1e-4),
    max_abs_Em: float = 1e12,
) -> np.ndarray:
    CinSS = np.asarray(CinSS, float)
    scale = _regularization_scale(CinSS)

    # Deterministic, diagonal (Voigt) perturbation pattern.
    # Chosen to be sign-alternating to break symmetry in a repeatable way
    # while keeping the perturbation small relative to ||CinSS||.
    S = np.diag([1.0, -1.0, 0.5, -0.5, 0.25, -0.25]).astype(float)

    xs: list[float] = []
    Ems: list[np.ndarray] = []

    for eps in eps_list:
        x = float(eps * scale)
        Ceps = CinSS + x * S
        seps = _sinss_from_cinss(Ceps)

        # Compute (p,A,L) on the perturbed (ideally non-degenerate) system.
        p, A, L, dbg = stroh_eigensystem_from_cinss(
            Ceps,
            s=seps,
            verbose=False,
            return_debug=True,
            snap_tol=1e-12,
            # Make degeneracy detection stricter here: we *want* eps to split roots.
            degeneracy_tol=1e-10,
        )

        # If still degenerate (eps too small), skip this point.
        if bool(dbg.get("degenerate", False)):
            continue

        if _is_bad_stroh_basis(A, L, tol=1e-8):
            continue

        Dm = D_from_stroh_bilinear(A, L, phi_deg, Nburger)
        Em = np.asarray(_FEm_from_ADR(A, Dm, p, Nburger, verbose=False), float)

        if not np.all(np.isfinite(Em)):
            continue
        if float(np.max(np.abs(Em))) > float(max_abs_Em):
            continue

        xs.append(x)
        Ems.append(Em)

    if len(Ems) < 3:
        raise RuntimeError(
            "Degenerate deterministic fallback failed: could not obtain >=3 stable Em(eps) samples."
        )

    # Use the smallest 3 eps points for a linear extrapolation Em(eps)=Em0 + a*eps.
    order = np.argsort(xs)
    xs_fit = np.asarray([xs[i] for i in order[:3]], dtype=float)
    Ems_fit = np.stack([Ems[i] for i in order[:3]], axis=0)  # (3,6,6)

    # Linear least squares for each matrix element.
    X = np.vstack([np.ones_like(xs_fit), xs_fit]).T  # (3,2)
    Y = Ems_fit.reshape(3, -1)  # (3,36)
    coeffs, *_ = np.linalg.lstsq(X, Y, rcond=None)  # (2,36)
    Em0 = coeffs[0, :].reshape(6, 6)

    if not np.all(np.isfinite(Em0)) or float(np.max(np.abs(Em0))) > float(max_abs_Em):
        raise RuntimeError("Degenerate deterministic fallback produced non-finite or implausibly large Em0.")

    return np.asarray(Em0, float)


# ------------------------------------------------------------
# Stroh eigensystem in slip frame
# ------------------------------------------------------------
def stroh_eigensystem_from_cinss(
    CinSS: np.ndarray,
    s: Optional[np.ndarray] = None,
    verbose: bool = False,
    return_debug: bool = False,
    *,
    snap_tol: float = 1e-12,
    degeneracy_tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    Q, R, T = _QRT_from_Cvoigt(CinSS)

    detT = float(np.linalg.det(T))
    if abs(detT) < 1e-14:
        T = T + (1e-12 * np.eye(3))

    N = _stroh_N(Q, R, T)
    w, v = np.linalg.eig(N)

    idx_pos = _select_three_im_positive_roots(w)

    p_raw = w[idx_pos]
    p_clean = canonicalize_roots(p_raw, tol=1e-12)
    p_snap = snap_roots(p_clean, tol=snap_tol)

    degenerate = (
        is_degenerate_after_chop(p_snap, tol=degeneracy_tol)
        or is_near_degenerate(p_clean, tol=degeneracy_tol)
    )

    # eigenvectors from eig (only trusted when non-degenerate)
    A_eig = v[0:3, idx_pos]
    L_eig = v[3:6, idx_pos]

    # ensure we have s for sorting
    if s is None:
        s = _sinss_from_cinss(CinSS)

    # non-degenerate path: keep eig vectors but make ordering + phase deterministic
    A_eig_use, L_eig_use = normalize_stroh_columns(A_eig, L_eig)
    p_clean_ord, A_eig_use, L_eig_use = sort_roots_like_mathematica(p_clean, A_eig_use, L_eig_use, s)

    # LL/AA path (used for degeneracy or bad basis)
    lambdas_snap = _lambdas_from_roots(p_snap, CinSS)
    LL = _LL_matrix(p_snap, lambdas_snap)
    AA = _A_matrix_from_KK_LL(p_snap, CinSS)

    # deterministic ordering for LL/AA
    p_snap_ord, AA, LL = sort_roots_like_mathematica(p_snap, AA, LL, s)

    AA_n, LL_n = normalize_stroh_columns(AA, LL)
    AA_n, LL_n = columnwise_bilinear_normalize(AA_n, LL_n, tol=1e-14)
    AA_n, LL_n = normalize_stroh_columns(AA_n, LL_n)
    AA_n, LL_n = rescale_stroh_columns(AA_n, LL_n, target=1.0)

    # decide which to use
    bad_eig = _is_bad_stroh_basis(A_eig_use, L_eig_use, tol=1e-8)

    if degenerate or bad_eig:
        p_use = p_snap_ord
        A_use, L_use = AA_n, LL_n
        method = "LLAA_colnorm_deg_snap"
    else:
        p_use = p_clean_ord
        A_use, L_use = A_eig_use, L_eig_use
        method = "eig"

    dbg: Dict[str, Any] = {
        "Q": Q, "R": R, "T": T, "detT": detT,
        "N": N,
        "eig_w": w,
        "eig_v": v,
        "idx_pos": np.array(idx_pos, dtype=int),

        "p_raw": p_raw,
        "p_clean": p_clean,
        "p_snap": p_snap,
        "degenerate": degenerate,
        "method": method,

        "A_eig_raw": A_eig,
        "L_eig_raw": L_eig,

        "lambdas": lambdas_snap,
        "LL_mathematica": LL,
        "AA_mathematica": AA,
        "LL_norm": LL_n,
        "AA_norm": AA_n,

        "G_eig_ATL": (A_eig_use.T @ L_eig_use),
        "G_LLAA_ATL": (AA_n.T @ LL_n),

        "A_used": A_use,
        "L_used": L_use,
        "p_used": p_use,
        "bad_eig": bad_eig,
    }

    if return_debug:
        return p_use, A_use, L_use, dbg

    return p_use, A_use, L_use


def stroh_eigensystem(cf_input: ContrastFactorInput, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    CinSS = stiffness_in_slip_system(cf_input, verbose=False)
    s = _sinss_from_cinss(CinSS)
    return stroh_eigensystem_from_cinss(CinSS, s=s, verbose=verbose)


# ------------------------------------------------------------
# Burgers in slip frame (debug)
# ------------------------------------------------------------
def _burgers_in_slip_frame(cf_input: ContrastFactorInput) -> np.ndarray:
    u, v, w = cf_input.slip_system.burgers_uvw
    b_cryst = np.array([float(u), float(v), float(w)], dtype=float)

    A_dir = _direct_basis_from_cell(cf_input.cell)
    b_cart = A_dir @ b_cryst

    e1, e2, e3 = build_slip_frame(cf_input)
    P = np.column_stack((e1, e2, e3))
    return P.T @ b_cart


# ------------------------------------------------------------
# Public API: elastic_E_matrix
# ------------------------------------------------------------
def elastic_E_matrix(
    cf_input: ContrastFactorInput,
    use_cinss_as_em: bool = False,
    n_phi: Optional[int] = None,
) -> np.ndarray:
    cache = _get_cf_cache(cf_input)

    ss = cf_input.slip_system
    C6 = np.asarray(cf_input.elastic.as_voigt_6x6(), dtype=float)

    key = (
        "Em",
        _EM_CACHE_VERSION,
        tuple(map(int, ss.plane_hkl)),
        tuple(map(int, ss.burgers_uvw)),
        float(ss.phi_deg),
        C6.round(12).tobytes(),
        float(cf_input.cell.a), float(cf_input.cell.b), float(cf_input.cell.c),
        float(cf_input.cell.alpha), float(cf_input.cell.beta), float(cf_input.cell.gamma),
        bool(use_cinss_as_em),
    )
    if key in cache:
        return cache[key].copy()

    CinSS = stiffness_in_slip_system(cf_input, verbose=False)
    s = _sinss_from_cinss(CinSS)

    if use_cinss_as_em:
        cache[key] = CinSS.copy()
        return cache[key].copy()

    p, A, L, stroh_dbg = stroh_eigensystem_from_cinss(
        CinSS, s=s, verbose=False, return_debug=True
    )

    # Nburger = |b| in Cartesian
    u, v, w = ss.burgers_uvw
    b_cryst = np.array([float(u), float(v), float(w)], dtype=float)
    A_dir = _direct_basis_from_cell(cf_input.cell)
    b_cart = A_dir @ b_cryst
    Nburger = float(np.linalg.norm(b_cart))

    deg = bool(stroh_dbg.get("degenerate", False))
    bad = _is_bad_stroh_basis(A, L, tol=1e-8)

    # diagnostic FEM from actually used (A,L,p)
    Dm = D_from_stroh_bilinear(A, L, ss.phi_deg, Nburger)
    Em_fem, fem_dbg = _FEm_from_ADR(A, Dm, p, Nburger, verbose=False, return_debug=True)

    if deg:
        # Degenerate (or near-degenerate) Stroh roots:
        # Use a deterministic root-splitting perturbation of CinSS, compute Em with the
        # standard non-degenerate machinery, and extrapolate eps -> 0.
        Em = _Em_degenerate_limit_deterministic(CinSS, s, ss.phi_deg, Nburger)
        fem_dbg["note"] = (
            "Em returned is from _Em_degenerate_limit_deterministic (deterministic root-splitting + extrapolation); "
            "Em inside FEm_dbg is diagnostic (_FEm_from_ADR) using the current (A,L,p)."
        )
    elif bad:
        Em = _Em_degenerate_limit(CinSS, s, ss.phi_deg, Nburger)
        fem_dbg["note"] = (
            "Em returned is from _Em_degenerate_limit; "
            "Em inside FEm_dbg is diagnostic (_FEm_from_ADR) using the current (A,L,p)."
        )
    else:
        Em = Em_fem

    if DEBUG_DUMP_FEM_NPZ:
        phi = float(cf_input.slip_system.phi_deg)
        dump_fem_inputs(f"{DEBUG_FEM_NPZ_PREFIX}{phi:.0f}.npz", fem_dbg, Nburger)

    cache[key] = np.asarray(Em, float).copy()
    return cache[key].copy()


# ------------------------------------------------------------
# Debug helpers / debug API
# ------------------------------------------------------------
import hashlib


def _sha256_arr(x: np.ndarray) -> str:
    x = np.ascontiguousarray(x)
    return hashlib.sha256(x.tobytes()).hexdigest()


def _fmt_cplx_vec(v: np.ndarray) -> str:
    v = np.asarray(v, dtype=complex).ravel()
    return "[" + ", ".join(f"{z.real:+.3e}{z.imag:+.3e}j" for z in v) + "]"


def elastic_E_matrix_debug(cf_input: ContrastFactorInput) -> tuple[np.ndarray, Dict[str, Any]]:
    e1, e2, e3 = build_slip_frame(cf_input)

    A_dir = _direct_basis_from_cell(cf_input.cell)
    Ainv = np.linalg.inv(A_dir)
    P_crys = (Ainv @ np.column_stack((e1, e2, e3))).T

    CinSS, cinss_dbg = stiffness_in_slip_system(cf_input, verbose=False, return_debug=True)
    s = _sinss_from_cinss(CinSS)

    p, A_use, L_use, stroh_dbg = stroh_eigensystem_from_cinss(
        CinSS, s=s, verbose=False, return_debug=True
    )

    # Nburger
    u, v, w = cf_input.slip_system.burgers_uvw
    b_cryst = np.array([float(u), float(v), float(w)], dtype=float)
    b_cart = A_dir @ b_cryst
    Nburger = float(np.linalg.norm(b_cart))

    deg = bool(stroh_dbg.get("degenerate", False))
    bad = _is_bad_stroh_basis(A_use, L_use, tol=1e-8)

    if deg:
        Dm = D_from_stroh_bilinear(A_use, L_use, cf_input.slip_system.phi_deg, Nburger)
        Em = _Em_degenerate_limit_deterministic(CinSS, s, cf_input.slip_system.phi_deg, Nburger)
        Em_fem, fem_dbg = _FEm_from_ADR(A_use, Dm, p, Nburger, verbose=False, return_debug=True)
        fem_dbg["note"] = "Em returned is from _Em_degenerate_limit_deterministic (deterministic root-splitting + extrapolation); Em in FEm_dbg is diagnostic (_FEm_from_ADR)."
    elif bad:
        Em = _Em_degenerate_limit(CinSS, s, cf_input.slip_system.phi_deg, Nburger)
        Dm = D_from_stroh_bilinear(A_use, L_use, cf_input.slip_system.phi_deg, Nburger)
        Em_fem, fem_dbg = _FEm_from_ADR(A_use, Dm, p, Nburger, verbose=False, return_debug=True)
        fem_dbg["note"] = "Em returned is from _Em_degenerate_limit; Em in FEm_dbg is diagnostic (_FEm_from_ADR)."
    else:
        Dm = D_from_stroh_bilinear(A_use, L_use, cf_input.slip_system.phi_deg, Nburger)
        Em, fem_dbg = _FEm_from_ADR(A_use, Dm, p, Nburger, verbose=False, return_debug=True)

    b_slip = _burgers_in_slip_frame(cf_input)

    dbg: Dict[str, Any] = {
        "P_rows_cart": cinss_dbg["P_rows_cart"],
        "CinSS_dbg": cinss_dbg,
        "P_rows_crys": P_crys,
        "CinSS": CinSS,

        "p": p,
        "A_used": A_use,
        "L_used": L_use,

        "b_slip": b_slip,
        "D_bilinear": Dm,
        "Nburger": Nburger,

        "stroh_dbg": stroh_dbg,
        "FEm_dbg": fem_dbg,

        "p_raw": stroh_dbg.get("p_raw"),
        "p_clean": stroh_dbg.get("p_clean"),
        "p_snap": stroh_dbg.get("p_snap"),
        "degenerate": stroh_dbg.get("degenerate"),
        "method": stroh_dbg.get("method"),

        "LL_mathematica": stroh_dbg.get("LL_mathematica"),
        "AA_mathematica": stroh_dbg.get("AA_mathematica"),
        "G_eig_ATL": stroh_dbg.get("G_eig_ATL"),
        "G_LLAA_ATL": stroh_dbg.get("G_LLAA_ATL"),
    }

    return Em, dbg
