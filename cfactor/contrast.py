# cfactor/contrast.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, List, Tuple

import numpy as np

from .models import ContrastFactorInput
from .geometry import reciprocal_basis, direct_basis, build_slip_frame
from .symmetry import build_point_group_rotations_from_laue
from .elasticity import elastic_E_matrix

# ===================== PATCH: use FULL 48-frame average ======================
# Goal:
#   - Powder average should be over ALL operator-applied frames (≈48 for cubic m3m with improper)
#   - Keep the old Mathematica-pruned behavior available for debugging, but DO NOT use it for averaging.
#
# What to change:
#   1) Replace your current EquivalentSlipSystem(...) with TWO functions:
#        - EquivalentSlipSystemFull(...)   -> returns all operator-applied frames (NO pruning)
#        - EquivalentSlipSystemPruned(...) -> your current Mathematica DeleteCases pruning (legacy)
#   2) Update AveGeoMatrix / AllgInSlipSystem / GeometricalMatrixNumber / dump_powder_average_debug
#      to call EquivalentSlipSystemFull.
#   3) Update dump_selected_12_frames_debug_m3m to call EquivalentSlipSystemPruned.
# ============================================================================


def _build_operator_applied_frames(
    cf_input: ContrastFactorInput,
    include_improper: bool = True,
    chop_tol: float = 1e-6,
) -> List[np.ndarray]:
    """
    Build ALL frames in OperatorInE order (NO pruning):
      t1 = SlipSystem()  (rows)
      t2 = OperatorInE()
      frames = Chop( {R.e1, R.e2, R.e3}, chop_tol )
    """
    t1 = SlipSystem(cf_input)  # rows = (e1,e2,e3)
    ops = OperatorInE(cf_input, include_improper=include_improper)

    frames: List[np.ndarray] = []
    for R in ops:
        Rm = np.asarray(R, float).reshape(3, 3)
        F = np.vstack((Rm @ t1[0], Rm @ t1[1], Rm @ t1[2]))
        frames.append(Chop(F, tol=chop_tol))
    return frames

def v6_mathematica(tau: np.ndarray) -> np.ndarray:
    t1, t2, t3 = np.asarray(tau, float).reshape(3)
    # Voigt order: {11,22,33,23,13,12}
    return np.array([t1*t1, t2*t2, t3*t3, t2*t3, t1*t3, t1*t2], dtype=float)


def debug_single_crystal_details(
    cf_input: ContrastFactorInput,
    hkl: Sequence[int] | Sequence[float],
    F_rows: np.ndarray,
    Em: np.ndarray,
) -> dict:
    B = reciprocal_basis(cf_input)
    hkl_vec = np.asarray(hkl, float).reshape(3)

    g_cart = B @ hkl_vec                 # contains d*
    dstar = float(np.linalg.norm(g_cart))
    g1 = g_cart / dstar if dstar != 0.0 else np.zeros(3)

    F = np.asarray(F_rows, float).reshape(3, 3)
    FinvT = np.linalg.inv(F.T)
    tau = FinvT @ g1

    v6 = v6_mathematica(tau)
    Em = np.asarray(Em, float).reshape(6, 6)

    vEm = v6 @ Em
    c = float(vEm @ v6)

    return {
        "hkl": tuple(int(x) for x in np.rint(hkl_vec)),
        "g_cart": g_cart,
        "dstar": dstar,
        "g1": g1,
        "F_rows": F,
        "FinvT": FinvT,
        "tau": tau,
        "v6": v6,
        "Em": Em,
        "vEm": vEm,
        "c": c,
    }


# =============================================================================
# Small helpers kept for legacy single-crystal and scattering parts
# =============================================================================

def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _tau_for_hkl_in_frame_rows(
    cf_input: ContrastFactorInput,
    hkl: Sequence[int] | Sequence[float],
    F_rows: np.ndarray,
) -> np.ndarray:
    """
    Legacy wrapper: identical to gInSlipSystem() (Mathematica tau definition).
    """
    return gInSlipSystem(cf_input, hkl, F_rows)


def _G_from_tau_mathematica(tau: np.ndarray) -> np.ndarray:
    """
    Legacy wrapper: same indexing pattern as GeometricalMatrixNumber().
    """
    t = np.asarray(tau, float).reshape(3)
    G = np.zeros((6, 6), float)
    for j in range(2):
        for i in range(3):
            row = 3 * j + i
            for m in range(2):
                for n in range(3):
                    col = 3 * m + n
                    G[row, col] = t[i] * t[j] * t[n] * t[m]
    return G


# =============================================================================
# Mathematica-exact powder average implementation (Step 2A -> forward)
# Implements: SlipSystem, OperatorInE, EquivalentSlipSystem, AllgInSlipSystem,
#             GeometricalMatrixNumber, AveGeoMatrix, AveContrastFactor
# =============================================================================

def Chop(A, tol: float = 1e-6) -> np.ndarray:
    """
    Mathematica Chop[x, tol]:
      sets entries with Abs[x] < tol to exact 0.0
    """
    A = np.asarray(A, float).copy()
    A[np.abs(A) < tol] = 0.0
    return A


def _chop_key(A: np.ndarray, tol: float = 1e-6) -> tuple[int, ...]:
    """
    Emulate Mathematica "Chop then exact equality" robustly for floats:
      - Chop small values to 0
      - Quantize onto the tol-grid
      - Compare integer keys
    """
    Ac = Chop(A, tol=tol)
    Q = np.rint(Ac / tol).astype(np.int64)
    return tuple(Q.reshape(-1).tolist())


def frameEqQ(A: np.ndarray, B: np.ndarray, tol: float = 1e-6) -> bool:
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    d = Chop(A - B, tol=tol)
    return np.array_equal(d, np.zeros_like(d))


def vecEqQ(a: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> bool:
    a = np.asarray(a, float).reshape(-1)
    b = np.asarray(b, float).reshape(-1)
    d = Chop(a - b, tol=tol)
    return np.array_equal(d, np.zeros_like(d))


def SlipSystem(cf_input: ContrastFactorInput) -> np.ndarray:
    """
    Mathematica:
      SlipSystem[glideplane, burgervector, phi, cp] -> {e1,e2,e3}

    IMPORTANT:
      Do NOT Chop here. Early chopping can collapse distinct frames at phi=60°
      before symmetry expansion. We only Chop at the t3 stage (1e-6), matching MMA.
    """
    e1, e2, e3 = build_slip_frame(cf_input)
    return np.vstack((e1, e2, e3))


def OperatorInE(cf_input: ContrastFactorInput, include_improper: bool = True) -> List[np.ndarray]:
    """
    Match Mathematica OperatorInE[sym, cp] ordering as closely as possible.

    Mathematica path:
      symop = Sort[...]  (SpaceGroup)
      t22   = rotations in that sorted order
      t2    = Prepend[t22, Identity]
      tf    = while-loop that keeps first occurrence and DeleteCases duplicates

    For cubic groups, rotation matrices are integer {-1,0,1}. Mathematica Sort
    matches lexicographic order of the matrix-as-nested-list. We emulate that by
    sorting by row-major integer tuple after Chop/rounding.
    """
    ops_raw = build_point_group_rotations_from_laue(cf_input, include_improper=include_improper)

    mats: List[np.ndarray] = []
    for R in ops_raw:
        Rm = np.asarray(R, float).reshape(3, 3)
        Rc = Chop(Rm, tol=1e-12)
        Ri = np.rint(Rc).astype(int)
        if np.allclose(Rc, Ri.astype(float), atol=1e-12, rtol=0.0):
            mats.append(Ri.astype(float))
        else:
            mats.append(Rc)

    def _mma_sort_key(Rm: np.ndarray) -> tuple:
        Rc = Chop(Rm, tol=1e-12)
        Ri = np.rint(Rc).astype(int)
        if np.allclose(Rc, Ri.astype(float), atol=1e-12, rtol=0.0):
            flat = tuple(int(x) for x in Ri.reshape(-1).tolist())
        else:
            Q = np.rint(Rc / 1e-12).astype(np.int64)
            flat = tuple(int(x) for x in Q.reshape(-1).tolist())
        return flat

    mats_sorted = sorted(mats, key=_mma_sort_key)

    ops = [np.eye(3, dtype=float)] + mats_sorted

    out: List[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for R in ops:
        key = _chop_key(R, tol=1e-12)
        if key in seen:
            continue
        seen.add(key)
        out.append(R)
    return out


def _EquivalentSlipSystem_DeleteCases_predicate(x: np.ndarray, k: np.ndarray, tol: float = 1e-6) -> bool:
    """
    EXACT predicate from Mathematica EquivalentSlipSystem:

      x==k ||
      x[[1]] == -k[[1]] ||
      x == {k1,-k2,k3} ||
      x == {-k1,k2,k3} ||
      x == {k1,k2,-k3} ||
      x == {k1,-k2,-k3} ||
      x == {-k1,-k2,k3}

    with Chop[...,1e-6] applied to x and k beforehand.
    Frames are 3x3 with ROWS = (e1,e2,e3).
    """
    x = np.asarray(x, float).reshape(3, 3)
    k = np.asarray(k, float).reshape(3, 3)
    k1, k2, k3 = k[0], k[1], k[2]

    return (
        frameEqQ(x, k, tol=tol)
        or vecEqQ(x[0], -k1, tol=tol)
        or frameEqQ(x, np.vstack((k1, -k2, k3)), tol=tol)
        or frameEqQ(x, np.vstack((-k1, k2, k3)), tol=tol)
        or frameEqQ(x, np.vstack((k1, k2, -k3)), tol=tol)
        or frameEqQ(x, np.vstack((k1, -k2, -k3)), tol=tol)
        or frameEqQ(x, np.vstack((-k1, -k2, k3)), tol=tol)
    )


def EquivalentSlipSystemFull(
    cf_input: ContrastFactorInput,
    include_improper: bool = True,
) -> List[np.ndarray]:
    """
    FULL set of operator-applied equivalent slip frames (NO pruning).
    This is what you should use for powder averaging (48 for cubic m3m if include_improper=True).
    """
    return _build_operator_applied_frames(cf_input, include_improper=include_improper, chop_tol=1e-6)


def EquivalentSlipSystemPruned(
    cf_input: ContrastFactorInput,
    include_improper: bool = True,
) -> List[np.ndarray]:
    """
    Legacy Mathematica-pruned EquivalentSlipSystem (While/DeleteCases).
    KEEP ONLY for comparison/debugging against the original package.
    """
    t3 = _build_operator_applied_frames(cf_input, include_improper=include_improper, chop_tol=1e-6)

    tf: List[np.ndarray] = []
    while len(t3) > 0:
        k = t3[0]
        tf.append(k)
        t3 = [x for x in t3 if not _EquivalentSlipSystem_DeleteCases_predicate(x, k, tol=1e-6)]

    return tf


def gInSlipSystem(cf_input: ContrastFactorInput, hkl: Sequence[int] | Sequence[float], frame_rows: np.ndarray) -> np.ndarray:
    """
    Mathematica:
      g1 = RepToOrthoN[H,cp]  ~ unit(B_rec @ H)
      tau = Inverse[Transpose[frame]] . g1
    """
    B = reciprocal_basis(cf_input)
    g1 = np.asarray(B @ np.asarray(hkl, float).reshape(3), float)
    ng = float(np.linalg.norm(g1))
    if ng == 0.0:
        return np.zeros(3, float)
    g1 = g1 / ng

    F = np.asarray(frame_rows, float).reshape(3, 3)
    return np.linalg.inv(F.T) @ g1


def AllgInSlipSystem(
    cf_input: ContrastFactorInput,
    hkl: Sequence[int] | Sequence[float],
    include_improper: bool = True,
) -> List[np.ndarray]:
    """
    g in each equivalent slip system (FULL set, no pruning)
    """
    frames = EquivalentSlipSystem(cf_input, include_improper=include_improper)
    return [gInSlipSystem(cf_input, hkl, F) for F in frames]


def GeometricalMatrixNumber(
    cf_input: ContrastFactorInput,
    hkl: Sequence[int] | Sequence[float],
    number: int,
    include_improper: bool = True,
) -> np.ndarray:
    """
    Here number is 1-based (Mathematica indexing), but now over FULL frames list.
    """
    frames = EquivalentSlipSystem(cf_input, include_improper=include_improper)
    if number < 1 or number > len(frames):
        raise IndexError(f"Slip-system number {number} out of range 1..{len(frames)}")

    tau = gInSlipSystem(cf_input, hkl, frames[number - 1])
    t = np.asarray(tau, float).reshape(3)

    G = np.zeros((6, 6), float)
    for j in range(2):
        for i in range(3):
            row = 3 * j + i
            for m in range(2):
                for n in range(3):
                    col = 3 * m + n
                    G[row, col] = t[i] * t[j] * t[n] * t[m]
    return G


def AveGeoMatrix(
    cf_input: ContrastFactorInput,
    hkl: Sequence[int] | Sequence[float],
    include_improper: bool = True,
) -> np.ndarray:
    """
    Average geometrical matrix over ALL operator-applied slip frames (FULL set).
    """
    frames = EquivalentSlipSystem(cf_input, include_improper=include_improper)
    if not frames:
        return np.zeros((6, 6), float)

    acc = np.zeros((6, 6), float)
    for idx in range(1, len(frames) + 1):
        acc += GeometricalMatrixNumber(cf_input, hkl, idx, include_improper=include_improper)
    return acc / float(len(frames))


def AveContrastFactor(
    cf_input: ContrastFactorInput,
    hkl: Sequence[int] | Sequence[float],
    include_improper: bool = True,
    elasticmatrix: np.ndarray | None = None,
) -> float:
    """
    Mathematica AveContrastFactor:
      Cbar = Sum_{i,j=1..6} AveGeoMatrix[[i,j]] * elasticmatrix[[i,j]]
    """
    if elasticmatrix is None:
        elasticmatrix = elastic_E_matrix(cf_input)

    Gbar = AveGeoMatrix(cf_input, hkl, include_improper=include_improper)
    Em = np.asarray(elasticmatrix, float).reshape(6, 6)
    return float(np.sum(Gbar * Em))


def contrast_factor_for_hkl(
    cf_input: ContrastFactorInput,
    hkl: Sequence[int] | Sequence[float],
    include_improper: bool = True,
    Em: np.ndarray | None = None,
) -> float:
    """
    Keep your public API, but now it is a direct Mathematica clone (AveContrastFactor),
    using FULL (unpruned) symmetry-applied frames through AveGeoMatrix.
    """
    return AveContrastFactor(cf_input, hkl, include_improper=include_improper, elasticmatrix=Em)


def dump_powder_average_debug(
    cf_input: ContrastFactorInput,
    hkl: Sequence[int] | Sequence[float],
    include_improper: bool = True,
    Em: np.ndarray | None = None,
    precision: int = 12,
) -> float:
    """
    Debug driven by FULL (unpruned) symmetry-applied frames.
    """
    if Em is None:
        Em = elastic_E_matrix(cf_input)

    frames = EquivalentSlipSystem(cf_input, include_improper=include_improper)

    print("\n============================================================")
    print(f"[POWDER DEBUG - FULL] phi={float(cf_input.slip_system.phi_deg):g} deg  hkl={tuple(hkl)}")
    print(f"[POWDER DEBUG - FULL] Nslip = {len(frames)}")

    Gs: List[np.ndarray] = []
    for s, F in enumerate(frames, start=1):
        tau = gInSlipSystem(cf_input, hkl, F)
        t = np.asarray(tau, float).reshape(3)
        G = np.zeros((6, 6), float)
        for j in range(2):
            for i in range(3):
                row = 3 * j + i
                for m in range(2):
                    for n in range(3):
                        col = 3 * m + n
                        G[row, col] = t[i] * t[j] * t[n] * t[m]

        c = float(np.sum(G * Em))

        print(f"\n-- slip frame s={s} --")
        print("F_rows (3x3, rows=e1,e2,e3) =\n", np.array_str(F, precision=precision, suppress_small=False))
        print("tau =", np.array_str(tau, precision=precision, suppress_small=False))
        print("C_single =", f"{c:.15g}")

        Gs.append(G)

    Gbar = np.mean(Gs, axis=0) if Gs else np.zeros((6, 6), float)
    Cbar = float(np.sum(Gbar * Em))

    print("\n-- Em (6x6) --\n", np.array_str(Em, precision=precision, suppress_small=False))
    print("\n-- Gbar (6x6) --\n", np.array_str(Gbar, precision=precision, suppress_small=False))
    print("\n-- Cbar --", f"{Cbar:.15g}")
    print("============================================================\n")

    return Cbar


# =============================================================================
# Single-crystal orbit over equivalent HKLs (kept for main.py)
# =============================================================================

def equivalent_hkls_from_point_group(
    cf_input: ContrastFactorInput,
    hkl: Sequence[int] | Sequence[float],
    tol: float = 1e-5,
    include_improper: bool = False,
) -> List[Tuple[int, int, int]]:
    B = reciprocal_basis(cf_input)
    hkl_vec = np.asarray(hkl, float).reshape(3)
    g_cart = B @ hkl_vec

    rots = build_point_group_rotations_from_laue(cf_input, include_improper=include_improper)

    eq: set[Tuple[int, int, int]] = set()
    for R in rots:
        g_rot = np.asarray(R, float) @ g_cart
        hkl_real = np.linalg.solve(B, g_rot)
        hkl_round = np.rint(hkl_real).astype(int)
        if np.linalg.norm(hkl_real - hkl_round) <= tol:
            eq.add((int(hkl_round[0]), int(hkl_round[1]), int(hkl_round[2])))

    eq.discard((0, 0, 0))
    if not eq:
        return [(int(hkl_vec[0]), int(hkl_vec[1]), int(hkl_vec[2]))]
    return sorted(eq)


def _P_rows_slipsystem(cf_input: ContrastFactorInput) -> np.ndarray:
    e1, e2, e3 = build_slip_frame(cf_input)
    return np.vstack((e1, e2, e3))


def contrast_factor_single_slip_frame(
    cf_input: ContrastFactorInput,
    hkl: Sequence[int] | Sequence[float],
    F_rows: np.ndarray,
    Em: np.ndarray | None = None,
) -> float:
    if Em is None:
        Em = elastic_E_matrix(cf_input)
    tau = _tau_for_hkl_in_frame_rows(cf_input, hkl, F_rows)
    G = _G_from_tau_mathematica(tau)
    return float(np.sum(G * Em))


def single_crystal_orbit_over_equiv_hkls(
    cf_input: ContrastFactorInput,
    hkl: Sequence[int] | Sequence[float],
    include_improper: bool = False,
    tol: float = 1e-5,
    F_rows: np.ndarray | None = None,
    Em: np.ndarray | None = None,
) -> List[Tuple[Tuple[int, int, int], float]]:
    hkls_eq = equivalent_hkls_from_point_group(cf_input, hkl, tol=tol, include_improper=include_improper)

    if F_rows is None:
        F_rows = _P_rows_slipsystem(cf_input)
    if Em is None:
        Em = elastic_E_matrix(cf_input)

    out: List[Tuple[Tuple[int, int, int], float]] = []
    for heq in hkls_eq:
        out.append((heq, contrast_factor_single_slip_frame(cf_input, heq, F_rows=F_rows, Em=Em)))
    return out


# =============================================================================
# Scattering helper (kept for main.py)
# =============================================================================

def _auto_unique_axis_letter(cf_input: ContrastFactorInput) -> str:
    """
    Decide which crystallographic axis should be used for delta when ref_axis="auto".

    Conventions:
      - tetragonal/hexagonal/trigonal/rhombohedral -> c
      - monoclinic -> detect unique axis from which TWO angles are ~90:
            unique-b: alpha~90 and gamma~90 (beta != 90)
            unique-c: alpha~90 and beta~90  (gamma != 90)
            unique-a: beta~90  and gamma~90 (alpha != 90)
        If ambiguous, fall back to b (common setting).
      - orthorhombic -> choose the *longest lattice parameter* as a stable plotting axis
        (helps when you use nonstandard axis permutations).
      - otherwise -> c
    """
    cell = cf_input.cell
    lat = str(cell.lattice).lower()

    if lat in ("tetragonal", "hexagonal", "trigonal", "rhombohedral"):
        return "c"

    if lat == "monoclinic":
        a90 = abs(float(cell.alpha) - 90.0) < 1e-6
        b90 = abs(float(cell.beta) - 90.0) < 1e-6
        g90 = abs(float(cell.gamma) - 90.0) < 1e-6

        # detect unique axis from the pair of 90° angles
        if a90 and g90 and not b90:
            return "b"   # unique-b
        if a90 and b90 and not g90:
            return "c"   # unique-c
        if b90 and g90 and not a90:
            return "a"   # unique-a

        # fallback if someone gives a weird/near-orthogonal mono cell
        return "b"

    if lat == "orthorhombic":
        a = float(cell.a); b = float(cell.b); c = float(cell.c)
        if a >= b and a >= c:
            return "a"
        if b >= a and b >= c:
            return "b"
        return "c"

    return "c"


def _reference_axis_cartesian(cf_input: ContrastFactorInput, ref_axis: str = "auto") -> np.ndarray:
    """
    Reference axis used to define delta (angle between g and a chosen crystal axis).
    """
    A_dir = direct_basis(cf_input)  # columns = a,b,c in Cartesian

    ra = ref_axis.lower()
    if ra in ("a", "b", "c"):
        idx = {"a": 0, "b": 1, "c": 2}[ra]
        return _unit(A_dir[:, idx])

    # auto
    ax = _auto_unique_axis_letter(cf_input)
    idx = {"a": 0, "b": 1, "c": 2}[ax]
    return _unit(A_dir[:, idx])



def scattering_phi_deg(
    cf_input: ContrastFactorInput,
    hkl: Sequence[int] | Sequence[float],
    ref_axis: str = "auto",
) -> float:
    hkl_vec = np.asarray(hkl, float).reshape(3)
    B_rec = reciprocal_basis(cf_input)
    g_cart = B_rec @ hkl_vec
    ng = float(np.linalg.norm(g_cart))
    if ng == 0.0:
        return 0.0
    g_hat = g_cart / ng

    axis_hat = _reference_axis_cartesian(cf_input, ref_axis=ref_axis)
    cosphi = float(np.clip(np.dot(g_hat, axis_hat), -1.0, 1.0))
    cosphi = abs(cosphi)  # fold 0..180 into 0..90
    return float(np.degrees(np.arccos(cosphi)))
   

def dump_full_48_frames_debug_m3m(
    cf_input: ContrastFactorInput,
    hkl: Sequence[int] | Sequence[float],
    include_improper: bool = True,
    Em: np.ndarray | None = None,
    precision: int = 12,
) -> float:
    """
    FULL 48-frame debug (cubic m3m / Oh-like Laue):
      - prints ALL operator-applied frames BEFORE EquivalentSlipSystem pruning
      - for each: e1,e2,e3 (rows), tau, G, C_single = Sum(G*Em)
      - prints Em, arithmetic mean G (over ALL frames), and Cbar (over ALL frames)

    NOTE:
      This is intentionally NOT the 12-frame pruned list.
      It shows the raw 48 (or whatever OperatorInE returns) symmetry-applied frames.
    """
    if Em is None:
        Em = elastic_E_matrix(cf_input)
    Em = np.asarray(Em, float).reshape(6, 6)

    t1 = SlipSystem(cf_input)
    ops = OperatorInE(cf_input, include_improper=include_improper)

    frames: List[np.ndarray] = []
    dets: List[float] = []
    for R in ops:
        Rm = np.asarray(R, float).reshape(3, 3)
        F = np.vstack((Rm @ t1[0], Rm @ t1[1], Rm @ t1[2]))
        frames.append(Chop(F, tol=1e-6))
        dets.append(float(np.linalg.det(Rm)))

    print("\n============================================================")
    print("[FULL 48 DEBUG - m3m] Raw operator-applied frames (NO pruning)")
    print(f"[FULL 48 DEBUG - m3m] phi={float(cf_input.slip_system.phi_deg):g} deg  hkl={tuple(hkl)}")
    print(f"[FULL 48 DEBUG - m3m] Nops = {len(ops)}  (include_improper={include_improper})")

    Gs: List[np.ndarray] = []
    Cs: List[float] = []

    for idx, (F, detR) in enumerate(zip(frames, dets), start=1):
        tau = gInSlipSystem(cf_input, hkl, F)
        G = _G_from_tau_mathematica(tau)
        c = float(np.sum(G * Em))

        print(f"\n-- op/frame #{idx:02d} -- det(R)={detR:+.6f}")
        print("F_rows (e1,e2,e3 as ROWS) =\n", np.array_str(F, precision=precision, suppress_small=False))
        print("e1 =", np.array_str(F[0], precision=precision, suppress_small=False))
        print("e2 =", np.array_str(F[1], precision=precision, suppress_small=False))
        print("e3 =", np.array_str(F[2], precision=precision, suppress_small=False))
        print("tau =", np.array_str(tau, precision=precision, suppress_small=False))
        print("G (6x6) =\n", np.array_str(G, precision=precision, suppress_small=False))
        print("C_single =", f"{c:.15g}")

        Gs.append(G)
        Cs.append(c)

    Gbar = np.mean(Gs, axis=0) if Gs else np.zeros((6, 6), float)
    Cbar_from_Gbar = float(np.sum(Gbar * Em))
    Cbar_from_Cs = float(np.mean(Cs)) if Cs else 0.0

    print("\n-- Em (6x6) --\n", np.array_str(Em, precision=precision, suppress_small=False))
    print("\n-- Gbar over ALL frames (6x6) --\n", np.array_str(Gbar, precision=precision, suppress_small=False))
    print("\n-- Cbar --")
    print("  from Gbar . Em =", f"{Cbar_from_Gbar:.15g}")
    print("  mean(C_single) =", f"{Cbar_from_Cs:.15g}")
    print("============================================================\n")

    return Cbar_from_Gbar


def dump_selected_12_frames_debug_m3m(
    cf_input: ContrastFactorInput,
    hkl: Sequence[int] | Sequence[float],
    include_improper: bool = True,
    Em: np.ndarray | None = None,
    precision: int = 12,
) -> float:
    """
    Debug the legacy Mathematica-pruned frames (typically 12).
    """
    if Em is None:
        Em = elastic_E_matrix(cf_input)
    Em = np.asarray(Em, float).reshape(6, 6)

    frames = EquivalentSlipSystemPruned(cf_input, include_improper=include_improper)

    print("\n============================================================")
    print("[SELECTED FRAMES DEBUG - MMA] Pruned frames from EquivalentSlipSystem (legacy)")
    print(f"[SELECTED FRAMES DEBUG - MMA] phi={float(cf_input.slip_system.phi_deg):g} deg  hkl={tuple(hkl)}")
    print(f"[SELECTED FRAMES DEBUG - MMA] include_improper={include_improper}")
    print(f"[SELECTED FRAMES DEBUG - MMA] N_selected = {len(frames)}")

    Gs: List[np.ndarray] = []
    Cs: List[float] = []

    for s, F in enumerate(frames, start=1):
        tau = gInSlipSystem(cf_input, hkl, F)
        G = _G_from_tau_mathematica(tau)
        c = float(np.sum(G * Em))

        print(f"\n-- selected frame s={s:02d} --")
        print("F_rows (e1,e2,e3 as ROWS) =\n", np.array_str(F, precision=precision, suppress_small=False))
        print("tau =", np.array_str(tau, precision=precision, suppress_small=False))
        print("C_single =", f"{c:.15g}")

        Gs.append(G)
        Cs.append(c)

    Gbar = np.mean(Gs, axis=0) if Gs else np.zeros((6, 6), float)
    Cbar_from_Gbar = float(np.sum(Gbar * Em))
    Cbar_from_Cs = float(np.mean(Cs)) if Cs else 0.0

    print("\n-- Cbar over SELECTED (pruned) frames --")
    print("  from Gbar·Em =", f"{Cbar_from_Gbar:.15g}")
    print("  mean(C_single) =", f"{Cbar_from_Cs:.15g}")
    print("============================================================\n")

    return Cbar_from_Gbar


def _equiv_delete_reason(x: np.ndarray, k: np.ndarray, tol: float = 1e-6) -> str | None:
    """
    Return a string identifying WHICH Mathematica DeleteCases clause matched,
    or None if x is NOT deleted relative to seed k.

    Clauses (as in your docstring):
      (1) x == k
      (2) x[[1]] == -k[[1]]
      (3) x == { k1, -k2,  k3 }
      (4) x == { -k1, k2,  k3 }
      (5) x == { k1,  k2, -k3 }
      (6) x == { k1, -k2, -k3 }
      (7) x == { -k1,-k2,  k3 }
    """
    x = np.asarray(x, float).reshape(3, 3)
    k = np.asarray(k, float).reshape(3, 3)
    k1, k2, k3 = k[0], k[1], k[2]

    if frameEqQ(x, k, tol=tol):
        return "(1) x == k"
    if vecEqQ(x[0], -k1, tol=tol):
        return "(2) x[[1]] == -k[[1]]  (row1 flip only)"
    if frameEqQ(x, np.vstack((k1, -k2, k3)), tol=tol):
        return "(3) { k1, -k2,  k3 }"
    if frameEqQ(x, np.vstack((-k1, k2, k3)), tol=tol):
        return "(4) { -k1, k2,  k3 }"
    if frameEqQ(x, np.vstack((k1, k2, -k3)), tol=tol):
        return "(5) { k1,  k2, -k3 }"
    if frameEqQ(x, np.vstack((k1, -k2, -k3)), tol=tol):
        return "(6) { k1, -k2, -k3 }"
    if frameEqQ(x, np.vstack((-k1, -k2, k3)), tol=tol):
        return "(7) { -k1,-k2,  k3 }"

    return None


def all_slip_frames_with_ops48(
    cf_input: ContrastFactorInput,
    include_improper: bool = True,
) -> List[Tuple[int, float, np.ndarray]]:
    """
    Returns a list of tuples: (op_index_1based, det(R), F_rows)
    with EXACT construction order:
      t1 = SlipSystem()
      t2 = OperatorInE()
      t3 = Chop( Table[{i, Det[t2[[i]]], {R.e1,R.e2,R.e3}}, {i}], 1e-6 )
    """
    t1 = SlipSystem(cf_input)  # rows
    t2 = OperatorInE(cf_input, include_improper=include_improper)

    out: List[Tuple[int, float, np.ndarray]] = []
    for i0, R in enumerate(t2, start=1):
        Rm = np.asarray(R, float).reshape(3, 3)
        F = np.vstack((Rm @ t1[0], Rm @ t1[1], Rm @ t1[2]))
        F = Chop(F, tol=1e-6)
        out.append((i0, float(np.linalg.det(Rm)), F))
    return out


def dump_equiv_pruning_trace(
    cf_input: ContrastFactorInput,
    include_improper: bool = True,
    tol: float = 1e-6,
    precision: int = 12,
    *,
    hkl: Sequence[int] | Sequence[float] | None = None,
    Em: np.ndarray | None = None,
) -> List[np.ndarray]:
    """
    Prints a Mathematica-like While/DeleteCases trace for EquivalentSlipSystem.

    NOTE: this traces the PRUNED selection behavior (EquivalentSlipSystemPruned).
    """
    if Em is not None:
        Em = np.asarray(Em, float).reshape(6, 6)

    raw = all_slip_frames_with_ops48(cf_input, include_improper=include_improper)

    print("\n============================================================")
    print("[EQUIV PRUNING TRACE - MMA-STYLE]")
    print(f"phi={float(cf_input.slip_system.phi_deg):g} deg")
    print(f"Nops(frames)={len(raw)}  include_improper={include_improper}")
    print("--- Raw list t3 = {opIndex, det, F_rows} ---")

    for k, (op_idx, detR, F) in enumerate(raw, start=1):
        print(f"\n[raw frame #{k:02d}] op#={op_idx:02d}  det={detR:+.12f}")
        print("F_rows =\n", np.array_str(F, precision=precision, suppress_small=False))

    t3: List[Tuple[int, float, np.ndarray]] = [(op_idx, detR, F.copy()) for (op_idx, detR, F) in raw]
    tf: List[np.ndarray] = []

    def _C_single_for_frame(F_rows: np.ndarray) -> float | None:
        if hkl is None or Em is None:
            return None
        return float(contrast_factor_single_slip_frame(cf_input, hkl, F_rows=F_rows, Em=Em))

    print("\n--- PRUNING (While/DeleteCases) ---")

    step = 0
    while len(t3) > 0:
        step += 1
        op_seed, det_seed, kF = t3[0]
        tf.append(kF)

        print("\n------------------------------------------------------------")
        print(f"[seed #{step:02d}] taking first remaining frame: op#={op_seed:02d} det={det_seed:+.6f}")
        print("kF =\n", np.array_str(kF, precision=precision, suppress_small=False))

        kept: List[Tuple[int, float, np.ndarray]] = []
        removed: List[Tuple[int, float, np.ndarray, str]] = []

        for (op_idx, detR, xF) in t3:
            reason = _equiv_delete_reason(xF, kF, tol=tol)
            if reason is None:
                kept.append((op_idx, detR, xF))
            else:
                removed.append((op_idx, detR, xF, reason))

        seedC = _C_single_for_frame(kF)

        print(f"[seed #{step:02d}] DeleteCases removed {len(removed)} frames (including the seed itself).")
        if seedC is not None:
            print(f"  seed C_single = {seedC:.15g}")

        for (op_idx, detR, xF, reason) in removed:
            c = _C_single_for_frame(xF)
            if seedC is None or c is None:
                print(f"  - removed op#={op_idx:02d} det={detR:+.6f} because {reason}")
            else:
                print(
                    f"  - removed op#={op_idx:02d} det={detR:+.6f} because {reason}"
                    f"   C_single={c:.15g}   ΔC={c-seedC:+.3e}"
                )

        t3 = kept

    print("\n============================================================")
    print(f"[EQUIV PRUNING TRACE] N_selected = {len(tf)}")
    print("============================================================\n")
    return tf


def match_selected_frames_to_ops(
    cf_input: ContrastFactorInput,
    selected_frames: List[np.ndarray],
    include_improper: bool = True,
    tol: float = 1e-6,
) -> List[Tuple[int, float, int]]:
    """
    For each selected frame, find the raw operator index that produced it.
    Returns list of (selected_index_1based, detR, op_index_1based).
    If multiple match, returns the first match (Mathematica order relevance).
    If none match, op_index = -1.
    """
    raw = all_slip_frames_with_ops48(cf_input, include_improper=include_improper)

    out: List[Tuple[int, float, int]] = []
    for s_idx, Fsel in enumerate(selected_frames, start=1):
        found_op = -1
        found_det = 0.0
        for (op_idx, detR, Fraw) in raw:
            if frameEqQ(Fraw, Fsel, tol=tol):
                found_op = op_idx
                found_det = detR
                break
        out.append((s_idx, found_det, found_op))
    return out

# -----------------------------------------------------------------------------
# SWITCH: choose which EquivalentSlipSystem you want to use everywhere
#   False -> FULL  (recommended; ~48 for cubic m3m with improper)
#   True  -> PRUNED (legacy Mathematica DeleteCases selection; typically 12)
# -----------------------------------------------------------------------------
USE_PRUNED_EQUIV = False

def EquivalentSlipSystem(
    cf_input: ContrastFactorInput,
    include_improper: bool = True,
) -> List[np.ndarray]:
    return (
        EquivalentSlipSystemPruned(cf_input, include_improper=include_improper)
        if USE_PRUNED_EQUIV
        else EquivalentSlipSystemFull(cf_input, include_improper=include_improper)
    )


def dump_full_ops_frames_and_powder_debug(
    cf_input: ContrastFactorInput,
    hkl: Sequence[int] | Sequence[float],
    include_improper: bool = True,
    Em: np.ndarray | None = None,
    precision: int = 12,
    *,
    show_ops_raw: bool = True,
    show_ops_after_operator_in_e: bool = True,
) -> float:
    """
    Generic debug (works for cubic/tetragonal/hex/etc):
      - prints ops from build_point_group_rotations_from_laue (raw)
      - prints ops from OperatorInE (sorted + Identity prepended + dedup)
      - prints ALL operator-applied slip frames (NO pruning)
      - prints tau, v6, G, C_single for each frame
      - prints Gbar and Cbar
    """
    if Em is None:
        Em = elastic_E_matrix(cf_input)
    Em = np.asarray(Em, float).reshape(6, 6)

    hkl_vec = tuple(int(x) for x in np.rint(np.asarray(hkl, float).reshape(3)))

    print("\n============================================================")
    print("[FULL SYMMETRY+POWDER DEBUG]")
    print(f"lattice={cf_input.cell.lattice}  sg={cf_input.space_group}  phi={float(cf_input.slip_system.phi_deg):g} deg")
    print(f"hkl={hkl_vec}  include_improper={include_improper}")
    print("============================================================\n")

    # --- (A) raw ops (as produced by your group generator) ---
    if show_ops_raw:
        ops_raw = build_point_group_rotations_from_laue(cf_input, include_improper=include_improper)
        print(f"[ops_raw] N = {len(ops_raw)} (from build_point_group_rotations_from_laue)")
        for i, R in enumerate(ops_raw, start=1):
            Rm = np.asarray(R, float).reshape(3, 3)
            detR = float(np.linalg.det(Rm))
            print(f"\n[ops_raw #{i:03d}] det={detR:+.6f}")
            print(np.array_str(Rm, precision=precision, suppress_small=False))

    # --- (B) ops as used by averaging (OperatorInE order) ---
    ops = OperatorInE(cf_input, include_improper=include_improper)
    if show_ops_after_operator_in_e:
        print(f"\n[OperatorInE] N = {len(ops)} (after sort+prepend(I)+dedup)")
        for i, R in enumerate(ops, start=1):
            Rm = np.asarray(R, float).reshape(3, 3)
            detR = float(np.linalg.det(Rm))
            print(f"\n[op #{i:03d}] det={detR:+.6f}")
            print(np.array_str(Rm, precision=precision, suppress_small=False))

    # --- (C) slip base frame ---
    t1 = SlipSystem(cf_input)  # rows = e1,e2,e3
    print("\n[SlipSystem] base frame rows (e1,e2,e3):")
    print(np.array_str(np.asarray(t1, float), precision=precision, suppress_small=False))

    # --- (D) build all operator-applied frames (NO pruning) ---
    frames = _build_operator_applied_frames(cf_input, include_improper=include_improper, chop_tol=1e-6)
    print(f"\n[frames] N = {len(frames)} (operator-applied frames, no pruning)")

    Gs: list[np.ndarray] = []
    Cs: list[float] = []

    for idx, F in enumerate(frames, start=1):
        tau = gInSlipSystem(cf_input, hkl_vec, F)   # unit g in slip frame
        v6 = v6_mathematica(tau)                    # {11,22,33,23,13,12}
        G = _G_from_tau_mathematica(tau)            # 6x6
        c = float(np.sum(G * Em))

        print(f"\n-- frame #{idx:03d} --")
        print("F_rows =\n", np.array_str(F, precision=precision, suppress_small=False))
        print("tau    =", np.array_str(tau, precision=precision, suppress_small=False))
        print("v6     =", np.array_str(v6, precision=precision, suppress_small=False))
        print("C_single =", f"{c:.15g}")

        Gs.append(G)
        Cs.append(c)

    Gbar = np.mean(Gs, axis=0) if Gs else np.zeros((6, 6), float)
    Cbar_from_Gbar = float(np.sum(Gbar * Em))
    Cbar_mean = float(np.mean(Cs)) if Cs else 0.0

    print("\n-- Em (6x6) --\n", np.array_str(Em, precision=precision, suppress_small=False))
    print("\n-- Gbar (6x6) --\n", np.array_str(Gbar, precision=precision, suppress_small=False))
    print("\n-- Cbar --")
    print("  from Gbar·Em =", f"{Cbar_from_Gbar:.15g}")
    print("  mean(C_single) =", f"{Cbar_mean:.15g}")
    print("============================================================\n")

    return Cbar_from_Gbar


def delta_monoclinic_unique_b(h: int, k: int, l: int,
                              a: float, b: float, c: float, beta_deg: float) -> float:
    """
    Monoclinic (unique axis b, alpha=gamma=90): delta as in the paper screenshot.
    Returns dimensionless delta (NOT degrees).
    """
    beta = np.deg2rad(beta_deg)

    ab = a / b
    ac = a / c

    s = np.sin(beta)
    if abs(s) < 1e-15:
        raise ValueError("sin(beta) ~ 0; invalid beta for monoclinic cell")

    csc = 1.0 / s
    cot = np.cos(beta) / s

    h2, k2, l2 = h*h, k*k, l*l

    inside = (h2 + (ab**2)*k2 + (ac**2)*l2) * (csc**2) \
             - (ab**2)*k2 * (cot**2) \
             - 2.0*ac*h*l * cot * csc

    if inside <= 0.0:
        raise ValueError(f"delta sqrt argument <= 0 for hkl=({h},{k},{l}): {inside}")

    return (ab * k) / np.sqrt(inside)