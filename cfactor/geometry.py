# cfactor/geometry.py
from __future__ import annotations

import math
import numpy as np

from .models import ContrastFactorInput, CellParam


def _deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def _get_cf_cache(cf_input: ContrastFactorInput) -> dict:
    cache = getattr(cf_input, "_cfactor_cache", None)
    if cache is None:
        cache = {}
        setattr(cf_input, "_cfactor_cache", cache)
    return cache


def _direct_basis_from_cell(cell: CellParam) -> np.ndarray:
    a, b, c = cell.a, cell.b, cell.c
    alpha = _deg2rad(cell.alpha)
    beta = _deg2rad(cell.beta)
    gamma = _deg2rad(cell.gamma)

    a1 = np.array([a, 0.0, 0.0], dtype=float)
    a2 = np.array([b * math.cos(gamma), b * math.sin(gamma), 0.0], dtype=float)

    cx = c * math.cos(beta)
    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) < 1e-12:
        raise ValueError("gamma too close to 0 or 180 degrees")

    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_gamma
    cz_sq = c * c - cx * cx - cy * cy
    if cz_sq < -1e-10:
        raise ValueError(f"c_z^2 negative ({cz_sq}). Check cell parameters.")
    cz = math.sqrt(max(cz_sq, 0.0))
    a3 = np.array([cx, cy, cz], dtype=float)

    return np.column_stack((a1, a2, a3))


def direct_basis(cf_input: ContrastFactorInput) -> np.ndarray:
    """
    Cached direct basis A (columns: a,b,c in Cartesian/orthonormal frame).
    """
    cache = _get_cf_cache(cf_input)
    if "A_dir" in cache:
        return cache["A_dir"]
    A = _direct_basis_from_cell(cf_input.cell)
    cache["A_dir"] = A
    return A


def reciprocal_basis(cf_input: ContrastFactorInput) -> np.ndarray:
    """
    Cached reciprocal basis B = inv(A)^T.
    """
    cache = _get_cf_cache(cf_input)
    if "B_rec" in cache:
        return cache["B_rec"]
    A = direct_basis(cf_input)
    B = np.linalg.inv(A).T
    cache["B_rec"] = B
    return B


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("zero vector in normalization")
    return v / n


def _miller_direction_to_cartesian(uvw: np.ndarray, A: np.ndarray) -> np.ndarray:
    return A @ uvw


def _miller_plane_normal_to_cartesian(hkl: np.ndarray, B: np.ndarray) -> np.ndarray:
    return B @ hkl


def to_crystal_components(v_ortho: np.ndarray, cell: CellParam) -> np.ndarray:
    A = _direct_basis_from_cell(cell)
    return np.linalg.inv(A) @ v_ortho


def build_slip_frame(cf_input: ContrastFactorInput):
    """
    EXACT Mathematica SlipSystem[] convention:

      e2 = unit( RepToOrthoN[glideplane] )  -> plane normal in orthonormal E
      e3 = RotClockMatrix(e2, phi) . unit(CryToOrthoN[burgervector])  -> CLOCKWISE rotation about e2
      e1 = Cross(e2, e3)

    Returns rows-vectors (e1,e2,e3) as numpy arrays.
    NOTE: This frame is NOT forced to be orthonormal (same as Mathematica).
    """
    ss = cf_input.slip_system
    A = direct_basis(cf_input)
    B = reciprocal_basis(cf_input)

    # e2 = plane normal (RepToOrthoN[glideplane,cp])
    hkl = np.asarray(ss.plane_hkl, dtype=float)
    e2 = _normalize(_miller_plane_normal_to_cartesian(hkl, B))
    
    # ---------------------------
    # NEW: canonicalize e2 sign (Mathematica-like)
    # Choose the largest-magnitude component to be positive.
    # This removes the arbitrary ± ambiguity of plane normals.
    # ---------------------------
    imax = int(np.argmax(np.abs(e2)))
    if e2[imax] < 0.0:
        e2 = -e2
    
    # burgers direction in orthonormal E (CryToOrthoN[burgervector,cp])
    uvw = np.asarray(ss.burgers_uvw, dtype=float)
    b_hat = _normalize(_miller_direction_to_cartesian(uvw, A))
    
    # CLOCKWISE rotation about axis e2 by angle phi  => Rodrigues with angle = -phi
    phi = math.radians(float(ss.phi_deg))
    ang = -phi  # Mathematica "clockwise"
    
    k = e2
    v = b_hat
    e3 = (
        v * math.cos(ang)
        + np.cross(k, v) * math.sin(ang)
        + k * (np.dot(k, v)) * (1.0 - math.cos(ang))
    )
    
    # e1 = Cross(e2,e3) (must be recomputed after canonicalizing e2)
    e1 = np.cross(e2, e3)

    # Chop small noise (Mathematica Chop)
    for vec in (e1, e2, e3):
        vec[np.abs(vec) < 1e-12] = 0.0

    return e1, e2, e3









# Keep old names for backward compatibility (if other files import them)
def _reciprocal_basis_from_cell(cell: CellParam) -> np.ndarray:
    A = _direct_basis_from_cell(cell)
    return np.linalg.inv(A).T
