# CFactor/gbuilder.py
from __future__ import annotations

import numpy as np
from .models import ContrastFactorInput
from .geometry import _reciprocal_basis_from_cell, build_slip_frame

def geometric_tensor(cf_input: ContrastFactorInput, hkl_array: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Costruisce il tensore geometrico G(hkl) (6x6) in notazione Voigt 'Mathematica':
    indice: [11, 22, 33, 23, 13, 12] -> [0,1,2,3,4,5].
    """
    # Base reciproca per il piano hkl
    B = _reciprocal_basis_from_cell(cf_input.cell)
    n_cart = B @ np.asarray(hkl_array, dtype=float)

    # Slip frame
    e1, e2, e3 = build_slip_frame(cf_input)
    P = np.column_stack((e1, e2, e3))  # 3x3

    # Componenti della normale nello slip frame
    # Mathematica: tau = Inverse[Transpose[SlipSystem]] . g_cart
    # Use solve for numerical stability
    n_frame = np.linalg.solve(P.T, n_cart)
    nf = np.linalg.norm(n_frame)
    if nf == 0.0:
        if verbose:
            print("[G] nf=0: returning zeros")
        return np.zeros((6, 6), dtype=float)
    n1, n2, n3 = (n_frame / nf)

    # Vettore Voigt (Mathematica: 11,22,33,23,13,12)
    v = np.array([
        n1 * n1,     # 11
        n2 * n2,     # 22
        n3 * n3,     # 33
        n2 * n3,     # 23  (NO 2x)
        n1 * n3,     # 13  (NO 2x)
        n1 * n2      # 12  (NO 2x)
    ], dtype=float)
    
    G = np.outer(v, v)

    if verbose:
        print(f"[B] Reciprocal basis (columns):")
        print("  " + "  ".join(f"{x: .6f}" for x in B[:,0]))
        print("  " + "  ".join(f"{x: .6f}" for x in B[:,1]))
        print("  " + "  ".join(f"{x: .6f}" for x in B[:,2]))
        print(f"[hkl] = {tuple(int(x) for x in hkl_array)}")
        print("[n_cart] plane normal (cartesian):", np.array2string(n_cart, formatter={'float_kind':lambda x: f'{x: .6f}'}))
        print("[P] slip frame columns (e1,e2,e3):")
        print("  " + "  ".join(f"{x: .6f}" for x in P[:,0]))
        print("  " + "  ".join(f"{x: .6f}" for x in P[:,1]))
        print("  " + "  ".join(f"{x: .6f}" for x in P[:,2]))
        print("[n_frame] components in slip frame:", np.array2string(n_frame, formatter={'float_kind':lambda x: f'{x: .6f}'}))
        print("[v] Voigt vector (11,22,33,23,13,12):", np.array2string(v, formatter={'float_kind':lambda x: f'{x: .6f}'}))

    return G

# alias per compatibilità con main.py
build_geometric_tensor = geometric_tensor