# cfactor/main.py
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import sys
import argparse
import numpy.linalg as npl

from typing import TextIO
from contextlib import contextmanager

from cfactor.io_input import load_contrast_input
from cfactor.models import ContrastFactorInput
from cfactor.elasticity import elastic_E_matrix_debug

from cfactor.contrast import (
    contrast_factor_for_hkl,
    single_crystal_orbit_over_equiv_hkls,
    scattering_phi_deg,
    dump_powder_average_debug,
    dump_full_48_frames_debug_m3m,
    dump_selected_12_frames_debug_m3m,
    dump_equiv_pruning_trace,
    match_selected_frames_to_ops,
    EquivalentSlipSystem,
    elastic_E_matrix,
)

from cfactor.InvariantCoefficients import run_fit_multi_phi

import cfactor.elasticity as _el
import cfactor.contrast as _ct
import cfactor.geometry as _geo

import cfactor.contrast as contrast_mod
from cfactor.geometry import direct_basis, reciprocal_basis

from cfactor import __version__ as CF_VERSION

# =============================================================================
# JSON validation + .error writer (ENGLISH ONLY)
# =============================================================================

class JsonInputFormatError(ValueError):
    """Input JSON structure/type error (not a runtime code failure)."""


def _write_input_error_file(out_dir: Path, in_path: Path, exc: Exception) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    err_path = (out_dir / f"{in_path.stem}.error").resolve()

    details = f"{type(exc).__name__}: {exc}"
    err_path.write_text(
        "Input JSON error.\n"
        f"File: {str(in_path)}\n\n"
        f"Details:\n{details}\n",
        encoding="utf-8",
    )
    return err_path


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _require_keys(d: dict, keys: list[str], where: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise JsonInputFormatError(f"{where}: missing keys: {missing}")


def _validate_hkl_list(name: str, arr, where: str) -> None:
    if arr is None:
        return
    if not isinstance(arr, list):
        raise JsonInputFormatError(f"{where}.{name} must be a list, got: {type(arr).__name__}")
    for i, hkl in enumerate(arr):
        if not (isinstance(hkl, list) and len(hkl) in (3, 4)):
            raise JsonInputFormatError(f"{where}.{name}[{i}] must be a list of 3 or 4 ints, got: {hkl!r}")
        for j, v in enumerate(hkl):
            if not isinstance(v, int) or isinstance(v, bool):
                raise JsonInputFormatError(
                    f"{where}.{name}[{i}][{j}] must be int, got: {type(v).__name__} ({v!r})"
                )


def _is_close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def _require_cell_keys_for_lattice(cell: dict, lattice: str) -> None:
    lat = str(lattice).lower()

    # Independent parameter requirements (table-driven)
    if lat == "cubic":
        _require_keys(cell, ["a"], where="root.cell")
        return
    if lat in ("hexagonal", "tetragonal", "trigonal"):
        _require_keys(cell, ["a", "c"], where="root.cell")
        return
    if lat == "orthorhombic":
        _require_keys(cell, ["a", "b", "c"], where="root.cell")
        return
    if lat == "monoclinic":
        _require_keys(cell, ["a", "b", "c", "beta"], where="root.cell")
        return
    if lat == "triclinic":
        _require_keys(cell, ["a", "b", "c", "alpha", "beta", "gamma"], where="root.cell")
        return

    raise JsonInputFormatError(f"root.cell.lattice invalid: {lattice!r}")


def _validate_cell_values(cell: dict) -> None:
    # Lengths must be > 0 when present
    for k in ("a", "b", "c"):
        if k in cell and cell[k] is not None:
            v = float(cell[k])
            if v <= 0.0:
                raise JsonInputFormatError(f"root.cell.{k} must be > 0, got: {v}")

    # Angles must be within (0, 180)
    for k in ("alpha", "beta", "gamma"):
        if k in cell and cell[k] is not None:
            v = float(cell[k])
            if not (0.0 < v < 180.0):
                raise JsonInputFormatError(f"root.cell.{k} must be in (0,180), got: {v}")


def _validate_elastic_nonzero(elastic: dict) -> None:
    # If full C is provided, ensure it isn't all zeros
    if "C" in elastic:
        C = elastic["C"]
        flat = [float(x) for row in C for x in row]
        if all(abs(v) == 0.0 for v in flat):
            raise JsonInputFormatError("root.elastic.C must not be all zeros")
        return

    # If Cij provided, ensure at least one nonzero and no NaN/inf
    cij_keys = [k for k in elastic.keys() if isinstance(k, str) and k.strip().upper().startswith("C")]
    vals = [float(elastic[k]) for k in cij_keys] if cij_keys else []
    if not vals:
        raise JsonInputFormatError("root.elastic must contain 'C' (6x6) or at least one 'Cij' coefficient (e.g. C11)")
    if all(abs(v) == 0.0 for v in vals):
        raise JsonInputFormatError("root.elastic must not have all Cij = 0")


def validate_contrast_factors_json(data: object) -> None:
    if not isinstance(data, dict):
        raise JsonInputFormatError(f"Root JSON must be an object/dict, got: {type(data).__name__}")

    _require_keys(data, ["cell", "elastic", "slip_systems"], where="root")

    if "space_group" in data and data["space_group"] is not None:
        if not isinstance(data["space_group"], int) or isinstance(data["space_group"], bool):
            raise JsonInputFormatError("root.space_group must be an int (1..230)")
        if not (1 <= int(data["space_group"]) <= 230):
            raise JsonInputFormatError(f"root.space_group out of range (1..230): {data['space_group']}")

    cell = data["cell"]
    if not isinstance(cell, dict):
        raise JsonInputFormatError(f"root.cell must be an object/dict, got: {type(cell).__name__}")

    # lattice name (if present) should be valid
    lattice = None
    if "lattice" in cell and cell["lattice"] is not None:
        if not isinstance(cell["lattice"], str):
            raise JsonInputFormatError("root.cell.lattice must be a string")
        lattice = cell["lattice"].lower()
        allowed = {"cubic", "hexagonal", "tetragonal", "trigonal", "orthorhombic", "monoclinic", "triclinic"}
        if lattice not in allowed:
            raise JsonInputFormatError(f"root.cell.lattice invalid: {cell['lattice']!r}. Expected: {sorted(allowed)}")

    # If lattice is missing but space_group exists, we still can enforce required keys.
    if lattice is None:
        sg = data.get("space_group", None)
        if sg is not None:
            from cfactor.io_input import infer_lattice_from_space_group
            lattice = infer_lattice_from_space_group(int(sg))

    # Always require at least "a" (kept from your code)
    _require_keys(cell, ["a"], where="root.cell")

    # Enforce table-driven required keys if we know the lattice/system
    if lattice is not None:
        _require_cell_keys_for_lattice(cell, lattice)

    # Existing numeric type checks
    for k in ("a", "b", "c", "alpha", "beta", "gamma"):
        if k in cell and cell[k] is not None and not _is_number(cell[k]):
            raise JsonInputFormatError(f"root.cell.{k} must be numeric, got: {type(cell[k]).__name__}")

    # New numeric value checks (zero / range)
    _validate_cell_values(cell)

    elastic = data["elastic"]
    if not isinstance(elastic, dict):
        raise JsonInputFormatError(f"root.elastic must be an object/dict, got: {type(elastic).__name__}")

    if "C" in elastic:
        C = elastic["C"]
        if not (isinstance(C, list) and len(C) == 6 and all(isinstance(r, list) and len(r) == 6 for r in C)):
            raise JsonInputFormatError("root.elastic.C must be a 6x6 list")
        for i in range(6):
            for j in range(6):
                if not _is_number(C[i][j]):
                    raise JsonInputFormatError(f"root.elastic.C[{i}][{j}] must be numeric, got: {C[i][j]!r}")
    else:
        cij_keys = [k for k in elastic.keys() if isinstance(k, str) and k.strip().upper().startswith("C")]
        if not cij_keys:
            raise JsonInputFormatError("root.elastic must contain 'C' (6x6) or at least one 'Cij' coefficient (e.g. C11)")
        for k in cij_keys:
            v = elastic[k]
            if not _is_number(v):
                raise JsonInputFormatError(f"root.elastic.{k} must be numeric, got: {type(v).__name__} ({v!r})")

    # IMPORTANT: enforce non-zero elastic inputs early (prevents singular matrix later)
    _validate_elastic_nonzero(elastic)

    slip_systems = data["slip_systems"]
    if not isinstance(slip_systems, list) or len(slip_systems) == 0:
        raise JsonInputFormatError("root.slip_systems must be a non-empty list")

    for i, ss in enumerate(slip_systems):
        if not isinstance(ss, dict):
            raise JsonInputFormatError(f"root.slip_systems[{i}] must be an object/dict")
        _require_keys(ss, ["plane_hkl", "burgers_uvw", "phi_deg"], where=f"root.slip_systems[{i}]")

        plane = ss["plane_hkl"]
        burgers = ss["burgers_uvw"]
        phi = ss["phi_deg"]

        if not (isinstance(plane, list) and len(plane) in (3, 4) and all(isinstance(x, int) and not isinstance(x, bool) for x in plane)):
            raise JsonInputFormatError(f"root.slip_systems[{i}].plane_hkl must be a list of 3 or 4 ints")
        if not (isinstance(burgers, list) and len(burgers) in (3, 4) and all(isinstance(x, int) and not isinstance(x, bool) for x in burgers)):
            raise JsonInputFormatError(f"root.slip_systems[{i}].burgers_uvw must be a list of 3 or 4 ints")

        if isinstance(phi, list):
            if len(phi) == 0:
                raise JsonInputFormatError(f"root.slip_systems[{i}].phi_deg must not be an empty list")
            for j, pv in enumerate(phi):
                if not _is_number(pv):
                    raise JsonInputFormatError(f"root.slip_systems[{i}].phi_deg[{j}] must be numeric")
        else:
            if not _is_number(phi):
                raise JsonInputFormatError(f"root.slip_systems[{i}].phi_deg must be a number or a list of numbers")

    _validate_hkl_list("hkls", data.get("hkls"), where="root")


# =============================================================================
# Helpers
# =============================================================================

def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _mono_plane_basis_b_u(cf_input: ContrastFactorInput) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (b_hat, u_hat, n_hat) in Cartesian.
    u_hat is chosen ~ (a_hat + c_hat) projected ⟂ b_hat.
    n_hat = unit(b_hat × u_hat), normal to the plotting plane.
    """
    A = direct_basis(cf_input)  # columns are a,b,c in Cartesian
    a_hat = _unit(A[:, 0])
    b_hat = _unit(A[:, 1])
    c_hat = _unit(A[:, 2])

    u0 = a_hat + c_hat
    u0 = u0 - np.dot(u0, b_hat) * b_hat
    u_hat = _unit(u0)

    n_hat = _unit(np.cross(b_hat, u_hat))
    return b_hat, u_hat, n_hat


def _delta_from_ghat_b(ghat: np.ndarray, b_hat: np.ndarray) -> float:
    """delta = angle between ghat and b axis, folded to [0,90] via abs(cos)."""
    cosd = float(np.clip(abs(np.dot(_unit(ghat), b_hat)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosd)))


def _make_monoclinic_scan_directions(
    cf_input: ContrastFactorInput,
    n_points: int = 181,   # 0..90 step 0.5 deg
) -> list[tuple[float, np.ndarray]]:
    """
    Smooth 1-parameter set of unit g-directions (Cartesian) for monoclinic unique-b:
      ghat(delta) = cos(delta) * b_hat + sin(delta) * u_hat
    where u_hat lies in plane ⟂ b_hat, chosen from (a_hat + c_hat) then projected ⟂ b_hat.
    Returns list of (delta_deg, ghat_cart).
    """
    A = direct_basis(cf_input)  # columns are a,b,c in Cartesian
    a_hat = _unit(A[:, 0])
    b_hat = _unit(A[:, 1])
    c_hat = _unit(A[:, 2])

    u0 = a_hat + c_hat
    u0 = u0 - np.dot(u0, b_hat) * b_hat
    u_hat = _unit(u0)

    out: list[tuple[float, np.ndarray]] = []
    for delta_deg in np.linspace(0.0, 90.0, int(n_points)):
        d = np.deg2rad(delta_deg)
        ghat = np.cos(d) * b_hat + np.sin(d) * u_hat
        out.append((float(delta_deg), _unit(ghat)))
    return out


def _contrast_along_direction_scan(
    cf_input: ContrastFactorInput,
    Em_phi: np.ndarray,
    n_points: int = 181,
) -> list[tuple[float, float]]:
    """
    Evaluate C along the monoclinic smooth scan.
    Uses float hkl_real = inv(B) @ ghat where B is reciprocal basis (cart <- hkl).
    Returns list of (delta_deg, C).
    """
    B = reciprocal_basis(cf_input)           # cart = B @ hkl
    Binv = np.linalg.inv(np.asarray(B, float))

    scan = _make_monoclinic_scan_directions(cf_input, n_points=n_points)

    out: list[tuple[float, float]] = []
    for delta_deg, ghat in scan:
        hkl_real = Binv @ ghat   # float 3-vector
        C = contrast_factor_for_hkl(
            cf_input,
            hkl_real,  # floats are OK in your code path
            include_improper=True,
            Em=Em_phi,
        )
        out.append((float(delta_deg), float(C)))
    return out


# =============================================================================
# Debug / flags
# =============================================================================

VERBOSE_PER_HKL = False
POWDER_DEBUG_ALL_HKLS = False
POWDER_DEBUG_ONLY_PHI = None

DEBUG_M3M_FULL48 = False
DEBUG_M3M_SELECTED12 = False
DEBUG_M3M_PRUNING_TRACE = False
DEBUG_M3M_ONLY_FIRST_HKL = False
DEBUG_M3M_ONLY_PHI = None

PRINT_CINSS_DEBUG = False
contrast_mod.USE_PRUNED_EQUIV = False
PRINT_ELASTIC_DEBUG = False

DEBUG_FULL_SYMMETRY_ANY = False
DEBUG_ANY_ONLY_PHI = 0.0
DEBUG_ANY_ONLY_FIRST_HKL = False

DEBUG_FEM_INPUTS = False
DEBUG_FEM_ONLY_PHI = 90.0   # set None for all phis


def _g6_voigt_math(tau: np.ndarray) -> np.ndarray:
    """
    Mathematica Voigt order: {11,22,33,23,13,12}
    v = {t1^2, t2^2, t3^2, t2*t3, t1*t3, t1*t2}
    """
    t1, t2, t3 = (float(tau[0]), float(tau[1]), float(tau[2]))
    return np.array([t1*t1, t2*t2, t3*t3, t2*t3, t1*t3, t1*t2], dtype=float)


class _TeeStdout:
    def __init__(self, console: TextIO, file: TextIO) -> None:
        self._console = console
        self._file = file

    def write(self, s: str) -> int:
        n1 = self._console.write(s)
        self._console.flush()
        self._file.write(s)
        self._file.flush()
        return n1

    def flush(self) -> None:
        self._console.flush()
        self._file.flush()


@contextmanager
def tee_stdout_to_file(log_path: Path):
    """
    Tee everything printed to stdout to both console and a log file.
    Also mirrors stderr to the same tee so tracebacks end up in the log.
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as flog:
        tee = _TeeStdout(original_stdout, flog)
        sys.stdout = tee
        sys.stderr = tee
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def _resolve_input_path() -> Path:
    """
    Default input discovery (GUI/.exe mode):
      - looks for 'contrast_factors.json' next to this module
      - or inside a 'Contrast_Factor' subfolder (legacy layout)

    NOTE: JSON only.
    """
    here = Path(__file__).resolve().parent
    for p in (
        here / "contrast_factors.json",
        here / "Contrast_Factor" / "contrast_factors.json",
        Path("Contrast_Factor/contrast_factors.json").resolve(),
        Path("contrast_factors.json").resolve(),
    ):
        if p.exists():
            return p
    raise FileNotFoundError(
        "Unable to find 'contrast_factors.json'. "
        "Place it next to the executable (or inside 'Contrast_Factor/') "
        "or pass an explicit input path: discocf <file>.json"
    )


def _print_cell_and_elastic(cf: ContrastFactorInput) -> None:
    cell = cf.cell
    print("\n== Cell parameters ==")
    print(f"  Lattice = {cell.lattice}")
    print(f"  a = {cell.a:.6f}, b = {cell.b:.6f}, c = {cell.c:.6f} Å")
    print(
        f"  alpha = {cell.alpha:.6f}°, "
        f"beta = {cell.beta:.6f}°, "
        f"gamma = {cell.gamma:.6f}°"
    )

    elastic = cf.elastic
    print("\n== Elastic stiffness C (GPa), Voigt 6x6 ==")
    print(np.array2string(np.asarray(elastic.C, dtype=float), precision=6, suppress_small=True))


ZERO_CUTOFF = 1e-6


def _zero_small_coeffs(coeffs: dict[str, float], tol: float = ZERO_CUTOFF) -> dict[str, float]:
    return {k: (0.0 if abs(float(v)) < tol else float(v)) for k, v in coeffs.items()}


def _print_fit_results(
    title: str,
    coeffs: dict[str, float],
    metrics: dict[str, float],
    is_cubic: bool,
) -> None:
    print(f"\n=== {title} ===")
    if "PHI" in title.upper():
        print("phi =", title.split("=", 1)[-1].strip())

    if is_cubic:
        if "E1" not in coeffs or "E2" not in coeffs:
            raise KeyError("Cubic output requires E1 and E2 in coeffs.")

        E1 = float(coeffs["E1"])
        E2 = float(coeffs["E2"])

        A = E1
        B = float((E2 - E1) * 2.0)
        slope_2E2 = float(E2 * 2.0)

        print(f"E1 = {E1: .12g}")
        print(f"E2 = {E2: .12g}")
        print(f"A = E1 = {A: .12g}")
        print(f"B = (E2 - E1)*2 = {B: .12g}")
        print(f"2*E2 (coefficiente davanti a H) = {slope_2E2: .12g}")
        print("metrics:", metrics)
        return

    for k in sorted(coeffs.keys(), key=lambda s: int(s[1:])):
        print(f"{k:>3s} = {coeffs[k]: .12g}")
    print("metrics:", metrics)


def _format_hkl(hkl: tuple[int, int, int]) -> str:
    return f"[{hkl[0]} {hkl[1]} {hkl[2]}]"


def _H_cubic(h: int, k: int, l: int) -> float:
    h2 = float(h * h)
    k2 = float(k * k)
    l2 = float(l * l)

    den = (h2 + k2 + l2) ** 2
    if den == 0.0:
        return 0.0

    num = (h2 * k2) + (k2 * l2) + (l2 * h2)
    return float(num / den)


def _write_single_crystal_tsv(
    path: Path,
    cf_input: ContrastFactorInput,
    hkls: list[tuple[int, int, int]],
    include_improper: bool,
) -> None:
    """
    Writes a Mathematica-style single-crystal report:
      - base header [h k l]
      - then each equivalent [h' k' l'] and its C_single value.

    IMPORTANT:
    We do NOT recompute tau6 @ Em @ tau6 here.
    We write the C_single that comes from single_crystal_orbit_over_equiv_hkls(),
    which is the same logic you debugged in the 48-frame dump.
    """
    with path.open("w", encoding="utf-8") as f:
        f.write(
            "# single-crystal contrast factors (Mathematica-style CFSingle debug), "
            f"phi={float(cf_input.slip_system.phi_deg):.6g} deg\n"
        )

        for base_hkl in hkls:
            base_hkl = (int(base_hkl[0]), int(base_hkl[1]), int(base_hkl[2]))
            f.write(_format_hkl(base_hkl) + "\n")

            orbit = single_crystal_orbit_over_equiv_hkls(
                cf_input,
                base_hkl,
                include_improper=include_improper,
                Em=None,  # let the function compute/use the correct Em internally (or ignore if it doesn't need it)
            )

            for heq, c_val in orbit:
                heq = (int(heq[0]), int(heq[1]), int(heq[2]))
                f.write(f"{_format_hkl(heq)}\t{float(c_val):.12g}\n")

            f.write("\n")


def _should_run_m3m_debug(phi: float, base_hkl_index: int) -> bool:
    if DEBUG_M3M_ONLY_PHI is not None and abs(phi - float(DEBUG_M3M_ONLY_PHI)) > 1e-9:
        return False
    if DEBUG_M3M_ONLY_FIRST_HKL and base_hkl_index != 0:
        return False
    return True


# =============================================================================
# Public API called by disco/cli.py
# =============================================================================

def run(input_path: Path | None = None, output_dir: Path | None = None) -> None:
    """
    Main entry for the CLI.

    - input_path: input JSON path. If None, auto-discover 'contrast_factors.json'
    - output_dir: output directory. If None, defaults to <input_dir>/output_fit
    """
    in_path = Path(input_path).expanduser().resolve() if input_path is not None else _resolve_input_path()
    if in_path.suffix.lower() != ".json":
        raise ValueError(f"Input must be a .json file, got: {in_path.name}")
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_path}")

    in_dir = in_path.parent
    out_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else (in_dir / "output_fit").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        with in_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        validate_contrast_factors_json(data)
    except (OSError, json.JSONDecodeError, JsonInputFormatError, ValueError) as ex:
        err_path = _write_input_error_file(out_dir, in_path, ex)
        print(f"[ERROR] Invalid input JSON. Details written to: {err_path}")
        return

    # NOTE: load_contrast_input can still fail on input issues; make it write .error too
    try:
        inputs = load_contrast_input(in_path)
    except Exception as ex:
        err_path = _write_input_error_file(out_dir, in_path, ex)
        print(f"[ERROR] Invalid input JSON. Details written to: {err_path}")
        return

    hkls: list[tuple[int, int, int]] = [tuple(map(int, h)) for h in data.get("hkls", [])]

    include_improper = True

    # Resolve output directory
    in_dir = in_path.parent
    out_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else (in_dir / "output_fit").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[
        tuple[tuple[int, int, int], tuple[int, int, int]],
        list[ContrastFactorInput],
    ] = {}

    for cf in inputs:
        slip = cf.slip_system
        key = (tuple(slip.plane_hkl), tuple(slip.burgers_uvw))
        groups.setdefault(key, []).append(cf)

    if not groups:
        print("[ERROR] No slip system groups found in input.")
        return

    for (plane_hkl, burgers_uvw), cf_group in groups.items():
        cf_group = sorted(cf_group, key=lambda x: float(x.slip_system.phi_deg))
        cf0 = cf_group[0]

        mat = cf0.material_name
        lat = cf0.cell.lattice
        plane_str = "_".join(str(int(x)) for x in plane_hkl)
        burg_str = "_".join(str(int(x)) for x in burgers_uvw)

        out_name = f"{mat}_{lat}_plane[{plane_str}]_burgers[{burg_str}].txt"
        out_path = (out_dir / out_name).resolve()
        run_log_path = out_path.with_suffix(".run.log")

        with tee_stdout_to_file(run_log_path):
            try:
                print(f"[INFO] Contrast_Factor version: {CF_VERSION}")

                print("[DEBUG] elasticity.py loaded from:", _el.__file__)
                print("[DEBUG] contrast.py loaded from:", _ct.__file__)
                print("[DEBUG] geometry.py loaded from:", _geo.__file__)

                print("\n============================================================")
                print(f"[Input] contrast_factors.json path: {in_path}")
                print("[Input] contrast_factors.json contents (JSON):")
                print(json.dumps(data, indent=2, ensure_ascii=False))
                print("============================================================")

                print("\n=== Slip system group ===")
                print(f"Material: {cf0.material_name}")
                print(f"Space group: {cf0.space_group} ({cf0.space_group_name or ''})")
                print(f"Lattice (inferred/used): {cf0.cell.lattice}")
                print(f"Laue symbol (legacy, optional): {cf0.laue_symbol or ''}")
                print(f"Plane (HKL) = {plane_hkl}, Burgers [UVW] = {burgers_uvw}")

                _print_cell_and_elastic(cf0)

                print("\n== Requested HKL list ==")
                if hkls:
                    print("  ", ", ".join(f"({h},{k},{l})" for (h, k, l) in hkls))
                else:
                    print("  <empty hkls list in input>")

                # ------------------------------------------------------------------
                # Elastic matrix per dislocation character (phi) + smooth curve
                # ------------------------------------------------------------------
                for cf_input in cf_group:
                    phi = float(cf_input.slip_system.phi_deg)

                    try:
                        Em_phi, dbg = elastic_E_matrix_debug(cf_input)
                    except npl.LinAlgError as ex:
                        raise JsonInputFormatError(
                            "Numerical failure: singular matrix while building the elastic compliance from the stiffness "
                            "tensor (elastic constants). This usually means the elastic stiffness matrix is singular or "
                            "non-physical (e.g., one or more required constants are zero/invalid). "
                            "Check elastic constants (e.g. C44>0 for hexagonal) and their units."
                        ) from ex

                    if DEBUG_FEM_INPUTS and (DEBUG_FEM_ONLY_PHI is None or abs(phi - DEBUG_FEM_ONLY_PHI) < 1e-9):
                        fem = dbg.get("FEm_dbg", None)
                        print("\n===============================")
                        print("[DEBUG] FEm inputs (A,D,R,pref,DC2)")
                        print("===============================")

                        if fem is None:
                            print("<missing FEm_dbg>")
                        else:
                            # print the core inputs at high precision
                            np.set_printoptions(precision=18, suppress=False, linewidth=220)

                            for key in ("A", "D", "R"):
                                if key in fem:
                                    print(f"\n[FEm_dbg] {key} =")
                                    print(np.array2string(np.asarray(fem[key]), precision=18, suppress_small=False))

                            if "pref" in fem:
                                print("\n[FEm_dbg] pref =", repr(float(fem["pref"])))

                            if "DC2" in fem:
                                print("\n[FEm_dbg] DC2 =")
                                print(np.array2string(np.asarray(fem["DC2"]), precision=18, suppress_small=False))

                            # optional: hashes so you can compare Windows vs Linux quickly
                            import hashlib
                            def _sha(x):
                                arr = np.ascontiguousarray(np.asarray(x))
                                return hashlib.sha256(arr.view(np.uint8)).hexdigest()

                            for key in ("A", "D", "R", "DC2", "Em"):
                                if key in fem:
                                    print(f"[FEm_dbg] sha256({key}) =", _sha(fem[key]))

                        print("===============================\n")

                    is_mono_unique_b = (
                        str(cf_input.cell.lattice).lower() == "monoclinic"
                        and abs(float(cf_input.cell.alpha) - 90.0) < 1e-9
                        and abs(float(cf_input.cell.gamma) - 90.0) < 1e-9
                    )

                    if is_mono_unique_b:
                        curve = _contrast_along_direction_scan(cf_input, Em_phi, n_points=181)
                        curve_path = out_path.with_suffix(f".curve_phi{phi:.0f}.tsv")
                        print(f"[Monoclinic smooth curve] writing: {curve_path}")
                        with curve_path.open("w", encoding="utf-8") as fcurve:
                            fcurve.write("# delta_deg\tC\n")
                            for ddeg, C in curve:
                                fcurve.write(f"{ddeg:.6f}\t{C:.12g}\n")

                    print("\n------------------------------------------------------------")
                    print(f"--- Dislocation state: phi = {phi:.3f} deg ---")

                    P_cart = np.asarray(dbg.get("P_rows_cart")) if dbg.get("P_rows_cart") is not None else None
                    P_crys = np.asarray(dbg.get("P_rows_crys")) if dbg.get("P_rows_crys") is not None else None
                    CinSS = np.asarray(dbg.get("CinSS")) if dbg.get("CinSS") is not None else None

                    if P_cart is not None:
                        print("\n[P_rows_cart] (orthonormal Cartesian rows = e1,e2,e3):")
                        print(np.array2string(np.asarray(P_cart, dtype=float), precision=6, suppress_small=True))
                    else:
                        print("\n[P_rows_cart] <missing>")

                    if P_crys is not None:
                        print("\n[P_rows_crys] (Mathematica Base->Crystal rows):")
                        print(np.array2string(np.asarray(P_crys, dtype=float), precision=6, suppress_small=True))
                    else:
                        print("\n[P_rows_crys] <missing>")

                    if PRINT_CINSS_DEBUG:
                        cinss_dbg = dbg.get("CinSS_dbg", None)
                        if cinss_dbg is not None:
                            print("\n===============================")
                            print(" CinSS DEBUG (Mathematica Ting method)")
                            print("===============================")

                            def pm(name: str, A: np.ndarray):
                                print(f"\n{name}")
                                print(np.array2string(np.asarray(A, dtype=float), precision=6, suppress_small=True))

                            pm("P = Pmatrix[...] (rows=e1,e2,e3)", cinss_dbg["P_rows_cart"])
                            pm("K1 (First quadrant) = squares", cinss_dbg["K1"])
                            pm("K2 (Second quadrant)", cinss_dbg["K2"])
                            pm("K3 (Third quadrant)", cinss_dbg["K3"])
                            pm("K4 (Fourth quadrant)", cinss_dbg["K4"])
                            pm("K = Kmatrix[...] (6x6)", cinss_dbg["K"])
                            pm("C(crystal) input matrix (6x6)", cinss_dbg["C"])
                            pm("left = K . C", cinss_dbg["left"])
                            pm("right = Transpose[K]", cinss_dbg["right"])
                            pm("raw = (K . C) . Transpose[K] (before Chop)", cinss_dbg["raw"])
                            pm("Chop[raw] (CinSS)", cinss_dbg["CinSS"])

                            print("\n===============================")
                        else:
                            print("\n[CinSS DEBUG] <CinSS_dbg missing>")

                    if CinSS is not None:
                        print("\n[CinSS] stiffness in slip frame (Voigt 6x6):")
                        print(np.array2string(np.asarray(CinSS, dtype=float), precision=6, suppress_small=True))
                    else:
                        print("\n[CinSS] <missing>")

                    b_slip = dbg.get("b_slip", None)
                    print("\n[Stroh] b_slip (Burgers in slip frame, bu):")
                    if b_slip is None:
                        print("<missing b_slip>")
                    else:
                        print(np.array2string(np.asarray(b_slip, dtype=float), precision=6, suppress_small=True))

                    if "p" in dbg:
                        print("\n[Stroh] eigenvalues p (selected):")
                        p = np.asarray(dbg["p"])
                        print(np.array2string(p, precision=10, suppress_small=True))
                    
                    if "method" in dbg:
                        print("\n[Stroh] method:", dbg["method"])
                    if "degenerate" in dbg:
                        print("[Stroh] degenerate:", dbg["degenerate"])
                    if "p_raw" in dbg and dbg["p_raw"] is not None:
                        print("\n[Stroh] p_raw:")
                        print(np.array2string(np.asarray(dbg["p_raw"]), precision=12, suppress_small=True))
                    if "p_clean" in dbg and dbg["p_clean"] is not None:
                        print("\n[Stroh] p_clean:")
                        print(np.array2string(np.asarray(dbg["p_clean"]), precision=12, suppress_small=True))
                    if "p_snap" in dbg and dbg["p_snap"] is not None:
                        print("\n[Stroh] p_snap:")
                        print(np.array2string(np.asarray(dbg["p_snap"]), precision=12, suppress_small=True))
                    
                    if "G_eig_ATL" in dbg and dbg["G_eig_ATL"] is not None:
                        print("\n[Stroh] Gram G = A^T L (eig):")
                        print(np.array2string(np.asarray(dbg["G_eig_ATL"]), precision=12, suppress_small=True))
                    if "G_LLAA_ATL" in dbg and dbg["G_LLAA_ATL"] is not None:
                        print("\n[Stroh] Gram G = A^T L (LL/AA):")
                        print(np.array2string(np.asarray(dbg["G_LLAA_ATL"]), precision=12, suppress_small=True))


                    if "LL_mathematica" in dbg:
                        print("\n[Mathematica] LL matrix:")
                        print(np.array2string(np.asarray(dbg["LL_mathematica"]), precision=10, suppress_small=True))

                    if "AA_mathematica" in dbg:
                        print("\n[Mathematica] A matrix:")
                        print(np.array2string(np.asarray(dbg["AA_mathematica"]), precision=10, suppress_small=True))

                    if "D_solved" in dbg:
                        print("\n[Stroh] D vector (solved from A @ D = -b):")
                        print(np.array2string(np.asarray(dbg["D_solved"]), precision=10, suppress_small=True))

                    if "A_eig" in dbg:
                        print("\n[Stroh] A eigenvectors (columns):")
                        print(np.array2string(np.asarray(dbg["A_eig"]), precision=10, suppress_small=True))

                    if "L_eig" in dbg:
                        print("\n[Stroh] L eigenvectors (columns):")
                        print(np.array2string(np.asarray(dbg["L_eig"]), precision=10, suppress_small=True))

                    if "T" in dbg:
                        T = np.asarray(dbg["T"], dtype=float)
                        print("\n[Stroh] T matrix and det(T):")
                        print(np.array2string(T, precision=6, suppress_small=True))
                        print("det(T) =", f"{float(np.linalg.det(T)):.6e}")

                    print("\n[Em] (6x6 used for the contrast factor at this phi):")
                    print(np.array2string(np.asarray(Em_phi, dtype=float), precision=6, suppress_small=True))

                print(f"\nWriting results to: {out_path}")

                # Collect results: per HKL -> per phi -> (delta_deg, C_powder)
                by_hkl: dict[tuple[int, int, int], dict[float, tuple[float, float]]] = {}

                for cf_input in cf_group:
                    phi = float(cf_input.slip_system.phi_deg)
                    Em_phi, _dbg_phi = elastic_E_matrix_debug(cf_input)

                    is_mono_unique_b = (
                        str(cf_input.cell.lattice).lower() == "monoclinic"
                        and abs(float(cf_input.cell.alpha) - 90.0) < 1e-9
                        and abs(float(cf_input.cell.gamma) - 90.0) < 1e-9
                    )

                    # Discrete curve points (per phi)
                    discrete_curve_pts: list[tuple[float, float, tuple[int, int, int]]] = []
                    if is_mono_unique_b:
                        b_hat, u_hat, n_hat = _mono_plane_basis_b_u(cf_input)
                        Brec = reciprocal_basis(cf_input)  # cart = Brec @ hkl
                        PLANE_TOL = 0.03

                    for idx_h, base in enumerate(hkls):
                        base_tuple = (int(base[0]), int(base[1]), int(base[2]))

                        # Powder contrast factor
                        C_powder = contrast_factor_for_hkl(
                            cf_input,
                            base_tuple,
                            include_improper=include_improper,
                            Em=Em_phi,
                        )

                        # --- Discrete curve selection (option A) ---
                        if is_mono_unique_b:
                            h, k, l = base_tuple
                            g_cart = np.asarray(Brec, float) @ np.array([h, k, l], float)
                            ghat = _unit(g_cart)

                            out_of_plane = abs(float(np.dot(ghat, n_hat)))
                            ub = float(np.dot(ghat, b_hat))
                            uu = float(np.dot(ghat, u_hat))

                            if out_of_plane < PLANE_TOL and ub >= 0.0 and uu >= 0.0:
                                delta_deg = _delta_from_ghat_b(ghat, b_hat)
                                discrete_curve_pts.append((delta_deg, float(C_powder), base_tuple))

                        # Optional full symmetry debug (generic)
                        if DEBUG_FULL_SYMMETRY_ANY:
                            if (DEBUG_ANY_ONLY_PHI is None or abs(phi - DEBUG_ANY_ONLY_PHI) < 1e-9) and (
                                not DEBUG_ANY_ONLY_FIRST_HKL or idx_h == 0
                            ):
                                _ct.dump_full_ops_frames_and_powder_debug(
                                    cf_input,
                                    base_tuple,
                                    include_improper=include_improper,
                                    Em=Em_phi,
                                    precision=12,
                                    show_ops_raw=True,
                                    show_ops_after_operator_in_e=True,
                                )

                        # Powder-average debug
                        if POWDER_DEBUG_ALL_HKLS and (
                            POWDER_DEBUG_ONLY_PHI is None or abs(phi - POWDER_DEBUG_ONLY_PHI) < 1e-9
                        ):
                            dump_powder_average_debug(
                                cf_input,
                                base_tuple,
                                include_improper=include_improper,
                                Em=Em_phi,
                                precision=12,
                            )

                        # Cubic m3m-only debug modes
                        if (DEBUG_M3M_FULL48 or DEBUG_M3M_SELECTED12 or DEBUG_M3M_PRUNING_TRACE) and _should_run_m3m_debug(phi, idx_h):
                            if DEBUG_M3M_FULL48:
                                dump_full_48_frames_debug_m3m(
                                    cf_input,
                                    base_tuple,
                                    include_improper=include_improper,
                                    Em=Em_phi,
                                    precision=12,
                                )

                            if DEBUG_M3M_SELECTED12:
                                dump_selected_12_frames_debug_m3m(
                                    cf_input,
                                    base_tuple,
                                    include_improper=include_improper,
                                    Em=Em_phi,
                                    precision=12,
                                )

                            if DEBUG_M3M_PRUNING_TRACE:
                                print("\n============================================================")
                                print("[DEBUG] Running full pruning trace + op mapping")
                                print(f"[DEBUG] phi={phi:g}  hkl={base_tuple}")
                                print("============================================================\n")

                                selected = dump_equiv_pruning_trace(
                                    cf_input,
                                    include_improper=include_improper,
                                    tol=1e-6,
                                    precision=12,
                                    hkl=base_tuple,
                                    Em=Em_phi,
                                )

                                selected_direct = EquivalentSlipSystem(cf_input, include_improper=include_improper)
                                print(f"[DEBUG] N_selected(trace)  = {len(selected)}")
                                print(f"[DEBUG] N_selected(direct) = {len(selected_direct)}")

                                mapping = match_selected_frames_to_ops(
                                    cf_input,
                                    selected_direct,
                                    include_improper=include_improper,
                                    tol=1e-6,
                                )

                                print("\n--- Selected frame -> (det, op#) mapping ---")
                                for (s_idx, detR, op_idx) in mapping:
                                    if op_idx < 0:
                                        print(f"  s={s_idx:02d}: op NOT FOUND")
                                    else:
                                        print(f"  s={s_idx:02d}: op#={op_idx:02d}  det={detR:+.6f}")

                        # Compute delta for the main output table
                        delta = float(scattering_phi_deg(cf_input, base_tuple, ref_axis="auto"))

                        by_hkl.setdefault(base_tuple, {})[phi] = (delta, float(C_powder))

                        if VERBOSE_PER_HKL:
                            orbit = single_crystal_orbit_over_equiv_hkls(
                                cf_input,
                                base_tuple,
                                include_improper=include_improper,
                                Em=Em_phi,
                            )
                            print("  Single crystal (fixed slip frame), orbit hkls:")
                            for heq, csc in orbit:
                                print(f"    {heq}\t{csc:.12f}")

                    

                phi_values = sorted({float(cf.slip_system.phi_deg) for cf in cf_group})
                if not phi_values:
                    raise ValueError("No phi values found for slip system group.")

                # Write output table
                with out_path.open("w", encoding="utf-8") as fout:
                    is_cubic = str(cf0.cell.lattice).lower() == "cubic"

                    if is_cubic:
                        cols = [f"H  C_eq_avg_phi{phi:.0f}" for phi in phi_values]
                        fout.write("# h k l   " + "   ".join(cols) + "\n")
                    else:
                        cols = [f"delta_phi{phi:.0f}  C_eq_avg_phi{phi:.0f}" for phi in phi_values]
                        fout.write("# h k l   " + "   ".join(cols) + "\n")

                    rows_out: list[tuple[float, int, int, int, list[tuple[float, float]]]] = []
                    for (h, k, l), per_phi in by_hkl.items():
                        pairs: list[tuple[float, float]] = []
                        for phi in phi_values:
                            d, c = per_phi.get(phi, (0.0, 0.0))
                            if is_cubic:
                                d = _H_cubic(int(h), int(k), int(l))
                            pairs.append((float(d), float(c)))

                        sort_key = pairs[0][0] if pairs else 0.0
                        rows_out.append((float(sort_key), int(h), int(k), int(l), pairs))

                    rows_out.sort(key=lambda r: (r[0], r[1], r[2], r[3]))

                    for _x, h, k, l, pairs in rows_out:
                        fout.write(f"{h:3d} {k:3d} {l:3d}  ")
                        for (d, c) in pairs:
                            fout.write(f"{d: .8f}  {c: .8f}   ")
                        fout.write("\n")

                # Fit invariants (TOPAS only)
                E_by_phi, metrics_by_phi, csv_by_phi = run_fit_multi_phi(
                    contrast_in_path=in_path,
                    input_table=out_path,
                    output_dir=out_dir,
                    out_prefix=out_path.stem,
                )

                from cfactor.wppm_macro import write_wppm_inp_from_fit

                example_hkl = hkls[0] if hkls else (1, 1, 0)

                # Example: write TOPAS input per slip-system group
                wppm_path = out_dir / f"{out_path.stem}.wppm.inp"
                write_wppm_inp_from_fit(
                    wppm_path,
                    E_by_phi=E_by_phi,
                    wsg=int(cf0.space_group),
                    burgers_uvw=tuple(cf0.slip_system.burgers_uvw),
                    include_macro_file="WPPM_macros.inc",
                    add_size_dist_block=True,
                    example_hkl=tuple(example_hkl),
                )
                print(f"[TOPAS] WPPM input written: {wppm_path}")

                is_cubic = str(cf0.cell.lattice).lower() == "cubic"

                print("\n========================================")
                print("Invariant fit completed (multi-phi)")
                print("Fit outputs (per phi) [TOPAS only]:")

                for phi in sorted(E_by_phi.keys()):
                    coeffs_print = _zero_small_coeffs(E_by_phi[phi], tol=1e-6)
                    _print_fit_results(
                        f"PHI = {phi:g} deg",
                        coeffs_print,
                        metrics_by_phi[phi],
                        is_cubic,
                    )
                    print("  csv:", str(csv_by_phi[phi]))

                print("Run log written:")
                print("  ", str(run_log_path))
                print("========================================")

                # Write single-crystal TSVs
                cf_by_phi = {float(x.slip_system.phi_deg): x for x in cf_group}
                cf_screw = cf_by_phi.get(0.0)
                cf_edge = cf_by_phi.get(90.0)

                if cf_screw is not None:
                    single_screw_path = out_path.with_suffix(".single_crystal_screw.tsv")
                    print(f"Writing single-crystal (screw) TSV to: {single_screw_path}")
                    _write_single_crystal_tsv(
                        single_screw_path,
                        cf_screw,
                        hkls,
                        include_improper=include_improper,
                    )

                if cf_edge is not None:
                    single_edge_path = out_path.with_suffix(".single_crystal_edge.tsv")
                    print(f"Writing single-crystal (edge) TSV to: {single_edge_path}")
                    _write_single_crystal_tsv(
                        single_edge_path,
                        cf_edge,
                        hkls,
                        include_improper=include_improper,
                    )

            except Exception as ex:
                # Write .error instead of crashing without the error file
                err_path = _write_input_error_file(out_dir, in_path, ex)
                print("\n[ERROR] Invalid input or numerical failure in this slip-system group.")
                print(f"[ERROR] Details written to: {err_path}")
                return

    print("\nDone.")


# =============================================================================
# Optional: allow `python -m cfactor.main ...` (not used by discocf entrypoint)
# =============================================================================

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="cfactor.main")
    p.add_argument("input_json", nargs="?", default=None, help="Input JSON path (optional).")
    p.add_argument("-o", "--output-dir", default=None, help="Output directory (optional).")
    ns = p.parse_args(argv)

    in_path = Path(ns.input_json).expanduser().resolve() if ns.input_json else None
    out_dir = Path(ns.output_dir).expanduser().resolve() if ns.output_dir else None
    run(input_path=in_path, output_dir=out_dir)


if __name__ == "__main__":
    main()
