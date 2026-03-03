# cfactor/io_input.py
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

from .models import CellParam, ElastConst, SlipSystem, ContrastFactorInput, miller4_to_miller3_for_hex_like



def infer_lattice_from_space_group(space_group: int) -> str:
    """
    Infer crystal system / lattice family from International space-group number (1..230).
    This is sufficient for building the point-group rotations used in this project.

    Ranges (International Tables):
      1-2   triclinic
      3-15  monoclinic
      16-74 orthorhombic
      75-142 tetragonal
      143-167 trigonal
      168-194 hexagonal
      195-230 cubic
    """
    sg = int(space_group)
    if sg < 1 or sg > 230:
        raise ValueError(f"space_group must be in 1..230, got {space_group}")
    if sg <= 2:
        return "triclinic"
    if sg <= 15:
        return "monoclinic"
    if sg <= 74:
        return "orthorhombic"
    if sg <= 142:
        return "tetragonal"
    if sg <= 167:
        return "trigonal"
    if sg <= 194:
        return "hexagonal"
    return "cubic"


# Mappa Voigt (ij) -> indice
_VOIGT_IDX: Dict[Tuple[int, int], int] = {
    (1, 1): 0, (2, 2): 1, (3, 3): 2,
    (2, 3): 3, (1, 3): 4, (1, 2): 5,
}
# Inversa: indice -> (ij)
_IDX_TO_VOIGT: Dict[int, Tuple[int, int]] = {v: k for k, v in _VOIGT_IDX.items()}

_EXPECTED_COUNTS = {
    "cubic": 3,
    "hexagonal": 5,
    "trigonal": 6,
    "tetragonal": 6,
    "orthorhombic": 9,
    "monoclinic": 13,
    "triclinic": 21,
}


def _assemble_C_from_dict(coeffs: dict) -> np.ndarray:
    C = np.zeros((6, 6), dtype=float)
    for key, val in coeffs.items():
        # key: C11, C12, ... C66
        m = key.strip().upper()
        if not (m.startswith("C") and len(m) == 3 and m[1].isdigit() and m[2].isdigit()):
            raise ValueError(f"Chiave elastica non valida: {key}. Atteso 'Cij' con i,j in 1..6")
        idx = m[1:]
        i = int(idx[0]) - 1
        j = int(idx[1]) - 1
        if i < 0 or i > 5 or j < 0 or j > 5:
            raise ValueError(f"Indice Voigt fuori range in {key}")
        C[i, j] = float(val)
        C[j, i] = float(val)  # simmetrizza
    return C


def _validate_expected_count(C: np.ndarray, lattice: str) -> None:
    """
    Confronta il numero di elementi indipendenti non nulli con un conteggio atteso per la simmetria.
    Non è una validazione rigorosa, emette warning (via print) se c'è forte discrepanza.
    """
    lat = lattice.lower()
    expected = _EXPECTED_COUNTS.get(lat)
    if expected is None:
        return
    upper = C[np.triu_indices(6)]
    distinct_nonzero = np.unique(np.round(upper[upper != 0.0], decimals=12)).size
    if distinct_nonzero and abs(distinct_nonzero - expected) >= 3:
        print(
            f"Warning: il numero di costanti elastiche indipendenti ({distinct_nonzero}) "
            f"non coincide con quello tipico per '{lat}' ({expected}). Verifica l'input."
        )


def _parse_elastic(elastic_json: dict, lattice: str) -> ElastConst:
    """
    Parsing flessibile:
    - Se presente 'C' come lista 6x6, la usa direttamente.
    - Altrimenti accetta C11,C12,... e costruisce la matrice.
    - Completa automaticamente alcune relazioni di simmetria se mancano.
    """
    coeffs = {}
    if "C" in elastic_json:
        C_list = elastic_json["C"]
        if not (
            isinstance(C_list, list)
            and len(C_list) == 6
            and all(isinstance(row, list) and len(row) == 6 for row in C_list)
        ):
            raise ValueError("Elastic.C deve essere una lista 6x6")
        C = np.array(C_list, dtype=float)
    else:
        coeffs = {k.upper(): float(v) for k, v in elastic_json.items() if k.upper().startswith("C")}
        if not coeffs:
            raise ValueError("Sezione 'elastic' deve contenere 'C' 6x6 oppure valori 'Cij'")
        C = _assemble_C_from_dict(coeffs)

    lat = lattice.lower()

    # COMPLETAMENTO PER SIMMETRIA (solo se coeffs disponibili)
    if coeffs:
        if lat == "cubic":
            c11 = coeffs.get("C11", C[0, 0])
            c12 = coeffs.get("C12", C[0, 1])
            c44 = coeffs.get("C44", C[3, 3])
            C[:] = 0.0
            C[0, 0] = C[1, 1] = C[2, 2] = c11
            C[0, 1] = C[0, 2] = C[1, 2] = c12
            C[1, 0] = C[2, 0] = C[2, 1] = c12
            C[3, 3] = C[4, 4] = C[5, 5] = c44

        elif lat == "hexagonal":
        # Hexagonal: independent (C11, C12, C13, C33, C44) and C66 = (C11 - C12)/2 unless provided.

            c11 = float(coeffs.get("C11", C[0, 0]))
            c12 = float(coeffs.get("C12", C[0, 1]))
            c13 = float(coeffs.get("C13", C[0, 2]))
            c33 = float(coeffs.get("C33", C[2, 2]))
            c44 = float(coeffs.get("C44", C[3, 3]))

            # allow explicit C66 override, otherwise derive it
            c66 = float(coeffs.get("C66", 0.5 * (c11 - c12)))

            C[:] = 0.0

            # normal block
            C[0, 0] = C[1, 1] = c11
            C[0, 1] = C[1, 0] = c12
            C[0, 2] = C[2, 0] = c13
            C[1, 2] = C[2, 1] = c13
            C[2, 2] = c33

            # shear block
            C[3, 3] = c44      # C44
            C[4, 4] = c44      # C55 = C44
            C[5, 5] = c66      # C66

        elif lat == "tetragonal":
            # Tetragonal:
            # - support both 6-independent (C11,C12,C13,C33,C44,C66)
            # - and 7-independent with C16 (as in Martinez-Garcia package)
            #
            # IMPORTANT: The Mathematica package's 7-stiffness form uses:
            #   row2,col2 = C12  (non-standard but we reproduce it if C16 is provided)

            c11 = float(coeffs.get("C11", C[0, 0]))
            c12 = float(coeffs.get("C12", C[0, 1]))
            c13 = float(coeffs.get("C13", C[0, 2]))
            c33 = float(coeffs.get("C33", C[2, 2]))
            c44 = float(coeffs.get("C44", C[3, 3]))
            c66 = float(coeffs.get("C66", C[5, 5]))

            # optional coupling (7th independent const)
            c16 = float(coeffs.get("C16", C[0, 5]))

            C[:] = 0.0

            if abs(c16) > 0.0:
                # --- Match Mathematica ElastConst[... c16 ..., Tetragonal] (7 stiffness) ---
                C[0, 0] = c11
                C[0, 1] = c12
                C[0, 2] = c13
                C[0, 5] = c16

                C[1, 0] = c12
                C[1, 1] = c12      # <-- yes, matches the Mathematica package
                C[1, 2] = c13
                C[1, 5] = -c16

                C[2, 0] = c13
                C[2, 1] = c13
                C[2, 2] = c33

                C[3, 3] = c44
                C[4, 4] = c44

                C[5, 0] = c16
                C[5, 1] = -c16
                C[5, 5] = c66

            else:
                # --- Standard tetragonal (6 stiffness) ---
                C[0, 0] = C[1, 1] = c11
                C[0, 1] = C[1, 0] = c12
                C[0, 2] = C[2, 0] = c13
                C[1, 2] = C[2, 1] = c13
                C[2, 2] = c33

                C[3, 3] = c44
                C[4, 4] = c44
                C[5, 5] = c66



        elif lat == "trigonal":
            # Trigonal in hexagonal axes (e.g. -3m / R-3c style):
            # [ C11 C12 C13  C14 -C14  0 ]
            # [ C12 C11 C13 -C14  C14  0 ]
            # [ C13 C13 C33   0    0   0 ]
            # [ C14 -C14  0  C44   0   0 ]
            # [ -C14 C14 0   0   C44  0 ]
            # [  0   0   0   0    0  C66 ]
            #
            # where C66 = (C11 - C12)/2 unless provided.

            c11 = float(coeffs.get("C11", C[0, 0]))
            c12 = float(coeffs.get("C12", C[0, 1]))
            c13 = float(coeffs.get("C13", C[0, 2]))
            c33 = float(coeffs.get("C33", C[2, 2]))
            c44 = float(coeffs.get("C44", C[3, 3]))
            c14 = float(coeffs.get("C14", C[0, 3]))

            c66 = float(coeffs.get("C66", 0.5 * (c11 - c12)))

            C[:] = 0.0

            # normal block
            C[0, 0] = C[1, 1] = c11
            C[0, 1] = C[1, 0] = c12
            C[0, 2] = C[2, 0] = c13
            C[1, 2] = C[2, 1] = c13
            C[2, 2] = c33

            # shear diagonal
            C[3, 3] = c44
            C[4, 4] = c44
            C[5, 5] = c66

            # trigonal coupling (C14 with ± pattern)
            C[0, 3] = C[3, 0] =  c14
            C[0, 4] = C[4, 0] = -c14
            C[1, 3] = C[3, 1] = -c14
            C[1, 4] = C[4, 1] =  c14

            # the standard form also implies these symmetric couplings are zero explicitly
            # (keep as 0.0): C[2,3], C[2,4], C[3,4], etc.


    _validate_expected_count(C, lattice=lat)
    return ElastConst(C=C)


def load_contrast_input(path: str | Path) -> List[ContrastFactorInput]:
    """
    Legge contrast_factors.in (JSON) e ritorna una lista di ContrastFactorInput
    (uno per combinazione slip system × phi_deg), supportando simmetrie e
    costanti elastiche generali.

    New behavior:
      - laue_symbol is optional (legacy)
      - cell.lattice is optional if space_group is given
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # --- Space group and lattice inference ---
    space_group = data.get("space_group", data.get("space_group_number", None))
    space_group_name = data.get("space_group_name", None)

    cell_json = data.get("cell", {})
    lattice_in = cell_json.get("lattice", None)

    if lattice_in is None:
        if space_group is None:
            raise ValueError(
                "Input must provide either cell.lattice or space_group (1..230). "
                "Recommended: provide space_group and omit cell.lattice."
            )
        lattice = infer_lattice_from_space_group(int(space_group))
    else:
        lattice = str(lattice_in).lower()
        if space_group is not None:
            inferred = infer_lattice_from_space_group(int(space_group))
            if inferred != lattice:
                print(
                    f"Warning: cell.lattice='{lattice}' but space_group={space_group} "
                    f"implies '{inferred}'. Using '{lattice}'."
                )

    cell = CellParam.from_dict(lattice=lattice, cell=cell_json)
    elastic = _parse_elastic(data["elastic"], lattice=lattice)

    laue_symbol = data.get("laue_symbol", "")  # legacy/optional
    material_name = data.get("material_name", "Material")

    slip_systems = data.get("slip_systems", [])
    if not slip_systems:
        raise ValueError("Nessun slip system definito in 'slip_systems'")

    def _is_close(a: float, b: float, tol: float = 1e-6) -> bool:
        return abs(float(a) - float(b)) <= tol

    result: List[ContrastFactorInput] = []

    for ss in slip_systems:
        plane = [int(x) for x in ss["plane_hkl"]]
        burgers = [int(x) for x in ss["burgers_uvw"]]

        if lattice == "hexagonal":
            if len(plane) == 4 or len(burgers) == 4:
                plane, burgers = miller4_to_miller3_for_hex_like(plane, burgers, strict=True)

        elif lattice == "trigonal":
            trig_hex_axes = _is_close(cell.gamma, 120.0) or _is_close(cell.gamma, 60.0)
            if len(plane) == 4 or len(burgers) == 4:
                if not trig_hex_axes:
                    raise ValueError(
                        "Trigonal 4-index Miller-Bravais indices are only supported in the hexagonal-axes setting "
                        "(cell.gamma ~ 60 or 120). Provide 3-index (h,k,l)/(u,v,w) instead."
                    )
                plane, burgers = miller4_to_miller3_for_hex_like(plane, burgers, strict=True)

        phi_raw = ss.get("phi_deg", 0.0)
        if isinstance(phi_raw, (int, float)):
            phi_list = [float(phi_raw)]
        else:
            phi_list = [float(x) for x in phi_raw]

        for phi in phi_list:
            slip = SlipSystem(
                plane_hkl=plane,
                burgers_uvw=burgers,
                phi_deg=phi,
            )
            result.append(
                ContrastFactorInput(
                    slip_system=slip,
                    cell=cell,
                    elastic=elastic,
                    laue_symbol=laue_symbol,
                    space_group=int(space_group) if space_group is not None else None,
                    space_group_name=space_group_name,
                    material_name=material_name,
                )
            )

    return result


def load_input(path: str | Path):
    # alias per compatibilità retro
    return load_contrast_input(path)
