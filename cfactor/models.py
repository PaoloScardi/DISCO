# cfactor/models.py
from dataclasses import dataclass
from typing import Literal, List
import numpy as np

LatticeType = Literal[
    "cubic",
    "hexagonal",
    "tetragonal",
    "trigonal",
    "orthorhombic",
    "monoclinic",
    "triclinic"
]

@dataclass
class CellParam:
    a: float  # Angstrom
    b: float  # Angstrom
    c: float  # Angstrom
    alpha: float  # degrees
    beta: float   # degrees
    gamma: float  # degrees
    lattice: LatticeType

    @classmethod
    def cubic(cls, a: float) -> "CellParam":
        return cls(a=a, b=a, c=a, alpha=90.0, beta=90.0, gamma=90.0, lattice="cubic")

    @classmethod
    def from_dict(cls, lattice: LatticeType, cell: dict) -> "CellParam":
        """
        Table-driven cell construction by crystal system.

        Independent parameters required (per project convention):
          - cubic:        a
          - tetragonal:   a, c
          - hexagonal:    a, c            (enforce b=a, alpha=beta=90, gamma=120)
          - trigonal:     default to hexagonal setting a, c (enforce hex-like angles)
          - orthorhombic: a, b, c
          - monoclinic:   a, b, c, beta   (enforce alpha=gamma=90)
          - triclinic:    a, b, c, alpha, beta, gamma
        """
        lat = str(lattice).lower()

        def req(name: str) -> float:
            if name not in cell:
                raise ValueError(f"cell.{name} is required for lattice='{lat}'")
            return float(cell[name])

        def opt(name: str, default: float) -> float:
            v = cell.get(name, None)
            return default if v is None else float(v)

        if lat == "cubic":
            a = req("a")
            return cls(a=a, b=a, c=a, alpha=90.0, beta=90.0, gamma=90.0, lattice="cubic")

        if lat == "tetragonal":
            a = req("a")
            c = req("c")
            return cls(a=a, b=a, c=c, alpha=90.0, beta=90.0, gamma=90.0, lattice="tetragonal")

        if lat == "hexagonal":
            a = req("a")
            c = req("c")
            return cls(a=a, b=a, c=c, alpha=90.0, beta=90.0, gamma=120.0, lattice="hexagonal")

        if lat == "trigonal":
            # Project convention: treat trigonal as hexagonal axes (a,c) unless explicitly handled elsewhere.
            # This matches your table's "Trigonal (hexagonal setting)" row.
            a = req("a")
            c = req("c")
            return cls(a=a, b=a, c=c, alpha=90.0, beta=90.0, gamma=120.0, lattice="trigonal")

        if lat == "orthorhombic":
            a = req("a")
            b = req("b")
            c = req("c")
            return cls(a=a, b=b, c=c, alpha=90.0, beta=90.0, gamma=90.0, lattice="orthorhombic")

        if lat == "monoclinic":
            a = req("a")
            b = req("b")
            c = req("c")
            beta = req("beta")
            # enforce unique-b convention commonly assumed elsewhere in this project
            return cls(a=a, b=b, c=c, alpha=90.0, beta=beta, gamma=90.0, lattice="monoclinic")

        if lat == "triclinic":
            a = req("a")
            b = req("b")
            c = req("c")
            alpha = req("alpha")
            beta = req("beta")
            gamma = req("gamma")
            return cls(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma, lattice="triclinic")

        raise ValueError(f"Unsupported lattice='{lattice}'")


@dataclass
class ElastConst:
    """
    Elastic stiffness in Voigt 6x6, GPa.
    Stored as full symmetric matrix C (6x6).
    """
    C: np.ndarray  # shape (6,6)

    def as_voigt_6x6(self) -> np.ndarray:
        """Backward-compatible API used by elasticity.py."""
        return np.asarray(self.C, dtype=float)

    def copy(self) -> "ElastConst":
        return ElastConst(C=np.array(self.C, dtype=float, copy=True))



def miller4_to_miller3_for_hex_like(
    plane_hkl: list[int],
    burgers_uvw: list[int],
    *,
    strict: bool = True,
) -> tuple[list[int], list[int]]:
    """
    Convert 4-index Miller-Bravais (hex/trig) to 3-index used by the rest of the code.

    Plane:  (h k i l) -> (H K L) = (h k l)  with i = -(h+k)
    Dir.:   [u v t w] -> [U V W] = (u v w)  with t = -(u+v)

    If strict=True, raises ValueError when i or t are not consistent.
    If strict=False, it will ignore inconsistent i/t and just drop them.

    Returns: (plane_hkl_3, burgers_uvw_3)
    """

    def _plane(p: list[int]) -> list[int]:
        if len(p) == 3:
            return [int(p[0]), int(p[1]), int(p[2])]
        if len(p) != 4:
            raise ValueError(f"plane_hkl must have 3 or 4 ints, got {len(p)}: {p}")
        h, k, i, l = map(int, p)
        if strict and i != -(h + k):
            raise ValueError(f"Inconsistent hex plane (h,k,i,l)={p}: expected i=-(h+k)={-(h+k)}")
        return [h, k, l]

    def _dir(b: list[int]) -> list[int]:
        if len(b) == 3:
            return [int(b[0]), int(b[1]), int(b[2])]
        if len(b) != 4:
            raise ValueError(f"burgers_uvw must have 3 or 4 ints, got {len(b)}: {b}")
        u, v, t, w = map(int, b)
        if strict and t != -(u + v):
            raise ValueError(f"Inconsistent hex direction (u,v,t,w)={b}: expected t=-(u+v)={-(u+v)}")
        return [u, v, w]

    return _plane(plane_hkl), _dir(burgers_uvw)



@dataclass
class SlipSystem:
    # (hkl) plane normal and [uvw] Burgers direction in Miller indices
    plane_hkl: List[int]   # e.g. [0, 1, 1]
    burgers_uvw: List[int] # e.g. [1, 0, 0]
    phi_deg: float         # dislocation character angle in degrees


@dataclass
class ContrastFactorInput:
    """Full set of inputs needed for a single calculation."""
    slip_system: SlipSystem
    cell: CellParam
    elastic: ElastConst

    # legacy / optional (NOT required anymore)
    laue_symbol: str = ""

    # preferred: provide space_group in the JSON input and omit laue_symbol + cell.lattice
    space_group: int | None = None   # 1..230
    space_group_name: str | None = None

    material_name: str = "Material"
