
---

# 📘 docs/INPUT_FORMAT.md

```markdown
# Input File Format

DISCO requires a JSON file with the following structure.

---

## Required Fields

### material_name (string)
Used for output file naming.

### space_group (integer)
International space group number (1–230).

### cell
Crystal lattice parameters.

Cubic:
- a

Hexagonal:
- a
- c

Monoclinic:
- a
- b
- c
- beta

Triclinic:
- a, b, c, alpha, beta, gamma

---

### elastic

Elastic stiffness constants in GPa.

Provide independent Cij components according to symmetry.

Example (hexagonal):

C11, C12, C13, C33, C44

---

### slip_systems

List of slip systems.

Each system requires:

- plane_hkl: [h, k, l]
- burgers_uvw: [u, v, w]
- phi_deg: [list of dislocation character angles]

φ = 0° → screw  
φ = 90° → edge  

---

### hkls

List of reflections for which contrast factors are computed.

Example:

"hkls": [
  [1,0,0],
  [1,0,1],
  [0,0,4]
]