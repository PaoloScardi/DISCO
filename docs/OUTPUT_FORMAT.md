# Output Files

DISCO generates:

---

## 1. Single Crystal Contrast Factors

File: *_single.csv*

Contains:

- h k l
- φ (deg)
- C_hkl (single crystal)

---

## 2. Powder-Averaged Contrast Factors

File: *_powder.csv*

Contains:

- h k l
- φ (deg)
- ⟨C⟩_hkl

Powder average is computed over symmetry-equivalent slip systems.

---

## 3. Invariant Coefficients

File: *_invariants.csv*

Contains fitted coefficients for Γ_hkl expansion.

Used in:

- WPPM
- Modified Williamson–Hall analysis

---

## 4. TOPAS Macro

File: *.inp

Snippet ready for Wilkens microstrain modelling.

---

## 5. Run Log

File: *.run.log

Contains:

- Parsed input
- Symmetry operations
- Elastic matrices
- Diagnostic info