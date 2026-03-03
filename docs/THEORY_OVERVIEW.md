# Theoretical Overview

DISCO implements the general formalism for dislocation contrast factors:

C_hkl = G_ijmn E_ijmn

Where:

G → geometric tensor (depends on slip system + diffraction vector)

E → elastic tensor (computed via Stroh formalism)

Elastic contribution is computed following:

Martinez-Garcia et al.,
Acta Cryst. A65, 109–119 (2009)

The powder-averaged contrast factor is:

⟨C_hkl⟩ = (1/N) Σ C_i

where the sum runs over symmetry-equivalent slip systems.

Invariant polynomial expansion Γ_hkl follows Popa (1998).