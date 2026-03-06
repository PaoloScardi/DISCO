
---

# DISCO – Dislocation Contrast Factor Calculator

DISCO is a Python-based software tool for the computation of **dislocation contrast factors (CFs)** for arbitrary crystal structures and slip systems.

It implements the general formalism of:

Martinez-Garcia et al., *Acta Cryst. A65*, 109–119 (2009)

and extends it to all space groups and Laue symmetries.

---

## Features

* All 230 space groups
* Single-crystal contrast factors
* Powder-averaged contrast factors
* Automatic symmetry handling
* Automatic invariant polynomial coefficient extraction
* TOPAS WPPM macro generation

---

## Scientific Reference

If you use DISCO, please cite:

Malagutti, M. A., D’Incau, M., & Scardi, P.
*Journal of Applied Crystallography* (2026).
DOI: (to be added)

The theoretical basis follows:

Martinez-Garcia, J., Leoni, M., & Scardi, P. (2009).
*A general approach for determining the diffraction contrast factor of straight-line dislocations.*
Acta Crystallographica Section A, 65, 109–119.
[https://doi.org/10.1107/S010876730804186X](https://doi.org/10.1107/S010876730804186X)

---

## Software Availability

Source code:
[https://github.com/PaoloScardi/DISCO](https://github.com/PaoloScardi/DISCO)

Archived version (1.01):
Zenodo DOI: https://doi.org/10.5281/zenodo.18875170

Web interface (JSON builder + server execution):
[https://energymaterials.unitn.it/tools/software/disco.html](https://energymaterials.unitn.it/tools/software/disco.html)

---

# Requirements (libraries will be installed automatically with the environment below)

* Python ≥ 3.11
* numpy
* scipy
* pandas

---

# Installation (Recommended: from GitHub)

The recommended installation method is via a Git clone and a virtual environment.

---

## 1. Clone the repository (in Power Shell or Linux prompt)

```bash
git clone https://github.com/PaoloScardi/DISCO
cd DISCO
```

---

## 2. Create a virtual environment

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
```

(or `py -3.12`, `py -3.14`, etc., depending on your installed version)

Activate (optional):

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Alternatively, you can skip activation and call the interpreter directly.

---

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install DISCO 

Run the following in the same folder as the pyproject.toml

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

---

# Usage

Run DISCO in the same folder where the .json file is located with:

```bash
discocf input.json
```
(input.json is a place holder here, put the name of your .json file)
See the examples folder with some ready-to-use .json files

Display help for more information:

```bash
discocf --help
```

Specify a custom output directory:

```bash
discocf input.json -o results_folder
```

By default, outputs are written to:

```
<input_directory>/output_fit
```

---

# Input File Overview

DISCO requires a JSON file containing:

### Crystal structure

* `space_group`
* lattice parameters (`a, b, c, alpha, beta, gamma` as required)

### Elastic stiffness constants

* Provided in GPa

### Slip system definition

* `plane_hkl`
* `burgers_uvw`
* `phi_deg`

where:

* `phi = 0°` → screw
* `phi = 90°` → edge

### Diffraction reflections

* `hkls`

Example input files are available in:

```
examples/
```

Full specification:

```
docs/INPUT_FORMAT.md
```

---

# Output Files

DISCO generates:

* Single-crystal contrast factors
* Powder-averaged contrast factors
* Symmetry invariant coefficients (Γ_hkl expansion)
* TOPAS macro snippet for WPPM
* Detailed `.run.log` file containing input parameters and intermediate data

Full description:

```
docs/OUTPUT_FORMAT.md
```

---

# Development Notes

For debugging in Visual Studio / VS Code:

* Open the cloned repository folder
* Select the `.venv` interpreter
* Install in editable mode (`pip install -e .`)
* Run `discocf` from the integrated terminal

---

# License

DISCO is distributed under the GNU General Public License v3.0 (GPL-3.0).

Commercial use or integration into proprietary software requires explicit permission from the authors.

---

# Version

DISCO v1.0.1 (2026)

---

# Authors

Marcelo A. Malagutti
Mirco D’Incau
Paolo Scardi

Department of Civil, Environmental & Mechanical Engineering
University of Trento, Italy

---
