[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# DISCO – Dislocation Contrast Factor Calculator

DISCO is a general Python-based software tool for the computation of **dislocation contrast factors (CFs)** for arbitrary crystal structures and slip systems.

It implements the general formalism of Martinez-Garcia et al. (Acta Cryst. A65, 109–119, 2009) and extends it to all space groups and Laue symmetries.

DISCO supports:

- All 230 space groups
- Single-crystal contrast factors
- Powder-averaged contrast factors
- Automatic symmetry handling
- Automatic invariant polynomial coefficient extraction
- TOPAS WPPM macro generation

---

## 📖 Scientific Reference

If you use DISCO, please cite:

Malagutti et al., *J. Appl. Crystallogr.* (2026).  
DOI: (to be added)

The theoretical basis follows:

Martinez-Garcia, J. et al., Acta Cryst. A65, 109–119 (2009).  
https://doi.org/10.1107/S010876730804186X

---

## 🔗 Software Availability

Source code:  
GitHub: (repository link)

Archived version (citable):  
Zenodo DOI: (to be added)

Web interface (JSON builder + server execution):  
https://energymaterials.unitn.it/tools/software/disco.html

---

## ⚙ Requirements

- Python ≥ 3.11
- numpy
- scipy
- pandas

---

## 💻 Installation

### Linux / macOS

```bash
python3 -m pip install --upgrade pip
python3 -m pip install . --upgrade

#Power shell

pip install . --upgrade


##Running disco

discocf [-h] [-o OUTPUT_DIR] [input]

positional arguments:
  input                 Input JSON file (e.g. contrast_factors.json)

options:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory where outputs will be written (default: <input_dir>/output_fit)

---

## 📂 Input File Overview

DISCO requires a JSON file containing:

- **Crystal structure:**
  - `space_group`
  - lattice parameters (`a`, `b`, `c`, `alpha`, `beta`, `gamma` as required)

- **Elastic stiffness constants** (in GPa)

- **Slip system definition:**
  - Slip plane (`plane_hkl`)
  - Burgers vector (`burgers_uvw`)
  - Dislocation character angle (`phi_deg`)
    - φ = 0° → screw
    - φ = 90° → edge

- **List of diffraction reflections** (`hkls`)

Example input files are available in the `examples/` folder.

A detailed description of the input structure is provided in:


---

## 📊 Output Files

DISCO generates:

- **Single-crystal contrast factors**
- **Powder-averaged contrast factors**
- **Symmetry invariant coefficients (Γₕₖₗ expansion)**
- **TOPAS macro snippet for WPPM**
- **Detailed `.run.log` file** containing input parameters and intermediate data

A full description of output files is available in:
docs/INPUT_FORMAT.md


---

## 📊 Output Files

DISCO generates:

- **Single-crystal contrast factors**
- **Powder-averaged contrast factors**
- **Symmetry invariant coefficients (Γₕₖₗ expansion)**
- **TOPAS macro snippet for WPPM**
- **Detailed `.run.log` file** containing input parameters and intermediate data

A full description of output files is available in:
docs/OUTPUT_FORMAT.md


---

## 📜 License

DISCO is distributed under the **GNU General Public License v3.0 (GPL-3.0)**.

Commercial use or integration into proprietary software requires explicit permission from the authors.

---

## 📌 Version

DISCO v1.0.0 (2026)

---

## 👨‍🔬 Authors

Marcelo A. Malagutti  
Mirco D’Incau  
Paolo Scardi  

Department of Civil, Environmental & Mechanical Engineering  
University of Trento, Italy
