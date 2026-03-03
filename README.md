Perfect — below is the **pure text version** (no emojis, no badges, no special formatting).
You can copy and paste this directly into GitHub.

---

# DISCO – Dislocation Contrast Factor Calculator

DISCO is a general Python-based software tool for the computation of dislocation contrast factors (CFs) for arbitrary crystal structures and slip systems.

It implements the general formalism of Martinez-Garcia et al. (Acta Cryst. A65, 109–119, 2009) and extends it to all space groups and Laue symmetries.

DISCO supports:

* All 230 space groups
* Single-crystal contrast factors
* Powder-averaged contrast factors
* Automatic symmetry handling
* Automatic invariant polynomial coefficient extraction
* TOPAS WPPM macro generation

## Scientific Reference

If you use DISCO, please cite:

Malagutti, M. A., D’Incau, M., & Scardi, P., Journal of Applied Crystallography (2026).
DOI: (to be added)

The theoretical basis follows:

Martinez-Garcia, J., Leoni, M., & Scardi, P. (2009).
A general approach for determining the diffraction contrast factor of straight-line dislocations.
Acta Crystallographica Section A, 65, 109–119.
[https://doi.org/10.1107/S010876730804186X](https://doi.org/10.1107/S010876730804186X)

## Software Availability

Source code:
GitHub: (repository link)

Archived version (citable):
Zenodo DOI: (to be added after release)

Web interface (JSON builder + server execution):
[https://energymaterials.unitn.it/tools/software/disco.html](https://energymaterials.unitn.it/tools/software/disco.html)

## Requirements

* Python ≥ 3.11
* numpy
* scipy
* pandas

## Installation

Linux / macOS:

```
python3 -m pip install --upgrade pip
python3 -m pip install . --upgrade
```

Windows (PowerShell):

```
pip install . --upgrade
```

## Usage

Run DISCO with:

```
discocf input.json
```

Display help:

```
discocf --help
```

Specify a custom output directory:

```
discocf input.json -o results_folder
```

By default, outputs are written to:

```
<input_directory>/output_fit
```

## Input File Overview

DISCO requires a JSON file containing:

* Crystal structure

  * space_group
  * lattice parameters (a, b, c, alpha, beta, gamma as required)

* Elastic stiffness constants (in GPa)

* Slip system definition

  * plane_hkl
  * burgers_uvw
  * phi_deg

    * phi = 0° → screw
    * phi = 90° → edge

* List of diffraction reflections (hkls)

Example input files are available in the examples folder.

A detailed description of the input structure is available in:

```
docs/INPUT_FORMAT.md
```

## Output Files

DISCO generates:

* Single-crystal contrast factors
* Powder-averaged contrast factors
* Symmetry invariant coefficients (Gamma_hkl expansion)
* TOPAS macro snippet for WPPM
* Detailed .run.log file containing input parameters and intermediate data

A full description of output files is available in:

```
docs/OUTPUT_FORMAT.md
```

## License

DISCO is distributed under the GNU General Public License v3.0 (GPL-3.0).

Commercial use or integration into proprietary software requires explicit permission from the authors.

## Version

DISCO v1.0.0 (2026)

## Authors

Marcelo A. Malagutti
Mirco D’Incau
Paolo Scardi

Department of Civil, Environmental & Mechanical Engineering
University of Trento, Italy

---

