
# NIFLOSIC: Non-Iterative Fermi–Löwdin Orbital Self-Interaction Correction

This repository implements the Non-Iterative Fermi–Löwdin Orbital Self-Interaction Correction (NIFLOSIC) method, a computationally efficient alternative to traditional FLOSIC. NIFLOSIC eliminates the need for iterative relaxation of Fermi orbital descriptors (FODs), significantly reducing computational cost while maintaining high accuracy.


## Usage

The main script is `run_test_flosic.py` and accepts the following arguments:

```bash
python run_test_flosic.py <molecule> <functional> <basis> <method> <mkorbs> <charge> <spin>
```

### Arguments

| Argument     | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `molecule`   | Name of the molecule (reads from `molecule.xyz`)                            |
| `functional` | Exchange-correlation functional(s), comma-separated (e.g., `pbe,pbe`)       |
| `basis`      | Basis set (e.g., `6-31G`)                                                    |
| `method`     | Method for orbital localization: `scdm-g`, `elf`, `lol`, `grid`             |
| `mkorbs`     | Orbital generation method: `scdm-g`, `flo`                                 |
| `charge`     | Integer charge of the system                                                |
| `spin`       | Spin multiplicity minus one (e.g., `1` for singlet, `2` for doublet, etc.)  |

### Example

To run a NIFLOSIC calculation for CH4 with the PBE functional and the 6-31G basis:

```bash
python run_test_flosic.py CH4 pbe,pbe 6-31G elf flo 0 1
```


The code has been tested using the following Python packages:

- PySCF: v2.6.0
- NumPy: v1.25.2
- SciPy: v1.15.1
- PyTorch: v2.9.0
