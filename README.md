# FLASh: Fast simulation tools for lattice structures

This repository contains the code and data used for the paper "---" (replace with the exact paper title and citation).

This code was developed by Gonzalo Bonilla Moreno during a stay at the MNS lab at EPFL (https://www.epfl.ch/labs/mns/) under the supervision of Pablo Antolin. The macro geometries for the wrench and wing examples were designed by Giuliano Guarino.

This code uses **[QUGaR](https://github.com/pantolin/qugar)** to compute numerical quadratures and **[FEniCS](https://fenicsproject.org)** for visualization of results.

---

## 🚀 Installation

The library can be installed in editable mode using the provided installer script.

### Run the installer script

```bash
python install.py
```

This will:
- Bootstrap and activate a conda environment (via `install_qugar_with_conda.sh`)
- Install required Python dependencies
- Install this repository in editable mode (`pip install -e .`)
- Download the ROM database from [Zenodo](https://zenodo.org/records/19254389)

To skip the data download:

```bash
python install.py --skip-data
```

### Building the documentation

The documentation is built using Sphinx. From the repository root, run:

```bash
python -m pip install -U sphinx
sphinx-build -b html docs/ docs/_build/html
```

Then open `docs/_build/html/index.html` in a browser.

---

## 📦 Data

Some examples rely on data stored under `examples/data/`. The ROM database (~6.3 GB) is automatically downloaded from [Zenodo](https://zenodo.org/records/19254389) during installation, which may take some time depending on your connection. To download it manually:

```bash
curl -L -o rom_data.tar.gz https://zenodo.org/records/19254389/files/rom_data.tar.gz
tar -xzf rom_data.tar.gz -C examples/data/
```

---

## 📁 Repository Structure

If you download our prebuilt rom models make sure the structure of the data repository is correct

```
FLASh/
│
├── FLASh/
│   ├── mesh/
│   ├── pde/
│   ├── rom/
│   └── utils/
│
├── examples/
│   ├── data/
│   │   ├── figs/
|   |   ├── results/
|   |   └── rom_data/
|   |       ├── ...
|   |       └── rom_model/
|   |           ├── bM_core
|   |           ├── M_core
|   |           └── K_core
│   │   ├── wing/
│   │   └── wrench/
│
└── docs/
```


---

## ▶️ Usage

Examples are located in the `examples/` folder. For instance:

Some examples have jupyter notebooks available like `examples/example_1.ipynb`

Examples can be run either in serial 

```bash
python examples/example_3.py
```

or in parallel 

```bash
mpirun -n 10 python examples/example_3.py
```
---

## 📝 Notes

- The script `install.sh` is also included for users who prefer running the installation steps directly from bash.
