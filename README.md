# FLASh: Fast simulation tools for lattice structures

This repository contains the code and data used for the paper "**Fast simulation tools for LAttice Structures**" (replace with the exact paper title and citation).

This code uses **[QUGaR](https://github.com/your-org/qugar)** to compute numerical quadratures and **FEniCS** for visualization of results.

> 📄 **Note:** Update the above line with the exact title, authors, and citation (DOI/arXiv) when ready.

---

## 🚀 Installation

The library can be installed in editable mode using the provided installer script.

### Run the installer script

```bash
python install_all.py
```

### Building the documentation

The documentation is built using Sphinx. From the repository root, run:

```bash
python -m pip install -U sphinx
sphinx-build -b html docs/ docs/_build/html
```

Then open `docs/_build/html/index.html` in a browser.

This will:
- Bootstrap and activate a conda environment (via `install_qugar_with_conda.sh`)
- Install required Python dependencies
- Install this repository in editable mode (`pip install -e .`)
- Optionally download example data if `DATA_URL` is set

---

## ▶️ Usage

Examples are located in the `examples/` folder. For instance:

```bash
python examples/example_3.py
```

---

## 📦 Data

Some examples rely on data stored under `examples/data/`. You can download recommended datasets by setting `DATA_URL` when running the installer:

```bash
DATA_URL="<google-drive-url>" python install_all.py
```

---

## 📝 Notes

- The script `install_all.sh` is also included for users who prefer running the installation steps directly from bash.
