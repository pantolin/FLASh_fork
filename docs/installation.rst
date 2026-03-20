Installation
============

This document explains how to install the FLASh Python library in editable
mode.

The library can be installed in editable mode using the provided installer
script.

Run the installer script:

.. code-block:: bash

   python install_all.py

When you run this script, it first activates / bootstraps a conda environment
and installs **QUGaR** (via `install_qugar_with_conda.sh`) before installing the
FLASh package itself.

The installer script performs:

- conda environment creation/activation (`install_qugar_with_conda.sh`)
- installation of required Python dependencies
- `pip install -e .` for FLASh
- optional data download via `DATA_URL` (if set)

If you prefer, you can also install manually from an existing Python
environment:

.. code-block:: bash

   python -m pip install -e .
