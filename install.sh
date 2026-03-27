#!/usr/bin/env bash
set -eo pipefail

# install.sh
# ----------
# Convenience script to:
#  1) Run the existing `install_qugar_with_conda.sh` conda bootstrap
#  2) Install additional Python dependencies via pip
#  3) Install the local FLASh library in editable mode
#  4) Download the ROM database from Zenodo
#
# Usage:
#   ./install.sh
#   ./install.sh --skip-data
#   DATA_DIR="/path/to/data" ./install.sh
#
# Notes:
# - The ROM database is downloaded from Zenodo by default.
# - Pass --skip-data to skip the download step.
# - This script assumes `conda` is on PATH. If not, start from a shell that sources conda.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

# Defaults (can be overridden via env vars)
ENV_NAME="${QUGAR_ENV_NAME:-flash-env}"
DATA_URL="https://zenodo.org/records/19254389/files/rom_data.tar.gz"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/examples/data}"
SKIP_DATA=false

# Parse arguments
for arg in "$@"; do
  case "$arg" in
    --skip-data) SKIP_DATA=true ;;
  esac
done

# 1) Run conda-based bootstrap (creates/updates conda env + builds/install QUGaR internals)
if [[ ! -f "${REPO_ROOT}/install_qugar_with_conda.sh" ]]; then
  echo "[ERROR] install_qugar_with_conda.sh not found in ${REPO_ROOT}"
  exit 1
fi

echo "[INFO] Running conda bootstrap script (this can take a while)..."
export PYTHON_VERSION=3.12
bash "${REPO_ROOT}/install_qugar_with_conda.sh"

# 2) Activate the conda environment
#    - Use `conda activate` via the conda.sh helper.
#    - Avoid `source activate` syntax which is deprecated.

if ! command -v conda &>/dev/null; then
  echo "[ERROR] conda not found in PATH. Ensure Miniconda/Anaconda is installed and available."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# 3) Install Python dependencies and the local library
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${REPO_ROOT}"

# Install pip packages
python -m pip install splipy
python -m pip install line_profiler
python -m pip install tqdm
python -m pip install sympy
python -m pip install numba
python -m pip install scikit-learn

# Install conda package (MPI-enabled h5py)
conda install -c conda-forge h5py=*=mpi_mpich_* -y

# 4) Download ROM database from Zenodo
if [[ "${SKIP_DATA}" == "false" ]]; then
  mkdir -p "${DATA_DIR}"
  echo "[INFO] Downloading ROM database from ${DATA_URL} into ${DATA_DIR}"

  if command -v curl &>/dev/null; then
    curl -L -o "${DATA_DIR}/rom_data.tar.gz" "${DATA_URL}"
  elif command -v wget &>/dev/null; then
    wget -O "${DATA_DIR}/rom_data.tar.gz" "${DATA_URL}"
  else
    echo "[ERROR] Neither curl nor wget found. Please install one and re-run."
    exit 1
  fi

  echo "[INFO] Unpacking ROM database..."
  tar -xzf "${DATA_DIR}/rom_data.tar.gz" -C "${DATA_DIR}"
  rm "${DATA_DIR}/rom_data.tar.gz"

  echo "[INFO] ROM database downloaded and unpacked in ${DATA_DIR}."
else
  echo "[INFO] Skipping ROM database download (--skip-data)."
fi

echo "[SUCCESS] Setup complete. Activate the environment with: conda activate ${ENV_NAME}"
