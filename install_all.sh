#!/usr/bin/env bash
set -euo pipefail

# install_all.sh
# --------------
# Convenience script to:
#  1) Run the existing `install_qugar_with_conda.sh` conda bootstrap
#  2) Install additional Python dependencies via pip
#  3) Install the local FLASh library in editable mode
#  4) Optionally download data from a Drive repository (Google Drive / shared folder)
#
# Usage:
#   ./install_all.sh
#   DATA_URL="<google-drive-folder-or-file-url>" ./install_all.sh
#   DATA_URL="<url>" DATA_DIR="/path/to/data" ./install_all.sh
#
# Notes:
# - `DATA_URL` is optional; if set, the script will try to download it using `gdown`.
# - This script assumes `conda` is on PATH. If not, start from a shell that sources conda.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

# Defaults (can be overridden via env vars)
ENV_NAME="${QUGAR_ENV_NAME:-flash-env}"
DATA_URL="${DATA_URL:-}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/examples/data}"

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

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

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

# 4) Optional: download data from a Drive repository
if [[ -n "${DATA_URL}" ]]; then
  mkdir -p "${DATA_DIR}"
  echo "[INFO] Downloading data from ${DATA_URL} into ${DATA_DIR}"

  # Use gdown if available, otherwise install it (it is in requirements.txt so should be present)
  if command -v gdown &>/dev/null; then
    gdown --folder --output "${DATA_DIR}" "${DATA_URL}" || true
    gdown "${DATA_URL}" -O "${DATA_DIR}/download" || true
  else
    echo "[WARN] gdown not found; attempting to install it..."
    python -m pip install gdown
    gdown --folder --output "${DATA_DIR}" "${DATA_URL}" || true
    gdown "${DATA_URL}" -O "${DATA_DIR}/download" || true
  fi

  echo "[INFO] Data download complete (check ${DATA_DIR})."
else
  echo "[INFO] DATA_URL not set; skipping data download."
fi

echo "[SUCCESS] Setup complete. Activate the environment with: conda activate ${ENV_NAME}"
