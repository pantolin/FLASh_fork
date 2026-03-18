"""Shared path helpers for the examples.

All example scripts should use these helpers to compute paths relative to the
examples directory. This makes the repository easy to reorganize without having
to update many hard-coded relative paths.
"""

from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parent
DATA_ROOT = EXAMPLES_ROOT / "data"
FIGS_DIR = DATA_ROOT / "figs"
RESULTS_DIR = DATA_ROOT / "results"
ROM_DATA_DIR = DATA_ROOT / "rom_data"