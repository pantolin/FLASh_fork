"""
Script for generating reduced-order models (ROM) and snapshots using the FLASh framework.
Configures parameters and calls routines to build and save ROM data for various microstructures.
Intended for offline ROM preparation and dataset creation.
"""
import numpy as np
from pathlib import Path

from FLASh.mesh import gyroid
from FLASh.rom import (
    generate_snapshots,
    generate_rom_model
)

# Paths
from _paths import ROM_DATA_DIR

if __name__ == "__main__":
    
    generate_rom_model(
        "K_core", 
        "schoen_iwp_3", 
        levelset = gyroid.SchoenIWP().make_function(), 
        epsilon_0 = -2.5, 
        epsilon_1 = 2.5, 
        n = 2, 
        p = 6, 
        d = 4, 
        samples_per_basis = 100,
        batch_size = 100,
        basis_size = 40,
        basis_oversample = 20,
        directory = str(ROM_DATA_DIR)
    )

    # full_K_core = generate_snapshots(
    #     np.array([1] * 4),
    #     levelset = gyroid.SchoenIWP().make_function(),
    #     get_full_K_core = True
    # )

    # np.save(str(ROM_DATA_DIR / "schoen_iwp_2" / "K_core" / "full_array.npy"), full_K_core)

    # generate_rom_model(
    #     "M_core", 
    #     "schoen_iwp_2", 
    #     levelset = gyroid.SchoenIWP().make_function(), 
    #     epsilon_0 = -2.5, 
    #     epsilon_1 = 3.0, 
    #     n = 2, 
    #     p = 2, 
    #     d = 4, 
    #     samples_per_basis = 100,
    #     batch_size = 100,
    #     basis_size = 40,
    #     basis_oversample = 10,
    #     directory = str(ROM_DATA_DIR)
    # )

    # generate_rom_model(
    #     "bM_core", 
    #     "schoen_iwp_3", 
    #     levelset = gyroid.SchoenIWP().make_function(), 
    #     epsilon_0 = -2.5, 
    #     epsilon_1 = 3.0, 
    #     n = 1, 
    #     p = 6, 
    #     d = 4, 
    #     samples_per_basis = 25,
    #     batch_size = 100,
    #     basis_size = 10,
    #     basis_oversample = 5,
    #     directory = str(ROM_DATA_DIR)
    # )

