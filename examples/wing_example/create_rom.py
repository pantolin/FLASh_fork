import numpy as np

from FLASh.mesh import gyroid
from FLASh.rom import (
    generate_snapshots,
    generate_rom_model
)

if __name__ == "__main__":
    
    generate_rom_model(
        "K_core", 
        "schwarz_diamond_1", 
        levelset = gyroid.SchwarzDiamond().make_function(), 
        epsilon_0 = 0.1, 
        epsilon_1 = 0.9, 
        n = 1, 
        p = 1, 
        d = 4, 
        samples_per_basis = 20,
        batch_size = 100,
        basis_size = 10,
        basis_oversample = 5,
        directory = "rom_data"
    )

    full_K_core = generate_snapshots(
        np.array([1] * 4),
        levelset = gyroid.SchwarzDiamond().make_function(),
        get_full_K_core = True
    )

    np.save("rom_data/schwarz_diamond_1/K_core/full_array.npy", full_K_core)

    generate_rom_model(
        "M_core", 
        "schwarz_diamond_1", 
        levelset = gyroid.SchwarzDiamond().make_function(), 
        epsilon_0 = 0.1, 
        epsilon_1 = 0.9, 
        n = 1, 
        p = 1, 
        d = 4, 
        samples_per_basis = 20,
        batch_size = 100,
        basis_size = 10,
        basis_oversample = 5,
        directory = "rom_data"
    )

    generate_rom_model(
        "bM_core", 
        "schwarz_diamond_1", 
        levelset = gyroid.SchwarzDiamond().make_function(), 
        epsilon_0 = 0.1, 
        epsilon_1 = 0.9, 
        n = 1, 
        p = 1, 
        d = 4, 
        samples_per_basis = 20,
        batch_size = 100,
        basis_size = 10,
        basis_oversample = 5,
        directory = "rom_data"
    )

