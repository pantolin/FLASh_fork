import numpy as np
from pathlib import Path

from FLASh.utils import Communicators

from FLASh.rom import MDEIM

from FLASh.mesh import (
    GlobalDofsManager,
    SplineGeometry,
    gyroid
)
from FLASh.pde import (
    Elasticity,
    BDDC,
    Cholesky
)

dtype = np.float64

import h5py

# Paths
import os
from _paths import RESULTS_DIR, ROM_DATA_DIR

if __name__ == "__main__":         

    communicators = Communicators()

    ### Load ROM models ###
    
    epsilon_min = 0.1
    epsilon_max = 0.9

    n_rom = 2
    p_rom = 6
    d_rom = 4

    p0 = np.array([epsilon_min] * d_rom)
    p1 = np.array([epsilon_max] * d_rom)

    k_core_model = MDEIM(n_rom, p_rom, p0, p1)
    k_core_model.set_up_from_files(str(ROM_DATA_DIR / "schwarz_diamond_3" / "K_core"))

    m_core_model = MDEIM(n_rom, p_rom, p0, p1)
    m_core_model.set_up_from_files(str(ROM_DATA_DIR / "schwarz_diamond_3" / "M_core"))

    bm_core_model = MDEIM(n_rom, p_rom, p0, p1)
    bm_core_model.set_up_from_files(str(ROM_DATA_DIR / "schwarz_diamond_3" / "bM_core"))

    K_core_full = np.load(str(ROM_DATA_DIR / "schwarz_diamond_3" / "K_core" / "full_array.npy"))

    ### Set geometry options ###

    P0 = np.array([0.0, 0.0])
    P1 = np.array([1.0, 1.0])

    def map(x, y, r = [0.6, 1.0], theta = [1.5, 2.0]):

        tx = theta[0] + (theta[1]-theta[0])*x
        ty = r[0] + (r[1]-r[0])*y

        return np.stack([ty*np.cos(np.pi*tx), ty*np.sin(np.pi*tx), 0*tx], axis=-1)

    def parameter_function(X):
        val = 0.9 - 0.8*X[0]
        return np.clip(val, 0.1, 0.9)
    
    basis_degree = 8
    spline_degree = 2
    
    geometry_opts = {
            "basis_degree": basis_degree,
            "spline_degree": spline_degree,
            "periodic": False
        }

    ### Set solver source and boundary conditions ###

    def source(X):
        return (0.0+0.0*X[0], 0.0+0.0*X[0])
    
    def h_bc(X):
        return (0.0+0.0*X[0], 0.0+0.0*X[0])

    def nh_bc(X):
        return (0.0+0.0*X[0], -0.1+0.0*X[0])
    
    exterior_bc = [
        (
            0, 
            h_bc, 
            lambda x: np.isclose(x[0], P0[0]), 
            0
        ),
        (
            1, 
            nh_bc, 
            lambda x: np.isclose(x[0], P1[0]), 
            0
        )
    ]

    ### Set pde problems ###
    
    elasticity_pde = Elasticity(
        exterior_bc = exterior_bc,
        source = source,
        E = 5,
        nu = 0.25
    )

    elasticity_pde_rom = Elasticity(
        exterior_bc = exterior_bc,
        source = source,
        E = 5,
        nu = 0.25,
        K_model = k_core_model,
        M_model = m_core_model,
        bM_model = bm_core_model,
        K_full_core = K_core_full
    )

    ### Set options ###

    stabilizations = [0.0, 1e-5, 1e-4, 5e-4, 1e-3, 1e-2]
    stabilize = True

    i_max = 10

    stab_errors = np.empty((i_max-1, len(stabilizations)))
    rom_errors = np.empty((i_max-1, len(stabilizations)))
    total_errors = np.empty((i_max-1, len(stabilizations)))

    iterations = np.empty((i_max-1, len(stabilizations)))
    rom_iterations = np.empty((i_max-1, len(stabilizations)))

    for i in range(1, i_max):

        ### Create geometry ###

        n = [4*i, 2*i]

        P0 = np.array([0.0, 0.0])
        P1 = np.array([1.0, 1.0])

        knots_x = [P0[0]]*spline_degree + list(np.linspace(P0[0],P1[0],n[0]+1)) + [P1[0]]*spline_degree
        knots_y = [P0[1]]*spline_degree + list(np.linspace(P0[1],P1[1],n[1]+1)) + [P1[1]]*spline_degree

        geometry = SplineGeometry.interpolate_map(
            [knots_x, knots_y],
            map,
            gyroid.SchwarzDiamond().make_function(),
            geometry_opts
        )

        geometry.coarse_mesh.set_parameter_field_from_function(parameter_function)

        ### Get baseline solution ###

        sbdmn_opts = {
            "stabilize" : False,
            "stabilization": 0.0,
            "assemble" : True
        }

        gdm_opts = {
            "subdomain_opts" : sbdmn_opts
        }

        opts = {
            "global_dofs_manager_opts": gdm_opts
        }


        solver = BDDC(geometry, elasticity_pde, communicators, opts = opts)
        solver.setup()
        solver.solve()

        baseline_solution = solver.get_solution()

        for idx, stabilization in enumerate(stabilizations):

            ### Set solver options ###

            sbdmn_opts = {
                "stabilize" : stabilize,
                "stabilization": stabilization,
                "assemble" : True
            }

            gdm_opts = {
                "subdomain_opts" : sbdmn_opts
            }

            opts = {
                "global_dofs_manager_opts": gdm_opts
            }

            ### Solve with rom ###

            solver = BDDC(geometry, elasticity_pde_rom, communicators, opts = opts)
            solver.setup()
            solver.solve()

            rom_stats = solver.get_stats()
            rom_solution = solver.get_solution()

            ### Solve without rom ###

            solver = BDDC(geometry, elasticity_pde, communicators, opts = opts)
            solver.setup()
            solver.solve()

            no_rom_stats = solver.get_stats()
            no_rom_solution = solver.get_solution()

            ### Compare solutions ###

            stab_error = solver.gbl_dofs_mngr.compute_error(no_rom_solution, baseline_solution)
            rom_error = solver.gbl_dofs_mngr.compute_error(rom_solution, no_rom_solution)
            total_error = solver.gbl_dofs_mngr.compute_error(rom_solution, baseline_solution)

            stab_errors[i-1, idx] = stab_error
            rom_errors[i-1, idx] = rom_error
            total_errors[i-1, idx] = total_error

            iterations[i-1, idx] = no_rom_stats["iterations"][0]
            rom_iterations[i-1, idx] = rom_stats["iterations"][0]

            if communicators.global_comm.Get_rank() == 0:
                print(f"Errors: {stab_error}, {rom_error}, {total_error}.")    


    if communicators.global_comm.Get_rank() == 0:

        number_of_subdomains = 8 * (np.arange(1, i_max) ** 2)
        stabilizations = np.array(stabilizations)

        folder = os.path.join(RESULTS_DIR, "test_2")
        os.makedirs(folder, exist_ok=True)

        file_path = os.path.join(folder, f"data.h5")

        with h5py.File(file_path, "w") as f:

            f.create_dataset("iterations", data=iterations)
            f.create_dataset("rom_iterations", data=rom_iterations)

            f.create_dataset("stab_errors", data=stab_errors)
            f.create_dataset("rom_errors", data=rom_errors)
            f.create_dataset("total_errors", data=total_errors)

            f.create_dataset("number_of_subdomains", data=number_of_subdomains)
            f.create_dataset("stabilizations", data=stabilizations)

        print(f"Saved to {file_path}")



        








