import numpy as np

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
import os

from _paths import RESULTS_DIR

if __name__ == "__main__":         

    communicators = Communicators()

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

    ### Set options ###

    fa_degrees = [1, 2, 3, 4, 5]

    i_max = 10

    errors = np.empty((i_max-1, len(fa_degrees)))

    iterations = np.empty((i_max-1))
    fa_iterations = np.empty((i_max-1, len(fa_degrees)))

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


        baseline_solver = BDDC(geometry, elasticity_pde, communicators, opts = opts)
        baseline_solver.setup()
        baseline_solver.solve()

        baseline_solution = baseline_solver.get_solution()
        baseline_stats = baseline_solver.get_stats()

        iterations[i-1] = baseline_stats["iterations"][0]

        for idx, degree in enumerate(fa_degrees):

            ### Set solver options ###

            sbdmn_opts = {
                "approximate_geometry" : True,
                "approximate_geometry_degree": degree,
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

            ### Solve with fa ###

            solver = BDDC(geometry, elasticity_pde, communicators, opts = opts)
            solver.setup()
            solver.solve()

            fa_stats = solver.get_stats()
            solution = solver.get_solution()

            ### Compare solutions ###

            error = baseline_solver.gbl_dofs_mngr.compute_error(solution, baseline_solution)

            errors[i-1, idx] = error

            fa_iterations[i-1, idx] = fa_stats["iterations"][0]

            if communicators.global_comm.Get_rank() == 0:
                print(f"Errors: {error}.")    


    if communicators.global_comm.Get_rank() == 0:

        number_of_subdomains = 8 * (np.arange(1, i_max) ** 2)
        fa_degrees = np.array(fa_degrees)

        folder = RESULTS_DIR / "test_6"
        folder.mkdir(parents=True, exist_ok=True)

        file_path = folder / "data.h5"

        with h5py.File(file_path, "w") as f:

            f.create_dataset("iterations", data=iterations)
            f.create_dataset("fa_iterations", data=fa_iterations)

            f.create_dataset("errors", data=errors)

            f.create_dataset("number_of_subdomains", data=number_of_subdomains)
            f.create_dataset("fa_degrees", data=fa_degrees)

        print(f"Saved to {file_path}")



        








