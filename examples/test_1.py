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
    AMG,
    Cholesky
)

import h5py

from _paths import RESULTS_DIR

dtype = np.float64

if __name__ == "__main__":         

    communicators = Communicators()

    ### Simulation paramters ###

    stabilize = False
    stabilization = 0

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

    ### Set pde problem ###

    elasticity_pde = Elasticity(
        exterior_bc = exterior_bc,
        source = source,
        E = 5,
        nu = 0.25
    )

    cholesky_stats = []
    bddc_stats = []
    amg_stats = []

    i_max = 10
    for i in range(1, i_max):

        ### Create geometry ###

        n = [4*i, 2*i]

        knots_x = [P0[0]]*spline_degree + list(np.linspace(P0[0],P1[0],n[0]+1)) + [P1[0]]*spline_degree
        knots_y = [P0[1]]*spline_degree + list(np.linspace(P0[1],P1[1],n[1]+1)) + [P1[1]]*spline_degree

        geometry = SplineGeometry.interpolate_map(
            [knots_x, knots_y],
            map,
            gyroid.SchwarzDiamond().make_function(),
            geometry_opts
        )

        geometry.coarse_mesh.set_parameter_field_from_function(parameter_function)

        ### Solve baseline problem with direct solver ###

        GlobalDofsManager.plot(geometry, communicators)
        solver = Cholesky(geometry, elasticity_pde, communicators, opts = opts)
        solver.setup()
        solver.solve()
        solver.plot_solution()

        cholesky_stats.append(solver.get_stats())

        ### Solve basiline problem with BDDC solver ###

        solver = BDDC(geometry, elasticity_pde, communicators, opts = opts)
        solver.setup()
        solver.solve()

        bddc_stats.append(solver.get_stats())

        ### Solve basiline problem with AMG solver ###

        solver = AMG(geometry, elasticity_pde, communicators, opts = opts)
        solver.setup()
        solver.solve()

        amg_stats.append(solver.get_stats())


    if communicators.global_comm.Get_rank() == 0:

        bddc_iters = np.array([stats["iterations"][0] for stats in bddc_stats])
        amg_iters = np.array([stats["iterations"] for stats in amg_stats])

        bddc_setup_time = np.array([stats["assemble time"] for stats in bddc_stats])
        amg_setup_time = np.array([stats["assemble time"] for stats in amg_stats])
        cholesky_setup_time = np.array([stats["assemble time"] for stats in cholesky_stats])

        bddc_solve_time = np.array([stats["solve time"] for stats in bddc_stats])
        amg_solve_time = np.array([stats["solve time"] for stats in amg_stats])
        cholesky_solve_time = np.array([stats["solve time"] for stats in cholesky_stats])

        number_of_subdomains = 8 * (np.arange(1, i_max) ** 2)

        folder = RESULTS_DIR / "test_1"
        folder.mkdir(parents=True, exist_ok=True)

        file_path = folder / "data.h5"

        with h5py.File(file_path, "w") as f:

            f.create_dataset("bddc_iters", data=bddc_iters)
            f.create_dataset("amg_iters", data=amg_iters)

            f.create_dataset("bddc_setup_time", data=bddc_setup_time)
            f.create_dataset("amg_setup_time", data=amg_setup_time)
            f.create_dataset("cholesky_setup_time", data=cholesky_setup_time)

            f.create_dataset("bddc_solve_time", data=bddc_solve_time)
            f.create_dataset("amg_solve_time", data=amg_solve_time)
            f.create_dataset("cholesky_solve_time", data=cholesky_solve_time)

            f.create_dataset("number_of_subdomains", data=number_of_subdomains)

        print(f"Saved to {file_path}")






    








