import numpy as np
from pathlib import Path

from FLASh.utils import Communicators

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

from FLASh.rom import MDEIM

dtype = np.float64

# Paths
from _paths import ROM_DATA_DIR

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

    ### Simulation paramters ###

    stabilization = 5e-4
    stabilize = True

    ### Set pde problems ###

    elasticity_pde = Elasticity(
        exterior_bc = exterior_bc,
        source = source,
        E = 5,
        nu = 0.25,
        K_model = k_core_model,
        M_model = m_core_model,
        bM_model = bm_core_model,
        K_full_core = K_core_full
    )

    ### Set solver options ###

    sbdmn_opts = {
        "stabilize" : True,
        "stabilization": stabilization,
        "assemble" : True
    }

    gdm_opts = {
        "subdomain_opts" : sbdmn_opts
    }

    opts = {
        "global_dofs_manager_opts": gdm_opts
    }

    i = 10
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
    
    ### Solve with BDDC ###

    solver = BDDC(geometry, elasticity_pde, communicators, opts = opts)
    solver.setup()
    solver.solve()
    solver.plot_solution()