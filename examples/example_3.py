import numpy as np
import splipy as sp
from pathlib import Path
 
from FLASh.utils import Communicators

import sys
sys.stdout.isatty = lambda: True

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

from FLASh.rom import (
    MDEIM
)

from scipy.io import loadmat

# Paths
from _paths import EXAMPLES_ROOT, ROM_DATA_DIR, RESULTS_DIR

dtype = np.float64

def h_bc(X):
    return (0+0*X[0], 0+0*X[0])

def source(X):
    return (0+0*X[0], -0.01+0*X[0])

if __name__ == "__main__":      

    ### Load ROM models ###
    
    epsilon_min = 0.1
    epsilon_max = 1.0

    n_rom = 2
    p_rom = 6
    d_rom = 4

    p0 = np.array([epsilon_min] * d_rom)
    p1 = np.array([epsilon_max] * d_rom)

    k_core_model = MDEIM(n_rom, p_rom, p0, p1)
    k_core_model.set_up_from_files(str(ROM_DATA_DIR / "schwarz_diamond_4" / "K_core"))

    m_core_model = MDEIM(n_rom, p_rom, p0, p1)
    m_core_model.set_up_from_files(str(ROM_DATA_DIR / "schwarz_diamond_4" / "M_core"))

    bm_core_model = MDEIM(n_rom, p_rom, p0, p1)
    bm_core_model.set_up_from_files(str(ROM_DATA_DIR / "schwarz_diamond_4" / "bM_core"))

    K_core_full = np.load(str(ROM_DATA_DIR / "schwarz_diamond_4" / "K_core" / "full_array.npy"))

    #### 

    data  = loadmat(str(EXAMPLES_ROOT / "wing_example" / "WingForce.mat"))
    coefs_f = data['coefs']
    knt_f   = data['knt'].flatten()

    data  = loadmat(str(EXAMPLES_ROOT / "wing_example" / "WingSection.mat"))
    coefs = data['coefs'].transpose(1, 2, 0)
    coefs = np.concatenate([coefs, np.zeros((*coefs.shape[:2], 1))], axis=-1)
    knt1  = data['knt1'].flatten()
    knt2  = data['knt2'].flatten()

    communicators = Communicators()

    geometry_opts = {
        "basis_degree": 8,
        "spline_degree": 2,
        "periodic": True
    }

    geometry = SplineGeometry.create_spline(
        [knt1, knt2],
        coefs,
        gyroid.SchwarzDiamond().make_function(),
        geometry_opts
    )

    basis_f = sp.BSplineBasis(3, knt_f)

    def boundary_force(X):
        basis_vals = basis_f.evaluate(X[0])
        return 5e4 * np.einsum("ij,kj->ik", coefs_f, basis_vals) 

    def parameter_function(X):
        vals = 0.1 + 0.2 + np.exp(-13 * (X[1])) + 1.0 * np.exp(13 * (X[1]-1))
        return np.clip(vals, 0.1, 1.0)
    
    geometry.coarse_mesh.set_parameter_field_from_function(parameter_function)

    # GlobalDofsManager.plot(geometry, communicators)

    exterior_bc = [
        (
            0, 
            h_bc, 
            lambda x: np.logical_and(
                np.logical_and(
                    np.less_equal(x[0], 0.3820),
                    np.greater_equal(x[0], 0.3260)
                ),
                np.isclose(x[1], 0)
            ), 
            0
        ),
        (
            0, 
            h_bc, 
            lambda x: np.logical_and(
                np.logical_and(
                    np.less_equal(x[0], 0.6740),
                    np.greater_equal(x[0], 0.6180)
                ),
                np.isclose(x[1], 0)
            ), 
            1
        ),
        (
            1, 
            boundary_force, 
            lambda x: np.isclose(x[1], 1), 
            0
        )
    ]
    
    elasticity_pde = Elasticity(
        exterior_bc = exterior_bc,
        source = source,
        E = 70 * 1e9,
        nu = 0.3,
        K_model = k_core_model,
        M_model = m_core_model,
        bM_model = bm_core_model,
        K_full_core = K_core_full
    )

    sbdmn_opts = {
        "stabilize" : True,
        "stabilization": 1e-3,
        "parametric_bc": True,
        "assemble" : True,
    }

    gdm_opts = {
        "periodic_mesh": True,
        "subdomain_opts" : sbdmn_opts
    }

    opts = {
        "make_plots": True,
        "global_dofs_manager_opts": gdm_opts
    }

    # GlobalDofsManager.plot(geometry, communicators)
    solver = BDDC(geometry, elasticity_pde, communicators, opts = opts)
    solver.setup()
    solver.solve()
    solver.plot_solution()
    # solver.write_solution(str(RESULTS_DIR / "wing_example"))

    








