import numpy as np
import splipy as sp
 
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
import os

from scipy.io import loadmat

dtype = np.float64

def h_bc(X):
    return (0+0*X[0], 0+0*X[0])

def source(X):
    return (0+0*X[0], -0.01+0*X[0])

if __name__ == "__main__":            

    data  = loadmat('examples/wing_example/WingForce.mat')
    coefs_f = data['coefs']
    knt_f   = data['knt'].flatten()

    data  = loadmat('examples/wing_example/WingSection.mat')
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
        gyroid.SchoenIWP().make_function(),
        geometry_opts
    )

    basis_f = sp.BSplineBasis(3, knt_f)

    def boundary_force(X):
        basis_vals = basis_f.evaluate(X[0])
        return 10e4 * np.einsum("ij,kj->ik", coefs_f, basis_vals) 

    def parameter_function(X):
        return 6.3 * np.exp(-13 * X[1]) + 6.3 * np.exp(13 * (X[1]-1))
    
    geometry.coarse_mesh.set_parameter_field_from_function(parameter_function)

    # cells_ids = np.hstack(
    #     [np.arange(0, 4000), np.arange(3200, 4000)]
    # )

    # values = [np.array([6.3] * 4)] * 1600

    # geometry.coarse_mesh.set_parameter_field_values(cells_ids, values)

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
        nu = 0.3
    )

    sbdmn_opts = {
        "stabilize" : False,
        "stabilization": 1e-5,
        "assemble" : True
    }

    gdm_opts = {
        "periodic_mesh": True,
        "subdomain_opts" : sbdmn_opts
    }

    opts = {
        "make_plots": True,
        "global_dofs_manager_opts": gdm_opts
    }

    solver = BDDC(geometry, elasticity_pde, communicators, opts = opts)
    solver.setup()
    solver.solve()
    solver.plot_solution()

    








