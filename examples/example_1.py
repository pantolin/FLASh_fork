import numpy as np

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

dtype = np.float64

def bc_1(X):
    return (0.0*X[0], 0*X[1])

def bc_2(X):
    return (0.0+0.0*X[0], 0+0*X[0])

def map(x, y):

    return np.stack([x, y, 0*x], axis=-1)


if __name__ == "__main__":        

    P0 = np.array([0.0, 0.0])
    P1 = np.array([3.0, 3.0])

    degree = 8

    epsilon_min = 0.5
    epsilon_max = 0.5

    communicators = Communicators()

    def parameter_function(X):
        return 3 - 0*X[0]

    def source(X):
        return (0.0+0.0*X[0], 0.0+0*X[0])

    def nh_bc(X):
        return (0.2 + 0*X[0], 0*X[0])
    
    #####

    exterior_bc = [
        (
            0, 
            bc_1, 
            lambda x: np.isclose(x[0], P0[0]), 
            0
        ),
        (
            0, 
            nh_bc, 
            lambda x: np.isclose(x[0], P1[0]), 
            1
        ),
    ]
    
    elasticity_pde = Elasticity(
        exterior_bc = exterior_bc,
        source = source,
        E = 5,
        nu = 0.0
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

    n = [3, 3]

    knots_x = [P0[0]]*degree + list(np.linspace(P0[0],P1[0],n[0]+1)) + [P1[0]]*degree
    knots_y = [P0[1]]*degree + list(np.linspace(P0[1],P1[1],n[1]+1)) + [P1[1]]*degree

    geometry_opts = {
        "basis_degree": 8,
        "spline_degree": 8,
        "periodic": False
    }

    geometry = SplineGeometry.interpolate_map(
        [knots_x, knots_y],
        map,
        gyroid.SchoenIWP().make_function(),
        geometry_opts
    )

    geometry.coarse_mesh.set_parameter_field_from_function(parameter_function)

    GlobalDofsManager.plot(geometry, communicators)
    # solver = Cholesky(geometry, elasticity_pde, communicators, opts = opts)
    # solver.setup()
    # solver.solve()
    # solver.plot_solution()

    








