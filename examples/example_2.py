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

def map(x, y, r = [0.6, 1.0], theta = [0, 2]):

    tx = theta[0] + (theta[1]-theta[0])*x
    ty = r[0] + (r[1]-r[0])*y

    return np.stack([ty*np.cos(np.pi*tx), ty*np.sin(np.pi*tx), 0*tx], axis=-1)


if __name__ == "__main__":            

    P0 = np.array([0.0, 0.0])
    P1 = np.array([1.0, 1.0])

    degree = 8

    epsilon_min = 0.5
    epsilon_max = 0.5

    communicators = Communicators()

    def parameter_function(X):
        return  np.exp(-13 * X[1]) + np.exp(13 * (X[1]-1))

    def source(X):
        return (0.0+0.0*X[0], 0.0+0*X[0])

    def nh_bc(X):
        return (0*X[0], 0.1 * np.exp(-((X[0])/0.4) ** 2))
    
    #####

    exterior_bc = [
        (
            0, 
            bc_1, 
            lambda x: np.isclose(x[1], P0[1]), 
            0
        ),
        (
            1, 
            nh_bc, 
            lambda x: np.logical_and(
                np.logical_and(
                    np.less_equal(x[0], 1.0),
                    np.greater_equal(x[0], 0.5)
                ),
                np.isclose(x[1], P1[1])
            ), 
            1
        )
    ]
    
    elasticity_pde = Elasticity(
        exterior_bc = exterior_bc,
        source = source,
        # E = 70e9,
        # nu = 0.3
    )

    sbdmn_opts = {
        "stabilize" : True,
        "stabilization": 1e-5,
        "assemble" : True
    }

    gdm_opts = {
        "periodic_mesh": True,
        "subdomain_opts" : sbdmn_opts
    }

    opts = {
        "global_dofs_manager_opts": gdm_opts
    }

    i = 3
    n = [15*i, 2*i]

    knots_x = [P0[0]]*degree + list(np.linspace(P0[0],P1[0],n[0]+1)) + [P1[0]]*degree
    knots_y = [P0[1]]*degree + list(np.linspace(P0[1],P1[1],n[1]+1)) + [P1[1]]*degree

    geometry_opts = {
        "basis_degree": 8,
        "spline_degree": 8,
        "periodic": True
    }

    geometry = SplineGeometry.interpolate_map(
        [knots_x, knots_y],
        map,
        gyroid.SchwarzDiamond().make_function(),
        geometry_opts
    )

    geometry.coarse_mesh.set_parameter_field_from_function(parameter_function)

    # if communicators.global_comm.Get_rank() == 0:
    #     geometry.plot()
    #     geometry.plot_det()
    #     geometry.plot_arclen()

    # communicators.global_comm.Barrier()

    GlobalDofsManager.plot(geometry, communicators)
    # solver = BDDC(geometry, elasticity_pde, communicators, opts = opts)
    # solver.setup()
    # solver.solve()
    # solver.plot_solution()

    








