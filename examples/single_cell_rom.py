import numpy as np
from pathlib import Path

from mpi4py import MPI  

from FLASh.utils import Communicators
from FLASh.rom import MDEIM

from FLASh.mesh import (
    Subdomain,
    SplineGeometry,
    gyroid
)
from FLASh.pde import (
    Elasticity
)
from FLASh.rom import (
    compute_deim_coefficients,
    compute_aproximations,
    compute_magic_points
)

dtype = np.float64

# Paths
from _paths import ROM_DATA_DIR

def map(x, y):

    return np.stack([x, y, 0*x], axis=-1)

def compute_operator_error(A, B):

    error = np.linalg.norm(A-B)
    norm = np.linalg.norm(A)

    print("Error: ", error, ". Relative error: ", error/norm, ".\n")

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

    K_core = k_core_model.evaluate(np.ones((1,4)))
    M_core = m_core_model.evaluate(np.ones((1,4)))
    bM_core = bm_core_model.evaluate(np.ones((1,4)))

    ### Create geometry ###

    basis_degree = 8
    spline_degree = 2

    n = [2, 2]

    P0 = np.array([0.0, 0.0])
    P1 = np.array([1.0, 1.0])

    knots_x = [P0[0]]*spline_degree + list(np.linspace(P0[0],P1[0],n[0]+1)) + [P1[0]]*spline_degree
    knots_y = [P0[1]]*spline_degree + list(np.linspace(P0[1],P1[1],n[1]+1)) + [P1[1]]*spline_degree

    geometry_opts = {
        "basis_degree": basis_degree,
        "spline_degree": spline_degree,
        "periodic": False
    }

    geometry = SplineGeometry.interpolate_map(
        [knots_x, knots_y],
        map,
        gyroid.SchwarzDiamond().make_function(),
        geometry_opts
    )

    ### Assemble subdomain ###

    sbdmn_opts = {
        "stabilize" : False,
        "stabilization": 0.0,
        "assemble" : True
    }

    def source(X):
        return (0.0+0.0*X[0], 0.0+0*X[0])
    
    elasticity_pde = Elasticity(
        source = source,
        E = 5,
        nu = 0.25,
        K_model = k_core_model,
        M_model = m_core_model,
        bM_model = bm_core_model,
        K_full_core = K_core_full
    )

    pts = geometry.coarse_mesh.get_cell_vertex_points(0)
    bezier_element = geometry.get_bezier_element(0)

    for _ in range(100):

        parameters = epsilon_min + (epsilon_max - epsilon_min) * np.random.rand(1, 4)

        sbdmn = Subdomain(
            [1, 1], 
            geometry.basis_degree, 
            2,
            pts[0], 
            pts[3], 
            np.squeeze(parameters),
            geometry.levelset,
            elasticity_pde,
            bezier_element,
            opts = sbdmn_opts
        )

        K = sbdmn.assemble_K()
        K_rom = sbdmn.K

        M = sbdmn.assemble_M()

        K_core = sbdmn.assemble_K_core()

        idx = int(k_core_model.locate_point(np.squeeze(parameters)))
        rom_basis = k_core_model._basis[idx]

        I = compute_magic_points(rom_basis)

        S = K_core[None, :]
        S = S.reshape((S.shape[0], -1)).T
        coeffs = compute_deim_coefficients(rom_basis, I, S.T)

        no_int_K_core_rom = compute_aproximations(rom_basis, coeffs).reshape(K_core.shape)
        K_core_rom = k_core_model.evaluate(parameters.reshape(1,4)).reshape(K_core.shape)

        print(np.linalg.norm(K_core-K_core_rom)/np.linalg.norm(K_core))
        print(np.linalg.norm(K_core-no_int_K_core_rom)/np.linalg.norm(K_core))

    def compute_solution_error(A, B):

        n = K.shape[0]

        a_dofs = sbdmn.all_dofs
        e_dofs = sbdmn.edges_dofs

        b1_dofs = e_dofs[0]
        b2_dofs = e_dofs[3]

        i_dofs = np.setdiff1d(a_dofs, np.union1d(b1_dofs, b2_dofs))

        ub = np.zeros(n)
        ub[b2_dofs] = -0.001

        f = np.zeros(n)
        f -= A @ ub

        u = np.zeros(n)
        u[i_dofs] = np.linalg.solve(A[i_dofs][:,i_dofs], f[i_dofs])

        f_rom = np.zeros(n)
        f_rom -= B @ ub

        u_rom = np.zeros(n)
        u_rom[i_dofs] = np.linalg.solve(B[i_dofs][:,i_dofs], f_rom[i_dofs])

        error = np.linalg.norm(u-u_rom)
        norm = np.linalg.norm(u)

        print("Error: ", error, ". Relative error: ", error/norm, ".")

        error = np.sqrt((u-u_rom).T @ M @ (u-u_rom))
        norm = np.sqrt(u.T @ M @ u)

        print("Error: ", error, ". Relative error: ", error/norm, ".\n")

    compute_operator_error(K, K_rom)
    compute_solution_error(K, K_rom)





        









    








