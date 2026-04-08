"""
Evaluates the accuracy of reduced-order modeling (ROM) for a single cell using FLASh.
Compares errors between exact and ROM-based fast assembly tensors (with and without interpolation),
computes stiffness ROM error, and solves a simple problem to assess ROM performance.
"""

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
    compute_magic_points,
    create_RBF_interpolator,
    interpolate_coefficients
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

class RBFInterpolator:

    def __init__(self, interpolator):

        self.interpolator = interpolator

    def evaluate(self, x):

        return interpolate_coefficients(self.interpolator, x).T

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

    ##########

    k_core_model_rbf = MDEIM(n_rom, p_rom, p0, p1)
    k_core_model_rbf.set_up_from_files(str(ROM_DATA_DIR / "schwarz_diamond_3" / "K_core"))

    for i, lag_interpolator in enumerate(k_core_model._interpolators):

        points = lag_interpolator.get_nodes()
        values = lag_interpolator._weights

        rbf_interpolator = create_RBF_interpolator(points, values)
        k_core_model_rbf._interpolators[i] = RBFInterpolator(rbf_interpolator)


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
    )

    pts = geometry.coarse_mesh.get_cell_vertex_points(0)
    bezier_element = geometry.get_bezier_element(0)

    no_int_error = 0
    lag_int_error = 0
    rbf_int_error = 0

    lag_time = 0
    rbf_time = 0

    total_samples = 20

    for _ in range(total_samples):

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

        K_core = sbdmn.assemble_K_core()

        idx = int(k_core_model.locate_point(np.squeeze(parameters)).item())
        rom_basis = k_core_model._basis[idx]

        I = compute_magic_points(rom_basis)

        S = K_core[None, :]
        S = S.reshape((S.shape[0], -1)).T
        coeffs = compute_deim_coefficients(rom_basis, I, S.T)

        no_int_K_core_rom = compute_aproximations(rom_basis, coeffs).reshape(K_core.shape)

        st = MPI.Wtime()

        K_core_rom = k_core_model.evaluate(parameters.reshape(1,4)).reshape(K_core.shape)

        lag_time += MPI.Wtime() - st
        st = MPI.Wtime()

        K_core_rom_rbf = k_core_model_rbf.evaluate(parameters.reshape(1,4)).reshape(K_core.shape)

        rbf_time += MPI.Wtime() - st

        lag_int_error += np.linalg.norm(K_core-K_core_rom)/np.linalg.norm(K_core)
        rbf_int_error += np.linalg.norm(K_core-K_core_rom_rbf)/np.linalg.norm(K_core)
        no_int_error += np.linalg.norm(K_core-no_int_K_core_rom)/np.linalg.norm(K_core)

    print(f"MDEIM error comparison for n = {total_samples} samples.")
    print(f"Error with no interpolation: {no_int_error/total_samples}")
    print(f"Error with lagrange interpolation: {lag_int_error/total_samples}")
    print(f"Error with rbf interpolation: {rbf_int_error/total_samples}")

    print(f"Lagrange model times: {lag_time/total_samples}")
    print(f"RBF time: {rbf_time/total_samples}")

