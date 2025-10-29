import numpy as np
import sympy as sy
import scipy as sp
import numba as nb

import numpy.typing as npt

from typing import Callable, cast, TypeAlias, Tuple
from sympy import Expr

import mpi4py.MPI as MPI
import ufl 

import dolfinx
from dolfinx.fem import FunctionSpace

import qugar
from qugar.mesh.unfitted_cart_mesh import UnfittedCartMesh

import line_profiler

UnfittedDomain: TypeAlias = qugar.cpp.UnfittedImplDomain_2D | qugar.cpp.UnfittedImplDomain_3D
type SparseMatrix = sp.sparse._csr.csr_matrix
dtype = np.float64

from dataclasses import dataclass

@dataclass
class Quad:
    points: np.ndarray 
    weights: np.ndarray 

def make_unit_square_quadrature(n: int) -> Quad:
    """
    Create an n-point-per-direction Gauss–Legendre quadrature rule on [0,1]^2.
    
    Returns
    -------
    Quad
        Object with attributes:
        - points: (n^2, 2) array of (x, y) quadrature points
        - weights: (n^2,) array of weights
    """
    x1d, w1d = np.polynomial.legendre.leggauss(n)

    x1d = 0.5 * (x1d + 1.0)
    w1d = 0.5 * w1d

    x2d, y2d = np.meshgrid(x1d, x1d, indexing="xy")
    w2d = np.outer(w1d, w1d)

    points = np.column_stack([x2d.ravel(), y2d.ravel()])
    weights = w2d.ravel()

    return Quad(points=points, weights=weights)


def zero_function(X):
    return (0*X[0], 0*X[1])


@nb.njit 
def compute_K(weighted_basis, basis, coeffs, n_b):
    K = np.zeros((n_b, n_b, 2, 2))
    for i in nb.prange(n_b):
        for j in range(n_b):
            for k in range(2):
                for l in range(2):
                    val = 0.0
                    for m in range(weighted_basis.shape[1]):
                        for p in range(2):
                            for q in range(2):
                                val += weighted_basis[p, m, i] * basis[q, m, j] * coeffs[m, p, k, q, l]
                    K[i, j, k, l] = val
    return K

@nb.njit
def compute_K_core(basis_vals, weighted_vals, n_b, n_c):
    K_core = np.zeros((n_b, n_b, 2, 2, n_c))
    for i in nb.prange(n_b):
        for j in range(n_b):
            for k in range(2):
                for l in range(2):
                    for c in range(n_c):
                        val = 0.0
                        for m in range(weighted_vals.shape[0]):
                            val += basis_vals[k, m, i] * basis_vals[l, m, j] * weighted_vals[m, c]
                        K_core[i, j, k, l, c] = val
    return K_core

def create_facet_quadrature(
    facet_points: npt.NDArray[dtype],
    facet_weights: npt.NDArray[dtype],
    local_face_id: int,
) -> Tuple[npt.NDArray[dtype], npt.NDArray[dtype], npt.NDArray[dtype]]:
    """
    Create quadrature points, weights, and normals for a given local face id.

    Extends the points from the facet to the higher-dimensional space by adding
    the constant coordinate of the face.
    It also generated the (constant) normal vectors for the quadrature points.

    Note:
        The weights are not modified.

    Args:
        facet_points (npt.NDArray[dtype]): Array of points on the facet.
        facet_weights (npt.NDArray[dtype]): Array of weights for the quadrature points.
        local_face_id (int): Identifier for the facet, used to determine
            the constant direction and side.

    Returns:
        Tuple[npt.NDArray[dtype], npt.NDArray[dtype], npt.NDArray[dtype]]:
            - points: Array of quadrature points in the higher-dimensional space.
            - facet_weights: Array of weights for the quadrature points.
            - normals: Array of normal vectors for the quadrature points.
    """
    facet_dim = facet_points.shape[1]
    dim = facet_dim + 1
    dtype = facet_points.dtype

    const_dir = local_face_id // 2
    side = local_face_id % 2

    points = np.zeros((facet_points.shape[0], dim), dtype=dtype)

    points[:, const_dir] = dtype.type(side)
    local_dir = 0
    for dir in range(dim):
        if dir != const_dir:
            points[:, dir] = facet_points[:, local_dir]
            local_dir += 1

    normals = np.zeros_like(points)
    normals[:, const_dir] = dtype.type(1.0) if side == 1 else dtype.type(-1.0)

    return points, facet_weights, normals


class Elasticity:

    def __init__(
        self,
        E: float = 2.5,
        nu: float = 0.25,
        dim: int = 2,
        exterior_bc: list[tuple] = [],
        source: Callable = zero_function,
        K_model = None,
        M_model = None,
        bM_model = None,
        K_full_core: np.ndarray = None
    ) -> None:
        
        self.E = E
        self.nu = nu
        self.dim = dim

        self.exterior_bc = exterior_bc
        self.source = source

        self.mu = E/(2*(1+nu))
        self.lambda_ = (E*nu)/((1+nu)*(1-2*nu))

        self.K_model = K_model
        self.M_model = M_model
        self.bM_model = bM_model
        self.K_full_core = K_full_core

    def create_function_space(self, unf_mesh: UnfittedCartMesh, degree: int) -> FunctionSpace:

        V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree, (self.dim,)))

        return V

    def assemble_stiffness(self, unf_domain: UnfittedDomain, basis: Callable, coefficients: Callable, full_cell: bool = False) -> np.ndarray:

        n_quad_pts = 8

        if full_cell:
            quad = make_unit_square_quadrature(n_quad_pts)
        else:
            quad = qugar.cpp.create_quadrature(unf_domain, np.array([0]), n_quad_pts)

        basis_vals = np.array(basis.evaluate_derivative(quad.points))
        a_vals = coefficients(quad.points)
        n_b = basis_vals.shape[2]

        weighted_basis =  basis_vals * quad.weights[None,:,None]

        K = compute_K(weighted_basis, basis_vals, a_vals, n_b)
        K = K.transpose(0, 2, 1, 3)
        K = K.reshape(n_b * 2, n_b * 2)

        return K
    
    def assemble_mass(self, unf_domain: UnfittedDomain, basis: Callable, coefficients: Callable) -> np.ndarray:

        n_quad_pts = 8

        quad = qugar.cpp.create_quadrature(unf_domain, np.array([0]), n_quad_pts)

        basis_vals = np.array(basis.evaluate(quad.points))
        coeff_vals = np.abs(coefficients(quad.points))
        n_b = basis_vals.shape[1]
        M = np.zeros((n_b, n_b))

        weighted_coeff =  coeff_vals * quad.weights

        for i in range(n_b):
            for j in range(i, n_b):
                M[i, j] = np.sum(basis_vals[:,i] * basis_vals[:,j] * weighted_coeff)
                if i != j:
                    M[j, i] = M[i, j]

        I_d = np.eye(2)
        M = M[:, None, :, None] * I_d[None, :, None, :]
        M = M.reshape(n_b * 2, n_b * 2)

        return M
    
    def assemble_boundary_mass(self, unf_domain: UnfittedDomain, basis: Callable, coefficients: Callable) -> np.ndarray:
        
        n_quad_pts = 8
        n_b = basis.get_total_number_basis()

        bM = np.zeros((n_b, n_b))

        facets = [
            np.array([0]),
            np.array([1]),
            np.array([2]),
            np.array([3])
        ]

        indices = [1, 1, 0, 0]

        for face_id, index in zip(facets, indices):

            facet_quad = qugar.cpp.create_facets_quadrature_exterior_integral(
                unf_domain, 
                np.array([0]), 
                face_id, 
                n_quad_pts
            )

            points, weights, normals = create_facet_quadrature(facet_quad.points, facet_quad.weights, face_id[0])

            basis_vals = basis.evaluate(points)
            coeff_vals = coefficients(points)

            bM[:,:] += np.einsum("mi,mj,m->ij", basis_vals, basis_vals, coeff_vals[:,index] * weights)

        I_d = np.eye(2)
        bM = bM[:, None, :, None] * I_d[None, :, None, :]
        bM = bM.reshape(n_b * 2, n_b * 2)

        return bM
    
    def assemble_right_hand_side(self, unf_domain: UnfittedDomain, basis: Callable, coefficients: Callable, centers: list[np.ndarray]) -> np.ndarray:

        n_quad_pts = 8
        cut_cells_quad = qugar.cpp.create_quadrature(unf_domain, np.array([0]), n_quad_pts)

        basis_vals = np.array(basis.evaluate(cut_cells_quad.points))
        coeff_vals = coefficients(cut_cells_quad.points)
        f_vals = np.array(self.source(cut_cells_quad.points.T))

        f = np.einsum("mi,m,km->ik", basis_vals, coeff_vals * cut_cells_quad.weights, f_vals).flatten()
        
        facets = [
            np.array([0]),
            np.array([1]),
            np.array([2]),
            np.array([3])
        ]

        indices = [1, 1, 0, 0]

        for (type_, fun, marker, ind) in self.exterior_bc:

            if type_ == 1:

                for face_id, index, center in zip(facets, indices, centers):

                    if marker(center):

                        facet_quad = qugar.cpp.create_facets_quadrature_exterior_integral(
                            unf_domain, 
                            np.array([0]), 
                            face_id, 
                            n_quad_pts
                        )

                        points, weights, normals = create_facet_quadrature(facet_quad.points, facet_quad.weights, face_id[0])

                        basis_vals = basis.evaluate(points)
                        coeff_vals = coefficients(points)
                        f_vals = np.array(fun(points.T))

                        f += np.einsum("mi,m,km->ik", basis_vals, coeff_vals[:,index] * weights, f_vals).flatten()

        return f


    def assemble_stiffness_core(self, unf_domain: UnfittedDomain, basis: Callable, approx_basis: Callable, full_cell = False) -> np.ndarray:

        n_quad_pts = 10

        if full_cell:
            quad = make_unit_square_quadrature(n_quad_pts)
        else:
            quad = qugar.cpp.create_quadrature(unf_domain, np.array([0]), n_quad_pts)

        basis_vals = np.array(basis.evaluate_derivative(quad.points))
        approx_basis_vals = approx_basis.evaluate_basis(quad.points) 

        n_b = basis_vals.shape[2]
        n_c = approx_basis_vals.shape[1] 
        K_core = np.zeros((n_b, n_b, 2, 2, n_c))  

        weighted_vals = approx_basis_vals * quad.weights[:, None]
        K_core = compute_K_core(basis_vals, weighted_vals, n_b, n_c)
        # K_core = np.einsum("kmi,lmj,mc->ijklc", basis_vals, basis_vals, weighted_vals)
        # for i in range(n_b):
        #     for j in range(i, n_b):
        #         K_core[i, j, ...] = np.einsum("km,lm,mc->klc", basis_vals[:,:,i], basis_vals[:,:,j], weighted_vals)
        #         if i != j:
        #             K_core[j, i, ...] = K_core[i, j, ...]

        return K_core

    def assemble_mass_core(self, unf_domain: UnfittedDomain, basis: Callable, approx_basis: Callable) -> np.ndarray:

        n_quad_pts = 10

        cut_cells_quad = qugar.cpp.create_quadrature(unf_domain, np.array([0]), n_quad_pts)

        basis_vals = basis.evaluate(cut_cells_quad.points)
        approx_basis_vals = approx_basis.evaluate_basis(cut_cells_quad.points) 

        n_b = basis_vals.shape[1]
        n_c = approx_basis_vals.shape[1] 
        M_core = np.zeros((n_b, n_b, n_c))  

        weighted_vals = approx_basis_vals * cut_cells_quad.weights[:, None]
        for i in range(n_b):
            for j in range(i, n_b):
                M_core[i, j, :] = np.einsum("m,m,mc->c", basis_vals[:,i], basis_vals[:,j], weighted_vals)
                if i != j:
                    M_core[j, i, :] = M_core[i, j, :]

        return M_core
    
    def assemble_boundary_mass_core(self, unf_domain: UnfittedDomain, basis: Callable, approx_basis: Callable) -> np.ndarray:

        n_quad_pts = 10

        n_c = 9
        n_b = basis.get_total_number_basis()

        bM_core = np.zeros((n_b, n_b, n_c, 2))

        facets = [
            np.array([0]),
            np.array([1]),
            np.array([2]),
            np.array([3])
        ]

        indices = [1, 1, 0, 0]

        for face_id, index in zip(facets, indices):

            facet_quad = qugar.cpp.create_facets_quadrature_exterior_integral(
                unf_domain, 
                np.array([0]), 
                face_id, 
                n_quad_pts
            )

            points, weights, normals = create_facet_quadrature(facet_quad.points, facet_quad.weights, face_id[0])

            basis_vals = basis.evaluate(points)
            approx_basis_vals = approx_basis.evaluate_basis(points)

            bM_core[:,:,:,index] += np.einsum("mi,mj,mc->ijc", basis_vals, basis_vals, approx_basis_vals * weights[:, None])

        return bM_core
    


    