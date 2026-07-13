"""
Compares convergence rates between cutFEM and p-FEM elements.
Useful for evaluating element technology choices in FLASh.

Solves one (degree, n_cells) configuration and prints its result as JSON.
Run as a subprocess per configuration by plot_convergence_test.py, with
PYTHONHASHSEED fixed: no stabilization is used here (matching the paper),
so the assembled system for high-degree/finely-cut configurations is
ill-conditioned, and Python's per-process hash randomization (which affects
dict/set iteration order somewhere in the qugar/FFCx assembly pipeline) can
change floating-point summation order enough to flip the solution between
runs. Fixing the hash seed makes results reproducible.

Reproduces Figure 9 (Section 5.1.1, "Accuracy of the p-FEM discretization").
"""

import argparse
import json

from mpi4py import MPI

import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.mesh
import numpy as np
import ufl

import qugar
import qugar.dolfinx
from qugar.dolfinx import ds_bdry_unf, mapped_normal
from qugar.mesh import create_unfitted_impl_Cartesian_mesh
from qugar.utils import has_FEniCSx, has_PETSc

from FLASh.mesh import gyroid

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found, required for this demo.")
if not has_PETSc:
    raise ValueError("petsc4py installation not found, required for this demo.")

dtype = np.float64

# Single Schoen FRD cell, threshold 0, identity map (paper Section 5.1.1,
# "Accuracy of the p-FEM discretization"; level-set shown in Figure 2c).
xmin = np.array([0.0, 0.0], dtype=dtype)
xmax = np.array([1.0, 1.0], dtype=dtype)
levelset = gyroid.SchoenFRD().make_function()
impl_func = levelset(np.array(4 * [0.0]), xmin, xmax)

# Lame constants both set to unity, as in the paper.
mu = 1.0
lmbda = 1.0

# Manufactured solution amplitude.
u0 = 0.1


def epsilon_expr(u):
    """Strain tensor."""
    return ufl.sym(ufl.grad(u))


def sigma_expr(u):
    """Stress tensor."""
    return 2.0 * mu * epsilon_expr(u) + lmbda * ufl.tr(epsilon_expr(u)) * ufl.Identity(2)


def u_exact_expr(x):
    """Exact displacement solution as a UFL expression."""
    u1 = u0 * ufl.sin(ufl.pi * x[0]) ** 2 * ufl.sin(ufl.pi * x[1]) ** 2 * (2 * x[1] - 1)
    u2 = u0 * ufl.sin(ufl.pi * x[0]) ** 2 * ufl.sin(ufl.pi * x[1]) ** 2 * (1 - 2 * x[0])
    return ufl.as_vector([u1, u2])


def u_exact_numpy(x):
    """Exact displacement solution as a numpy function, for the Dirichlet data."""
    u1 = u0 * np.sin(np.pi * x[0]) ** 2 * np.sin(np.pi * x[1]) ** 2 * (2 * x[1] - 1)
    u2 = u0 * np.sin(np.pi * x[0]) ** 2 * np.sin(np.pi * x[1]) ** 2 * (1 - 2 * x[0])
    return np.array([u1, u2], dtype=dtype)


def locate_boundary_dofs(unf_mesh, V):
    """Locate the DOFs on the four sides of the outer (conformal) boundary."""
    dim = unf_mesh.topology.dim

    sides = [
        lambda x: np.isclose(x[0], xmin[0]),
        lambda x: np.isclose(x[0], xmax[0]),
        lambda x: np.isclose(x[1], xmin[1]),
        lambda x: np.isclose(x[1], xmax[1]),
    ]

    return [
        dolfinx.fem.locate_dofs_topological(
            V=V, entity_dim=dim - 1,
            entities=dolfinx.mesh.locate_entities_boundary(unf_mesh, dim=dim - 1, marker=side),
        )
        for side in sides
    ]


def solve_elasticity(n_cells, degree):
    """Solve the manufactured-solution problem on an n_cells x n_cells grid of degree-p cells.

    Dirichlet conditions from the exact solution are imposed on the outer
    (conformal) boundary, and Neumann tractions from the exact stress field
    are imposed on the trimmed boundary, so that the discrete solution can be
    compared against the manufactured solution regardless of trimming.
    """

    unf_mesh = create_unfitted_impl_Cartesian_mesh(
        MPI.COMM_WORLD, impl_func, n_cells, xmin, xmax, exclude_empty_cells=True, dtype=dtype,
    )

    dim = unf_mesh.topology.dim
    V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree, (dim,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(unf_mesh)
    u_exact = u_exact_expr(x)
    sigma_u_exact = sigma_expr(u_exact)
    f_exact = -ufl.div(sigma_u_exact)

    u_D = dolfinx.fem.Function(V)
    u_D.interpolate(u_exact_numpy)
    bcs = [dolfinx.fem.dirichletbc(u_D, dofs) for dofs in locate_boundary_dofs(unf_mesh, V)]

    dx = ufl.dx(domain=unf_mesh)
    ds_unf = ds_bdry_unf(domain=unf_mesh)
    n_unf = mapped_normal(unf_mesh)

    a = ufl.inner(sigma_expr(u), epsilon_expr(v)) * dx
    L = ufl.dot(f_exact, v) * dx + ufl.dot(ufl.dot(sigma_u_exact, n_unf), v) * ds_unf

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "cholesky",
        "ksp_diagonal_scale": True,
    }

    problem = qugar.dolfinx.LinearProblem(a, L, bcs=bcs, petsc_options=petsc_options)
    uh = problem.solve()

    n_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    nnz = int(problem.A.getInfo()["nz_used"])

    return unf_mesh, uh, n_dofs, nnz


def compute_l2_error(uh, unf_mesh):
    """Compute the L2 error between uh and the exact solution, normalized by
    the L2 norm of the exact solution (as done throughout this section)."""
    x = ufl.SpatialCoordinate(unf_mesh)
    u_exact = u_exact_expr(x)
    error = uh - u_exact
    dx = ufl.dx(domain=unf_mesh)

    error_form = qugar.dolfinx.form_custom(ufl.dot(error, error) * dx)
    error_norm = np.sqrt(dolfinx.fem.assemble_scalar(error_form, coeffs=error_form.pack_coefficients()))

    exact_form = qugar.dolfinx.form_custom(ufl.dot(u_exact, u_exact) * dx)
    exact_norm = np.sqrt(dolfinx.fem.assemble_scalar(exact_form, coeffs=exact_form.pack_coefficients()))

    return error_norm / exact_norm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, required=True)
    parser.add_argument("--n-cells", type=int, required=True)
    args = parser.parse_args()

    unf_mesh, uh, n_dofs, nnz = solve_elasticity(args.n_cells, args.degree)
    L2_error = compute_l2_error(uh, unf_mesh)
    print(json.dumps({"n_dofs": n_dofs, "nnz": nnz, "L2_error": L2_error}))
