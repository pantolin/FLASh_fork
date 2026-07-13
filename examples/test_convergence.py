"""
Compares convergence rates between cutFEM and p-FEM elements.
Useful for evaluating element technology choices in FLASh.
"""

from pathlib import Path

from mpi4py import MPI

import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.io
import numpy as np
import ufl

import qugar
import qugar.impl
from qugar.dolfinx import ds_bdry_unf, mapped_normal
from qugar.mesh import create_unfitted_impl_Cartesian_mesh
from qugar.utils import has_FEniCSx, has_PETSc

from FLASh.mesh import gyroid

try:
    import matplotlib.pyplot as plt

    has_matplotlib = True
except ImportError:
    has_matplotlib = False

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found, required for this demo.")
if not has_PETSc:
    raise ValueError("petsc4py installation not found, required for this demo.")

# -

# ### Domain Definition
# We define the domain as the unit square with a circular hole:

# +

### OUR PAPER GEOMETRY


# xmin = np.array([0.0, 0.0], dtype=np.float64)
# xmax = np.array([1.0, 1.0], dtype=np.float64)


# dtype = np.float64
# levelset = gyroid.SchwarzDiamond().make_function()
# impl_func = levelset(np.array(4 * [0.5]), xmin, xmax)


### A square with a cilyndrical hole


xmin = np.array([0.0, 0.0], dtype=np.float64)
xmax = np.array([1.0, 1.0], dtype=np.float64)
center = (xmin + xmax) / 2.0
R = 0.2

dtype = np.float64
impl_func = qugar.impl.create_negative(qugar.impl.create_disk(R, center=center))

# -

# ### Material Parameters
# Define the material properties:

# +

# Young's modulus and Poisson's ratio
E = 1.0
nu = 0.3

# Lamé parameters
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

print(f"Young's modulus E = {E}")
print(f"Poisson's ratio ν = {nu}")
print(f"Lamé parameter μ = {mu:.4f}")
print(f"Lamé parameter λ = {lmbda:.4f}")

# -

# ### Boundary Condition Method
# Choose between Nitsche's method (`True`) or Neumann boundary (`False`) conditions:

# +
use_Nistche = False
# -

# ### Analytical Solution
# Define the exact displacement solution:

# +


def epsilon_expr(u):
    """Strain tensor."""
    return ufl.sym(ufl.grad(u))


def sigma_expr(u):
    """Stress tensor."""
    # Get spatial dimension from the mesh (2D in this case)
    # For a 2D problem, Identity(2) creates a 2x2 identity tensor
    return 2.0 * mu * epsilon_expr(u) + lmbda * ufl.tr(epsilon_expr(u)) * ufl.Identity(2)

    # x, y = sy.symbols('x y')    

    # u1 = 0.1 * sy.sin(sy.pi * (x))**2 * sy.sin(sy.pi * (y))**2 * (-1+2*y)
    # u2 = 0.1 * sy.sin(sy.pi * (x))**2 * sy.sin(sy.pi * (y))**2 * (1-2*x)

def u_exact_expr(x):
    """Exact displacement solution as a UFL expression."""
    u1 = 0.1 * (ufl.sin(ufl.pi * x[0]) ** 2) * (ufl.sin(ufl.pi * x[1]) ** 2) * (-1+2*x[1])
    u2 = 0.1 * (ufl.sin(ufl.pi * x[0]) ** 2) * (ufl.sin(ufl.pi * x[1]) ** 2) * (1-2*x[0])
    return ufl.as_vector([u1, u2])


def u_exact_numpy(x):
    """Exact displacement solution as a numpy function for error computation."""
    u1 = 0.1 * (np.sin(np.pi * x[0]) ** 2) * (np.sin(np.pi * x[1]) ** 2) * (-1+2*x[1])
    u2 = 0.1 * (np.sin(np.pi * x[0]) ** 2) * (np.sin(np.pi * x[1]) ** 2) * (1-2*x[0])
    return np.array([u1, u2], dtype=dtype)


# -

# ### Boundary DOF Location
# Function to locate degrees of freedom on the outer boundary facets:

# +


def locate_boundary_dofs(unf_mesh, V):
    dim = unf_mesh.topology.dim

    # Left boundary (x=0)
    left_facets = dolfinx.mesh.locate_entities_boundary(
        unf_mesh, dim=(dim - 1), marker=lambda x: np.isclose(x[0], xmin[0])
    )
    left_dofs = dolfinx.fem.locate_dofs_topological(V=V, entity_dim=1, entities=left_facets)

    # Bottom boundary (y=0)
    bottom_facets = dolfinx.mesh.locate_entities_boundary(
        unf_mesh, dim=(dim - 1), marker=lambda x: np.isclose(x[1], xmin[1])
    )
    bottom_dofs = dolfinx.fem.locate_dofs_topological(V=V, entity_dim=1, entities=bottom_facets)

    # Right boundary (x=1)
    right_facets = dolfinx.mesh.locate_entities_boundary(
        unf_mesh, dim=(dim - 1), marker=lambda x: np.isclose(x[0], xmax[0])
    )
    right_dofs = dolfinx.fem.locate_dofs_topological(V=V, entity_dim=1, entities=right_facets)

    # Top boundary (y=1)
    top_facets = dolfinx.mesh.locate_entities_boundary(
        unf_mesh, dim=(dim - 1), marker=lambda x: np.isclose(x[1], xmax[1])
    )
    top_dofs = dolfinx.fem.locate_dofs_topological(V=V, entity_dim=1, entities=top_facets)

    return left_dofs, right_dofs, bottom_dofs, top_dofs


# -

# ### Solve Function
# Function to solve the elasticity problem for a given mesh resolution:

# +


def solve_elasticity(n_cells, degree=1):
    unf_mesh = create_unfitted_impl_Cartesian_mesh(
        MPI.COMM_WORLD,
        impl_func,
        n_cells,
        xmin,
        xmax,
        exclude_empty_cells=True,
        dtype=dtype,
    )

    dim = unf_mesh.topology.dim
    V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree, (dim,)))
    
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(unf_mesh)
    u_exact = u_exact_expr(x)
    sigma_u_exact = sigma_expr(u_exact)

    # Compute body force: f = -div(sigma(u_exact))
    f_exact = -ufl.div(sigma_u_exact)

    # Dirichlet boundary conditions on outer boundary
    u_D = dolfinx.fem.Function(V)
    u_D.interpolate(lambda x: u_exact_numpy(x))
    bcs = [dolfinx.fem.dirichletbc(u_D, dofs) for dofs in locate_boundary_dofs(unf_mesh, V)]

    dx = ufl.dx(domain=unf_mesh)
    n_unf = mapped_normal(unf_mesh)
    ds_unf = ds_bdry_unf(domain=unf_mesh)

    # Bilinear form: a(u,v) = ∫_Ω σ(u) : ε(v) dx
    a = ufl.inner(sigma_expr(u), epsilon_expr(v)) * dx

    if use_Nistche:
        # Bilinear form: a(u,v) = ∫_Ω σ(u) : ε(v) dx
        # - ∫_Γ_N (σ(u)·n) · v ds - ∫_Γ_N u · (σ(v)·n) ds
        # + β/h ∫_Γ_N u · v ds
        h = np.linalg.norm(xmax - xmin) / n_cells
        beta = 30 * degree**2
        sigma_u_n = ufl.dot(sigma_expr(u), n_unf)
        sigma_v_n = ufl.dot(sigma_expr(v), n_unf)
        a -= ufl.dot(sigma_u_n, v) * ds_unf  # consistency
        a -= ufl.dot(u, sigma_v_n) * ds_unf  # symmetry
        a += beta / h * ufl.dot(u, v) * ds_unf  # stability

    # Linear form: L(v) = ∫_Ω f · v dx + ...
    L = ufl.dot(f_exact, v) * dx

    if use_Nistche:
        # Linear form: L(v) = ∫_Ω f · v dx
        # - ∫_Γ_N u_exact · (σ(v)·n) ds + β/h ∫_Γ_N u_exact · v ds
        sigma_v_n = ufl.dot(sigma_expr(v), n_unf)
        L += beta / h * ufl.dot(u_exact, v) * ds_unf
        L -= ufl.dot(u_exact, sigma_v_n) * ds_unf
    else:  # Neumann
        # Linear form: L(v) = ∫_Ω f · v dx + ∫_Γ_N (σ(u_exact)·n) · v ds
        sigma_u_exact_n = ufl.dot(sigma_u_exact, n_unf)
        L += ufl.dot(sigma_u_exact_n, v) * ds_unf

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "cholesky",
        "ksp_diagonal_scale": True,  # Jacobi preconditioner
    }

    problem = qugar.dolfinx.LinearProblem(a, L, bcs=bcs, petsc_options=petsc_options)
    uh = problem.solve()

    return unf_mesh, uh, V


# -

# ### Error Computation
# Function to compute L² and H¹ error norms for vector fields:

# +

def compute_l2_h1_errors(uh, unf_mesh):
    """Compute L² error: ||u_h - u_exact||_L² and H¹ error: ||u_h - u_exact||_H¹"""
    x = ufl.SpatialCoordinate(unf_mesh)
    u_exact = u_exact_expr(x)
    error = uh - u_exact
    grad_error = ufl.grad(error)

    dx = ufl.dx(domain=unf_mesh)

    # Compute L² norm using appropriate quadrature
    L2_error_form = ufl.dot(error, error) * dx
    L2_error_form = qugar.dolfinx.form_custom(L2_error_form)
    L2_error = np.sqrt(
        dolfinx.fem.assemble_scalar(L2_error_form, coeffs=L2_error_form.pack_coefficients())
    )

    # H¹ semi norm = grad L² (Frobenius norm of gradient)
    semi_H1_error_form = ufl.inner(grad_error, grad_error) * dx
    semi_H1_error_form = qugar.dolfinx.form_custom(semi_H1_error_form)

    semi_H1_error = np.sqrt(
        dolfinx.fem.assemble_scalar(
            semi_H1_error_form, coeffs=semi_H1_error_form.pack_coefficients()
        )
    )
    # H¹ norm = L² + grad L²
    H1_error = L2_error + semi_H1_error

    return L2_error, H1_error

# -

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
h_mesh_sizes = [2, 4, 6, 8, 10, 12, 14]  # Mesh refinement for low degrees
standard_degrees = [1, 2, 3]
spectral_degrees = [6, 8, 10]
all_degrees = standard_degrees + spectral_degrees

results_data = {}

print("\n" + "=" * 80)
print("CONVERGENCE STUDY: h-Refinement (1,2,3) vs. Single-Element p-Refinement (6,8,10)")
print("=" * 80)

for deg in all_degrees:
    results_data[deg] = {"h": [], "n_dofs": [], "L2_error": []}
    
    # Logic change: If degree is 6, 8, or 10, only solve for a single element
    if deg in spectral_degrees:
        current_iterations = [1] # Single element (mesh size 1)
        print(f"\nDegree {deg}: Evaluating Single Element (p-refinement mode)")
    else:
        current_iterations = h_mesh_sizes
        print(f"\nDegree {deg}: Evaluating Mesh Sequence (h-refinement mode)")

    for n_cells in current_iterations:
        # solve_elasticity and compute_l2_h1_errors are external calls
        mesh, uh, V = solve_elasticity(n_cells, degree=deg)
        
        h = 1.0 / n_cells
        n_dofs = V.dofmap.index_map.size_global
        L2_err, _ = compute_l2_h1_errors(uh, mesh)
        
        results_data[deg]["h"].append(h)
        results_data[deg]["n_dofs"].append(n_dofs)
        results_data[deg]["L2_error"].append(L2_err)
        
        print(f"  Mesh: {n_cells:3d}x{n_cells:3d} | DOFs: {n_dofs:7d} | L2 Error: {L2_err:.2e}")

# --- Plotting ---
fig, ax = plt.subplots(figsize=(10, 7))

# Define distinct markers/colors for clarity
styles = {1: 'o-', 2: 's-', 3: 'd-', 6: 'P', 8: 'X', 10: '*'}
colors = plt.cm.plasma(np.linspace(0, 0.8, len(all_degrees)))

for i, deg in enumerate(all_degrees):
    dofs = np.array(results_data[deg]["n_dofs"])
    errors = np.array(results_data[deg]["L2_error"])
    
    # Plotting against DOFs is the only way to compare a single element vs a mesh
    ax.loglog(dofs, errors, styles[deg], color=colors[i], 
              markersize=10, label=f'Degree {deg}' + (" (Single Elem)" if deg >= 6 else ""))

ax.set_xlabel("Degrees of Freedom (Total DOFs)", fontsize=12)
ax.set_ylabel("$L^2$ Error Norm", fontsize=12)
ax.set_title("Efficiency Comparison: $h$-refinement vs. High-Order Single Element", fontsize=14, fontweight='bold')
ax.grid(True, which="both", ls="--", alpha=0.5)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 7))

# Define distinct markers/colors for clarity
styles = {1: 'o-', 2: 's-', 3: 'd-'} # Markers for lines
spectral_markers = {6: 'P', 8: 'X', 10: '*'}
colors = plt.cm.plasma(np.linspace(0, 0.8, len(all_degrees)))

for i, deg in enumerate(all_degrees):
    h_vals = np.array(results_data[deg]["h"])
    errors = np.array(results_data[deg]["L2_error"])
    
    if deg in standard_degrees:
        # Standard h-refinement lines
        ax.loglog(h_vals, errors, styles[deg], color=colors[i], 
                  linewidth=2, markersize=8, label=f'Degree {deg}')
    else:
        # Spectral single-element results shown as horizontal reference lines
        # We use the error value from the single element (index 0)
        err_value = errors[0]
        ax.axhline(y=err_value, color=colors[i], linestyle='--', alpha=0.8,
                   label=f'Degree {deg} (Single Elem)')
        
        # Also plot the specific point at h=1 for clarity
        ax.plot(1.0, err_value, spectral_markers[deg], color=colors[i], markersize=12)

# Formatting the plot
ax.set_xlabel("Mesh size $h$", fontsize=12)
ax.set_ylabel("$L^2$ Error Norm", fontsize=12)
ax.set_title("h-Convergence vs. Spectral Error Thresholds", fontsize=14, fontweight='bold')

# Invert X-axis so that 'h' decreasing (refinement) goes from left to right
# Or leave as is; standard log-log plots usually have h decreasing to the left
ax.set_xlim(max(h_mesh_sizes)**-1 * 0.5, 1.5) 

ax.grid(True, which="both", ls="--", alpha=0.5)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# |
# | Convergence plot |
# |:---:|
# | ![Convergence plot](assets/demo_elasticity_convergence.png) |
