"""
Reproduces Figure 10 (Section 5.1.1, "Accuracy of the stabilization").
"""

from pathlib import Path
import sys
import os
import uuid
import shutil
import gc
from _paths import FIGS_DIR

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- Top-level flags ---
RUN_ANALYSIS = True
RUN_PLOTS = True

try:
    import matplotlib.pyplot as plt

    has_matplotlib = True
except ImportError:
    has_matplotlib = False

if RUN_ANALYSIS:
    from mpi4py import MPI
    from petsc4py import PETSc

    import dolfinx.fem
    import dolfinx.fem.petsc
    import ufl
    from dolfinx.la import create_petsc_vector_wrap

    import qugar
    import qugar.impl
    import qugar.cpp
    from qugar.dolfinx import ds_bdry_unf, mapped_normal
    from qugar.mesh import create_unfitted_impl_Cartesian_mesh
    from qugar.utils import has_FEniCSx, has_PETSc

    if not has_FEniCSx:
        raise ValueError("FEniCSx installation not found, required for this test.")
    if not has_PETSc:
        raise ValueError("petsc4py installation not found, required for this test.")


# --- Global configuration ---
xmin = np.array([0.0, 0.0], dtype=np.float64)
xmax = np.array([1.0, 1.0], dtype=np.float64)
dtype = np.float64

POLY_DEGREES = list(range(4,21))
STABILIZATION_VALUES = [1e-2,1e-3,1e-4,1e-5, 0]
N_CELLS = 1
QUAD_FACTOR = 4

RESULTS_FILE = Path(__file__).parent / "stabilization_geometry_p_sweep_results.npz"

if RUN_ANALYSIS:
    E = 1.0
    nu = 0.3
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def epsilon_expr(u):
        return ufl.sym(ufl.grad(u))

    def sigma_expr(u):
        eps = epsilon_expr(u)
        return 2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(2)

    def u_exact_expr(x):
        u1 = 0.1 * (ufl.sin(ufl.pi * x[0]) ** 2) * (ufl.sin(ufl.pi * x[1]) ** 2) * (-1 + 2 * x[1])
        u2 = 0.1 * (ufl.sin(ufl.pi * x[0]) ** 2) * (ufl.sin(ufl.pi * x[1]) ** 2) * (1 - 2 * x[0])
        return ufl.as_vector([u1, u2])

    def u_exact_numpy(x):
        u1 = 0.1 * (np.sin(np.pi * x[0]) ** 2) * (np.sin(np.pi * x[1]) ** 2) * (-1 + 2 * x[1])
        u2 = 0.1 * (np.sin(np.pi * x[0]) ** 2) * (np.sin(np.pi * x[1]) ** 2) * (1 - 2 * x[0])
        return np.array([u1, u2], dtype=dtype)

    def locate_boundary_dofs(unf_mesh, V):
        dim = unf_mesh.topology.dim
        left_facets = dolfinx.mesh.locate_entities_boundary(
            unf_mesh, dim=(dim - 1), marker=lambda x: np.isclose(x[0], xmin[0])
        )
        bottom_facets = dolfinx.mesh.locate_entities_boundary(
            unf_mesh, dim=(dim - 1), marker=lambda x: np.isclose(x[1], xmin[1])
        )
        right_facets = dolfinx.mesh.locate_entities_boundary(
            unf_mesh, dim=(dim - 1), marker=lambda x: np.isclose(x[0], xmax[0])
        )
        top_facets = dolfinx.mesh.locate_entities_boundary(
            unf_mesh, dim=(dim - 1), marker=lambda x: np.isclose(x[1], xmax[1])
        )

        left_dofs = dolfinx.fem.locate_dofs_topological(V=V, entity_dim=1, entities=left_facets)
        right_dofs = dolfinx.fem.locate_dofs_topological(V=V, entity_dim=1, entities=right_facets)
        bottom_dofs = dolfinx.fem.locate_dofs_topological(V=V, entity_dim=1, entities=bottom_facets)
        top_dofs = dolfinx.fem.locate_dofs_topological(V=V, entity_dim=1, entities=top_facets)

        return left_dofs, right_dofs, bottom_dofs, top_dofs

    def solve_elasticity(impl_func, n_cells, degree, stabilization):
        cache_dir = None
        try:
            cache_dir = os.path.join("/tmp/fenicsx_cache", uuid.uuid4().hex)
            os.makedirs(cache_dir, exist_ok=True)
            jit_options = {"cache_dir": cache_dir}

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
            f_exact = -ufl.div(sigma_u_exact)

            u_D = dolfinx.fem.Function(V)
            u_D.interpolate(lambda x: u_exact_numpy(x))
            bcs = [dolfinx.fem.dirichletbc(u_D, dofs) for dofs in locate_boundary_dofs(unf_mesh, V)]

            dx = ufl.dx(domain=unf_mesh)
            ds_unf = ds_bdry_unf(domain=unf_mesh)
            n_unf = mapped_normal(unf_mesh)

            sigma_u_exact_n = ufl.dot(sigma_u_exact, n_unf)
            L_expr = ufl.dot(f_exact, v) * dx + ufl.dot(sigma_u_exact_n, v) * ds_unf
            a_expr = ufl.inner(sigma_expr(u), epsilon_expr(v)) * dx

            petsc_options = {
                "ksp_type": "preonly",
                "pc_type": "cholesky",
                "ksp_diagonal_scale": True,
            }

            if stabilization == 0.0:
                # No stabilization (or below float64 machine epsilon): use qugar LinearProblem.
                # Values like 1e-100 are equivalent to 0 in float64 arithmetic since
                # 1.0 - 1e-100 == 1.0, so they must take this path for consistent results.
                problem = qugar.dolfinx.LinearProblem(
                    a_expr, L_expr, bcs=bcs, petsc_options=petsc_options,
                    jit_options=jit_options,
                )
                uh = problem.solve()
                return unf_mesh, uh

            # Düster–Rank alpha-stabilization over the fictitious domain:
            #   A_total = A_active + alpha * A_fictitious
            #           = (1 - alpha) * A_active + alpha * A_full
            # A_active → qugar form_custom  (active domain, custom quadrature on cut cells)
            # A_full   → dolfinx.fem.form   (standard Gauss on full reference cell)
            a_active_form = qugar.dolfinx.form_custom(
                (1.0 - stabilization) * a_expr, dtype=PETSc.ScalarType, jit_options=jit_options
            )
            a_full_form = dolfinx.fem.form(
                stabilization * a_expr, dtype=PETSc.ScalarType, jit_options=jit_options
            )
            L_form = qugar.dolfinx.form_custom(L_expr, dtype=PETSc.ScalarType, jit_options=jit_options)

            A = dolfinx.fem.petsc.create_matrix(a_active_form)
            b = dolfinx.fem.petsc.create_vector(L_form)

            A.zeroEntries()
            a_active_coeffs = a_active_form.pack_coefficients()
            dolfinx.fem.petsc.assemble_matrix(A, a_active_form, coeffs=a_active_coeffs)
            dolfinx.fem.petsc.assemble_matrix(A, a_full_form)
            A.assemble()
            # Zero BC rows/cols across the combined matrix, then set diagonal to 1.
            # Must be done after A.assemble() so both forms' contributions are present;
            # passing bcs only to the second assemble_matrix call would leave the
            # first form's off-diagonal BC entries intact.
            bc_dofs = np.sort(np.unique(np.concatenate([bc.dof_indices()[0] for bc in bcs])))
            A.zeroRowsColumns(bc_dofs.tolist(), 1.0)

            # Assemble RHS and apply lifting for non-zero Dirichlet data.
            with b.localForm() as b_loc:
                b_loc.set(0)
            b_coeffs = L_form.pack_coefficients()
            dolfinx.fem.petsc.assemble_vector(b, L_form, coeffs=b_coeffs)
            dolfinx.fem.petsc.apply_lifting(b, [a_active_form], bcs=[bcs], coeffs=[a_active_coeffs])
            dolfinx.fem.petsc.apply_lifting(b, [a_full_form], bcs=[bcs])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            for bc in bcs:
                bc.set(b.array_w)

            solver = PETSc.KSP().create(unf_mesh.comm)
            solver.setOperators(A)
            problem_prefix = f"dolfinx_solve_{id(A)}"
            solver.setOptionsPrefix(problem_prefix)
            opts = PETSc.Options()
            opts.prefixPush(problem_prefix)
            for k, v in petsc_options.items():
                opts[k] = v
            opts.prefixPop()
            solver.setFromOptions()

            uh = dolfinx.fem.Function(V)
            solver.solve(b, create_petsc_vector_wrap(uh.x))
            uh.x.scatter_forward()

            A.destroy()
            b.destroy()
            solver.destroy()

            return unf_mesh, uh
        finally:
            if cache_dir is not None and os.path.exists(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)
            gc.collect()

    def compute_normalized_l2_error(uh, unf_mesh, degree):
        cache_dir = None
        try:
            cache_dir = os.path.join("/tmp/fenicsx_cache", uuid.uuid4().hex)
            os.makedirs(cache_dir, exist_ok=True)
            jit_options = {"cache_dir": cache_dir}

            x = ufl.SpatialCoordinate(unf_mesh)
            u_exact = u_exact_expr(x)

            dx = ufl.dx(domain=unf_mesh)
            quad_opts = {"quadrature_degree": degree * QUAD_FACTOR}
            err_form = qugar.dolfinx.form_custom(
                ufl.dot(uh - u_exact, uh - u_exact) * dx,
                form_compiler_options=quad_opts,
                jit_options=jit_options,
            )
            ref_form = qugar.dolfinx.form_custom(
                ufl.dot(u_exact, u_exact) * dx,
                form_compiler_options=quad_opts,
                jit_options=jit_options,
            )

            l2_error = np.sqrt(dolfinx.fem.assemble_scalar(err_form, coeffs=err_form.pack_coefficients()))
            l2_ref = np.sqrt(dolfinx.fem.assemble_scalar(ref_form, coeffs=ref_form.pack_coefficients()))
            return l2_error / l2_ref
        finally:
            if cache_dir is not None and os.path.exists(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)
            gc.collect()


if RUN_ANALYSIS:
    GEOMETRIES = [
        ("schwarz_diamond", "Schwarz Diamond",
         qugar.impl.ImplicitFunc(qugar.cpp.create_functions_addition(
             qugar.impl.create_Schwarz_Diamond(periods=[1, 1], z=0.0).cpp_object,
             qugar.impl.create_constant(-0.1, dim=2).cpp_object,
         ))),
        ("schwarz_primitive", "Schwarz Primitive",
         qugar.impl.create_Schwarz_Primitive(periods=[1, 1], z=0.5)),
        ("schoen_frd", "Schoen FRD",
         qugar.impl.create_negative(qugar.impl.create_Schoen_FRD(periods=[1, 1], z=0))),
        ("schoen_iwp", "Schoen IWP",
         qugar.impl.create_Schoen_IWP(periods=[1, 1], z=0)),
    ]

    print("\n" + "=" * 90)
    print("STABILIZATION STUDY: normalized L2 displacement error vs polynomial degree")
    print("=" * 90)

    errors = np.zeros((len(GEOMETRIES), len(STABILIZATION_VALUES), len(POLY_DEGREES)), dtype=np.float64)

    for g_idx, (geom_key, geom_label, impl_func) in enumerate(GEOMETRIES):
        print(f"\nGeometry: {geom_label}")

        for s_idx, stabilization in enumerate(STABILIZATION_VALUES):
            print(f"  Stabilization = {stabilization:.0e}")

            for p_idx, degree in enumerate(POLY_DEGREES):
                mesh, uh = solve_elasticity(
                    impl_func=impl_func,
                    n_cells=N_CELLS,
                    degree=degree,
                    stabilization=stabilization,
                )
                rel_l2_error = compute_normalized_l2_error(uh, mesh, degree)
                errors[g_idx, s_idx, p_idx] = rel_l2_error
                print(f"    p = {degree:2d} | rel L2 error = {rel_l2_error:.3e}")

    np.savez(
        RESULTS_FILE,
        geometry_keys=np.array([g[0] for g in GEOMETRIES]),
        geometry_labels=np.array([g[1] for g in GEOMETRIES]),
        stabilizations=np.array(STABILIZATION_VALUES, dtype=np.float64),
        degrees=np.array(POLY_DEGREES, dtype=np.int32),
        rel_l2_errors=errors,
    )
    print(f"\n[INFO] Results saved to {RESULTS_FILE}")


if RUN_PLOTS:
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Results file not found: {RESULTS_FILE}. Run with RUN_ANALYSIS=True first.")

    data = np.load(RESULTS_FILE)
    geometry_keys = data["geometry_keys"]
    geometry_labels = data["geometry_labels"]
    stabilizations = data["stabilizations"]
    degrees = data["degrees"]
    rel_l2_errors = data["rel_l2_errors"]
    print(f"[INFO] Results loaded from {RESULTS_FILE}")

    if not has_matplotlib:
        raise ImportError("matplotlib is required for plotting but is not installed.")

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 22,
            "legend.fontsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )

    colors = ["blue", "red", "orange", "green", "brown", "purple"]
    markers = ["o", "s", "^", "d", "v", "p"]

    HELPING_DIR = FIGS_DIR / "helping"
    GEOM_IMAGES = {
        "schwarz_diamond":  HELPING_DIR / "schwarz.png",
        "schwarz_primitive": HELPING_DIR / "schwarz_primitive.png",
        "schoen_frd":       HELPING_DIR / "schoen_frd.png",
        "schoen_iwp":       HELPING_DIR / "schoen_iwp.png",
    }

    def _stab_label(s):
        if s == 0.0:
            return r"$\rho=0$"
        exp = int(round(np.log10(s)))
        return rf"$\rho=10^{{{exp}}}$"

    for g_idx, geom_key in enumerate(geometry_keys):
        fig, ax = plt.subplots(figsize=(8.5, 6.5))

        for s_idx, stabilization in enumerate(stabilizations):
            curve = rel_l2_errors[g_idx, s_idx, :]
            mask = np.isfinite(curve)
            if stabilization == 0.0:
                mask &= (degrees <= 13)
            ax.semilogy(
                degrees[mask],
                curve[mask],
                color=colors[s_idx],
                marker=markers[s_idx],
                linewidth=2.0,
                markersize=6.5,
                markerfacecolor="white",
                markeredgewidth=1.3,
            )

        ax.set_xlabel(r"Polynomial order $p$")
        ax.set_ylabel(r"$L^2$ error")
        ax.set_xlim(3, 21)
        ax.set_ylim(1e-9, 1e0)
        ax.grid(True, which="major", alpha=0.25)

        img_path = GEOM_IMAGES.get(str(geom_key))
        if img_path and img_path.exists():
            img = plt.imread(str(img_path))
            axins = ax.inset_axes([0.68, 0.02, 0.30, 0.30])
            axins.imshow(img)
            axins.axis("off")

        plt.tight_layout()
        plt.savefig(str(FIGS_DIR / f"stabilization_pconv_{geom_key}.pdf"), bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    # Standalone horizontal legend figure for the stabilization parameter rho
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0],
               color=colors[s_idx],
               marker=markers[s_idx],
               linewidth=2.0,
               markersize=6.5,
               markerfacecolor="white",
               markeredgewidth=1.3,
               label=_stab_label(float(stabilizations[s_idx])))
        for s_idx in range(len(stabilizations))
    ]
    labels = [_stab_label(float(s)) for s in stabilizations]

    fig_leg, ax_leg = plt.subplots(figsize=(len(stabilizations) * 1.4, 0.65))
    ax_leg.axis("off")
    leg = ax_leg.legend(
        handles, labels,
        loc="center",
        ncol=len(stabilizations),
        frameon=False,
        fontsize=16,
        handlelength=1.6,
        columnspacing=1.2,
        borderpad=0.0,
        borderaxespad=0.0,
    )
    fig_leg.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig_leg.canvas.draw()
    bbox = leg.get_window_extent().transformed(fig_leg.dpi_scale_trans.inverted())
    plt.savefig(str(FIGS_DIR / "stabilization_pconv_legend.pdf"), bbox_inches=bbox, pad_inches=0.08)
    plt.close(fig_leg)
    print("[INFO] Legend figure saved to stabilization_rho_legend.pdf")
