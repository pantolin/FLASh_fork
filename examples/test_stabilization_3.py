"""
Reproduces Figure 11(b) (Section 5.1.1, "Accuracy of the stabilization").
"""

from pathlib import Path
import sys
import os
import uuid
import shutil
import gc
from _paths import FIGS_DIR, RESULTS_DIR

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- Top-level flags ---
RUN_ANALYSIS = True
RUN_PLOTS = True
RUN_IMAGES = True

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
    from qugar.mesh import create_unfitted_impl_Cartesian_mesh
    from qugar.utils import has_FEniCSx, has_PETSc

    if not has_FEniCSx:
        raise ValueError("FEniCSx installation not found.")
    if not has_PETSc:
        raise ValueError("petsc4py installation not found.")


# --- Global configuration ---
xmin = np.array([0.0, 0.0], dtype=np.float64)
xmax = np.array([1.0, 1.0], dtype=np.float64)
dtype = np.float64

# Mapping parameter: x = 0.5 + (xi1-0.5)*(1 + (a-1)*(2*xi2 - xi2^2)),  y = xi2
# a=1 is the identity map; a=0.05 gives a strongly non-uniform deformation.
DEGREE = 14
STABILIZATION_VALUES = [1e-2, 1e-3, 1e-4, 1e-5, 0]
N_CELLS = 1
QUAD_FACTOR = 4
N_A_POINTS = 20
A_VALUES = np.linspace(0.1, 2.0, N_A_POINTS)

RESULTS_FILE = RESULTS_DIR / "stabilization_geometry_mapstudy_results.npz"

if RUN_ANALYSIS:
    E = 1.0
    nu = 0.3
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def u_exact_expr(x):
        """Exact displacement; x is a 2-vector of UFL expressions (physical coords)."""
        u1 = 0.1*(ufl.sin(ufl.pi*x[0])**2)*(ufl.sin(ufl.pi*x[1])**2)*(-1 + 2*x[1])
        u2 = 0.1*(ufl.sin(ufl.pi*x[0])**2)*(ufl.sin(ufl.pi*x[1])**2)*(1 - 2*x[0])
        return ufl.as_vector([u1, u2])

    def u_exact_numpy(x):
        """Exact displacement as numpy; x has shape (3, N) (dolfinx convention)."""
        u1 = 0.1*(np.sin(np.pi*x[0])**2)*(np.sin(np.pi*x[1])**2)*(-1 + 2*x[1])
        u2 = 0.1*(np.sin(np.pi*x[0])**2)*(np.sin(np.pi*x[1])**2)*(1 - 2*x[0])
        return np.array([u1, u2], dtype=dtype)

    def locate_boundary_dofs(unf_mesh, V):
        dim = unf_mesh.topology.dim
        left   = dolfinx.mesh.locate_entities_boundary(unf_mesh, dim-1, lambda x: np.isclose(x[0], xmin[0]))
        bottom = dolfinx.mesh.locate_entities_boundary(unf_mesh, dim-1, lambda x: np.isclose(x[1], xmin[1]))
        right  = dolfinx.mesh.locate_entities_boundary(unf_mesh, dim-1, lambda x: np.isclose(x[0], xmax[0]))
        top    = dolfinx.mesh.locate_entities_boundary(unf_mesh, dim-1, lambda x: np.isclose(x[1], xmax[1]))
        return (
            dolfinx.fem.locate_dofs_topological(V=V, entity_dim=1, entities=left),
            dolfinx.fem.locate_dofs_topological(V=V, entity_dim=1, entities=right),
            dolfinx.fem.locate_dofs_topological(V=V, entity_dim=1, entities=bottom),
            dolfinx.fem.locate_dofs_topological(V=V, entity_dim=1, entities=top),
        )

    def _build_map_quantities(unf_mesh, a_param):
        """Return UFL quantities for the mapping F(xi) -> (x,y)."""
        xi = ufl.SpatialCoordinate(unf_mesh)
        xi1, xi2 = xi[0], xi[1]
        a = dolfinx.fem.Constant(unf_mesh, PETSc.ScalarType(a_param))
        s    = 1 + (a - 1) * (2*xi2 - xi2**2)          # det J = s  (>0 for a>0)
        q    = (xi1 - 0.5) * (a - 1) * (2 - 2*xi2)     # off-diagonal Jacobian entry
        detJ = s
        J_inv = ufl.as_matrix([[1/s, -q/s],
                               [0.0,   1.0]])
        x_phys = ufl.as_vector([0.5 + (xi1 - 0.5)*s, xi2])
        return detJ, J_inv, x_phys

    def _epsilon_phys(w, J_inv):
        return ufl.sym(ufl.dot(ufl.grad(w), J_inv))

    def _sigma_phys(w, J_inv):
        eps = _epsilon_phys(w, J_inv)
        return 2*mu*eps + lmbda*ufl.tr(eps)*ufl.Identity(2)

    def solve_elasticity(impl_func, n_cells, degree, stabilization, a_param):
        cache_dir = None
        try:
            cache_dir = os.path.join("/tmp/fenicsx_cache", uuid.uuid4().hex)
            os.makedirs(cache_dir, exist_ok=True)
            jit_options = {"cache_dir": cache_dir}

            unf_mesh = create_unfitted_impl_Cartesian_mesh(
                MPI.COMM_WORLD, impl_func, n_cells, xmin, xmax,
                exclude_empty_cells=True, dtype=dtype)

            dim = unf_mesh.topology.dim
            V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree, (dim,)))
            u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

            detJ, J_inv, x_phys = _build_map_quantities(unf_mesh, a_param)
            u_e = u_exact_expr(x_phys)

            # Dirichlet BC: exact solution pulled back to reference coords
            def u_exact_at_F(xi_pts):
                xi2_np = xi_pts[1]
                s_np = 1.0 + (a_param - 1.0)*(2.0*xi2_np - xi2_np**2)
                x_np = 0.5 + (xi_pts[0] - 0.5)*s_np
                return u_exact_numpy(np.array([x_np, xi2_np, np.zeros_like(xi2_np)], dtype=dtype))

            u_D = dolfinx.fem.Function(V)
            u_D.interpolate(u_exact_at_F)
            bcs = [dolfinx.fem.dirichletbc(u_D, dofs) for dofs in locate_boundary_dofs(unf_mesh, V)]

            dx = ufl.dx(domain=unf_mesh)

            # Physical-space bilinear form pulled back to reference
            a_expr = ufl.inner(_sigma_phys(u, J_inv), _epsilon_phys(v, J_inv)) * detJ * dx

            # RHS: using integration-by-parts identity
            # ∫ f·v detJ dxi + ∫_Γ σ(u_e)·n_x·v dΓ = ∫ σ(u_e):ε(v) detJ dxi
            L_expr = ufl.inner(_sigma_phys(u_e, J_inv), _epsilon_phys(v, J_inv)) * detJ * dx

            petsc_options = {
                "ksp_type": "preonly",
                "pc_type": "cholesky",
                "ksp_diagonal_scale": True,
            }

            if stabilization == 0.0:
                problem = qugar.dolfinx.LinearProblem(
                    a_expr, L_expr, bcs=bcs, petsc_options=petsc_options,
                    jit_options=jit_options)
                uh = problem.solve()
                return unf_mesh, uh

            a_active_form = qugar.dolfinx.form_custom(
                (1.0 - stabilization) * a_expr, dtype=PETSc.ScalarType, jit_options=jit_options)
            a_full_form = dolfinx.fem.form(
                stabilization * a_expr, dtype=PETSc.ScalarType, jit_options=jit_options)
            L_form = qugar.dolfinx.form_custom(
                L_expr, dtype=PETSc.ScalarType, jit_options=jit_options)

            A = dolfinx.fem.petsc.create_matrix(a_active_form)
            b = dolfinx.fem.petsc.create_vector(L_form)

            A.zeroEntries()
            a_active_coeffs = a_active_form.pack_coefficients()
            dolfinx.fem.petsc.assemble_matrix(A, a_active_form, coeffs=a_active_coeffs)
            dolfinx.fem.petsc.assemble_matrix(A, a_full_form)
            A.assemble()
            bc_dofs = np.sort(np.unique(np.concatenate([bc.dof_indices()[0] for bc in bcs])))
            A.zeroRowsColumns(bc_dofs.tolist(), 1.0)

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
            for k, v_opt in petsc_options.items():
                opts[k] = v_opt
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
            if cache_dir and os.path.exists(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)
            gc.collect()

    def compute_normalized_l2_error(uh, unf_mesh, degree, a_param):
        cache_dir = None
        try:
            cache_dir = os.path.join("/tmp/fenicsx_cache", uuid.uuid4().hex)
            os.makedirs(cache_dir, exist_ok=True)
            jit_options = {"cache_dir": cache_dir}

            detJ, _, x_phys = _build_map_quantities(unf_mesh, a_param)
            u_e = u_exact_expr(x_phys)

            dx = ufl.dx(domain=unf_mesh)
            quad_opts = {"quadrature_degree": degree * QUAD_FACTOR}

            # L2 error in physical space: ||u_h - u_e||^2 = ∫|u_h - u_e|^2 detJ dxi
            err_form = qugar.dolfinx.form_custom(
                ufl.dot(uh - u_e, uh - u_e) * detJ * dx,
                form_compiler_options=quad_opts, jit_options=jit_options)
            ref_form = qugar.dolfinx.form_custom(
                ufl.dot(u_e, u_e) * detJ * dx,
                form_compiler_options=quad_opts, jit_options=jit_options)

            l2_err = np.sqrt(dolfinx.fem.assemble_scalar(err_form, coeffs=err_form.pack_coefficients()))
            l2_ref = np.sqrt(dolfinx.fem.assemble_scalar(ref_form, coeffs=ref_form.pack_coefficients()))
            return l2_err / l2_ref
        finally:
            if cache_dir and os.path.exists(cache_dir):
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
    print(f"STABILIZATION STUDY: normalized L2 error vs mapping parameter a (degree p={DEGREE})")
    print(f"a values: {A_VALUES}")
    print("=" * 90)

    n_g = len(GEOMETRIES)
    n_s = len(STABILIZATION_VALUES)
    n_a = len(A_VALUES)
    errors = np.full((n_g, n_s, n_a), np.nan, dtype=np.float64)

    for g_idx, (geom_key, geom_label, impl_func) in enumerate(GEOMETRIES):
        print(f"\nGeometry: {geom_label}")

        for a_idx, a_val in enumerate(A_VALUES):
            print(f"  a = {a_val:.3f}")

            for s_idx, stabilization in enumerate(STABILIZATION_VALUES):
                try:
                    mesh, uh = solve_elasticity(
                        impl_func=impl_func,
                        n_cells=N_CELLS,
                        degree=DEGREE,
                        stabilization=stabilization,
                        a_param=a_val,
                    )
                    rel_err = compute_normalized_l2_error(uh, mesh, DEGREE, a_val)
                    errors[g_idx, s_idx, a_idx] = rel_err
                    print(f"    stab = {stabilization:.0e}  |  rel L2 error = {rel_err:.3e}")
                except Exception as exc:
                    print(f"    [WARN] Failed for stab={stabilization:.0e}: {exc}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        RESULTS_FILE,
        geometry_keys=np.array([g[0] for g in GEOMETRIES]),
        geometry_labels=np.array([g[1] for g in GEOMETRIES]),
        stabilizations=np.array(STABILIZATION_VALUES, dtype=np.float64),
        a_values=A_VALUES,
        degree=np.int32(DEGREE),
        rel_l2_errors=errors,
    )
    print(f"\n[INFO] Results saved to {RESULTS_FILE}")


if RUN_PLOTS:
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Results file not found: {RESULTS_FILE}. Run with RUN_ANALYSIS=True first.")

    data = np.load(RESULTS_FILE)
    geometry_keys  = data["geometry_keys"]
    geometry_labels = data["geometry_labels"]
    stabilizations = data["stabilizations"]
    a_values       = data["a_values"]
    degree         = int(data["degree"])
    rel_l2_errors  = data["rel_l2_errors"]
    print(f"[INFO] Results loaded from {RESULTS_FILE}")

    if not has_matplotlib:
        raise ImportError("matplotlib is required for plotting but is not installed.")

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 22,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })

    colors  = ["blue", "red", "orange", "green", "brown", "purple"]
    markers = ["o", "s", "^", "d", "v", "p"]

    HELPING_DIR = FIGS_DIR / "helping"
    GEOM_IMAGES = {
        "schwarz_diamond":   HELPING_DIR / "schwarz.png",
        "schwarz_primitive": HELPING_DIR / "schwarz_primitive.png",
        "schoen_frd":        HELPING_DIR / "schoen_frd.png",
        "schoen_iwp":        HELPING_DIR / "schoen_iwp.png",
    }

    def _stab_label(s):
        if s == 0.0:
            return r"$\rho=0$"
        exp = int(round(np.log10(s)))
        return rf"$\rho=10^{{{exp}}}$"

    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    for g_idx, geom_key in enumerate(geometry_keys):
        fig, ax = plt.subplots(figsize=(8.5, 6.5))

        for s_idx, stabilization in enumerate(stabilizations):
            curve = rel_l2_errors[g_idx, s_idx, :]
            mask  = np.isfinite(curve)
            if not np.any(mask):
                continue
            ax.semilogy(
                a_values[mask],
                curve[mask],
                color=colors[s_idx % len(colors)],
                marker=markers[s_idx % len(markers)],
                linewidth=2.0,
                markersize=6.5,
                markerfacecolor="white",
                markeredgewidth=1.3,
            )

        ax.set_xlabel(r"Mapping parameter $a$")
        ax.set_ylabel(r"$L^2$ error")
        ax.set_ylim(1e-9, 10**1.5)
        ax.grid(True, which="major", alpha=0.25)

        # Top-left: a=0.1 deformed geometry
        map_path_01 = FIGS_DIR / f"stabilization_mapeffect_{geom_key}.png"
        if map_path_01.exists():
            axins_tl = ax.inset_axes([0.01, 0.73, 0.25, 0.25])
            axins_tl.imshow(plt.imread(str(map_path_01)))
            axins_tl.axis("off")

        # Top-middle: a=1 (identity mapping)
        map_path_a1 = FIGS_DIR / f"stabilization_mapeffect_a1_{geom_key}.png"
        if map_path_a1.exists():
            axins_tm = ax.inset_axes([0.375, 0.73, 0.25, 0.25])
            axins_tm.imshow(plt.imread(str(map_path_a1)))
            axins_tm.axis("off")

        # Top-right: a=2.0 deformed geometry
        map_path_a2 = FIGS_DIR / f"stabilization_mapeffect_a2_{geom_key}.png"
        if map_path_a2.exists():
            axins_tr = ax.inset_axes([0.74, 0.73, 0.25, 0.25])
            axins_tr.imshow(plt.imread(str(map_path_a2)))
            axins_tr.axis("off")

        plt.tight_layout()
        out_path = FIGS_DIR / f"stabilization_mapstudy_{geom_key}.pdf"
        plt.savefig(str(out_path), bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"[INFO] Figure saved to {out_path}")

    # Standalone horizontal legend
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0],
               color=colors[s_idx % len(colors)],
               marker=markers[s_idx % len(markers)],
               linewidth=2.0, markersize=6.5,
               markerfacecolor="white", markeredgewidth=1.3,
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
    plt.savefig(str(FIGS_DIR / "stabilization_mapstudy_legend.pdf"), bbox_inches=bbox, pad_inches=0.08)
    plt.close(fig_leg)
    print("[INFO] Legend figure saved to stabilization_mapstudy_legend.pdf")


if RUN_IMAGES:
    import qugar.impl as _impl
    import qugar.cpp as _cpp
    from matplotlib.colors import ListedColormap

    if not has_matplotlib:
        raise ImportError("matplotlib is required for RUN_IMAGES.")

    N_IMG = 500

    _grey  = [200/255, 197/255, 189/255, 1.0]
    _white = [1.0,     1.0,     1.0,     1.0]
    _cmap  = ListedColormap([_white, _grey])

    _GEOM_IMG = [
        ("schwarz_diamond", "Schwarz Diamond",
         _impl.ImplicitFunc(_cpp.create_functions_addition(
             _impl.create_Schwarz_Diamond(periods=[1, 1], z=0.0).cpp_object,
             _impl.create_constant(-0.1, dim=2).cpp_object,
         ))),
        ("schwarz_primitive", "Schwarz Primitive",
         _impl.create_Schwarz_Primitive(periods=[1, 1], z=0.5)),
        ("schoen_frd", "Schoen FRD",
         _impl.create_negative(_impl.create_Schoen_FRD(periods=[1, 1], z=0))),
        ("schoen_iwp", "Schoen IWP",
         _impl.create_Schoen_IWP(periods=[1, 1], z=0)),
    ]

    xi1_v = np.linspace(0.0, 1.0, N_IMG)
    xi2_v = np.linspace(0.0, 1.0, N_IMG)
    XI1, XI2 = np.meshgrid(xi1_v, xi2_v)
    pts_img  = np.column_stack([XI1.ravel(), XI2.ravel()])
    xi2_bnd  = np.linspace(0.0, 1.0, N_IMG)

    def _render_mapeffect(a_val, suffix):
        S_img   = 1.0 + (a_val - 1.0) * (2.0*XI2 - XI2**2)
        X_phys  = 0.5 + (XI1 - 0.5) * S_img
        Y_phys  = XI2.copy()
        s_bnd   = 1.0 + (a_val - 1.0) * (2.0*xi2_bnd - xi2_bnd**2)
        x_left  = 0.5 - 0.5 * s_bnd
        x_right = 0.5 + 0.5 * s_bnd

        for geom_key, _, impl_func in _GEOM_IMG:
            PHI    = impl_func.eval(pts_img).reshape(N_IMG, N_IMG)
            ACTIVE = (PHI < 0.0).astype(float)

            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.pcolormesh(X_phys, Y_phys, ACTIVE,
                          cmap=_cmap, vmin=0.0, vmax=1.0, shading="auto", rasterized=True)
            ax.contour(X_phys, Y_phys, PHI, levels=[0.0],
                       colors="black", linewidths=1.2)
            ax.plot([0.0, 1.0],             [0.0, 0.0], "k-", linewidth=2.0)  # bottom
            ax.plot([x_left[-1], x_right[-1]], [1.0, 1.0], "k-", linewidth=2.0)  # top
            ax.plot(x_left,  xi2_bnd, "k-", linewidth=1.2)                   # left
            ax.plot(x_right, xi2_bnd, "k-", linewidth=1.2)                   # right
            ax.set_aspect("equal")
            ax.axis("off")
            plt.tight_layout(pad=0.1)
            out_path = FIGS_DIR / f"stabilization_mapeffect{suffix}_{geom_key}.png"
            plt.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)
            print(f"[INFO] Image saved to {out_path}")

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    _render_mapeffect(0.1, "")       # a=0.1  →  stabilization_mapeffect_{geom_key}.png
    _render_mapeffect(1.0, "_a1")    # a=1.0  →  stabilization_mapeffect_a1_{geom_key}.png
    _render_mapeffect(2.0, "_a2")    # a=2.0  →  stabilization_mapeffect_a2_{geom_key}.png
