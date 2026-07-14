"""
Reproduces Figure 11(a) (Section 5.1.1, "Accuracy of the stabilization").
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

DEGREE = 14
STABILIZATION_VALUES = [1e-2, 1e-3, 1e-4, 1e-5, 0.0]
N_CELLS = 1
QUAD_FACTOR = 4
N_OFFSET_POINTS = 9
RESULTS_FILE = RESULTS_DIR / "stabilization_geometry_levelset_sweep_results.npz"

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

    def compute_active_area(unf_mesh):
        cache_dir = None
        try:
            cache_dir = os.path.join("/tmp/fenicsx_cache", uuid.uuid4().hex)
            os.makedirs(cache_dir, exist_ok=True)
            jit_options = {"cache_dir": cache_dir}

            dx = ufl.dx(domain=unf_mesh)
            area_form = qugar.dolfinx.form_custom(
                dolfinx.fem.Constant(unf_mesh, PETSc.ScalarType(1.0)) * dx,
                jit_options=jit_options,
            )
            return float(dolfinx.fem.assemble_scalar(area_form, coeffs=area_form.pack_coefficients()))
        finally:
            if cache_dir is not None and os.path.exists(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)
            gc.collect()

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
                problem = qugar.dolfinx.LinearProblem(
                    a_expr, L_expr, bcs=bcs, petsc_options=petsc_options,
                    jit_options=jit_options,
                )
                uh = problem.solve()
                return unf_mesh, uh

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
    # Each geometry entry: (key, label, impl_func, offset_min, offset_max)
    GEOMETRY_DEFS = [
        ("schwarz_diamond", "Schwarz Diamond",
        qugar.impl.ImplicitFunc(qugar.cpp.create_functions_addition(
            qugar.impl.create_Schwarz_Diamond(periods=[1, 1], z=0.0).cpp_object,
            qugar.impl.create_constant(-0.1, dim=2).cpp_object,
        )),
        -1, -0.1),
        ("schwarz_primitive", "Schwarz Primitive",
        qugar.impl.create_Schwarz_Primitive(periods=[1, 1], z=0.5),
        -1, 0.5),
        ("schoen_frd", "Schoen FRD",
        qugar.impl.create_negative(qugar.impl.create_Schoen_FRD(periods=[1, 1], z=0)),
        -7, 0.8),
        ("schoen_iwp", "Schoen IWP",
        qugar.impl.create_Schoen_IWP(periods=[1, 1], z=0),
        -3, 1.5),
    ]

    # Per-geometry offset arrays, shape (n_geoms, N_OFFSET_POINTS)
    all_offsets = np.array([
        np.linspace(g[3], g[4], N_OFFSET_POINTS) for g in GEOMETRY_DEFS
    ], dtype=np.float64)

    print("\n" + "=" * 90)
    print(f"STABILIZATION STUDY: normalized L2 error vs surface area (degree p={DEGREE})")
    print("=" * 90)

    # shapes: (n_geoms, n_stabs, n_offsets)
    n_g = len(GEOMETRY_DEFS)
    n_s = len(STABILIZATION_VALUES)
    n_o = N_OFFSET_POINTS
    errors = np.full((n_g, n_s, n_o), np.nan, dtype=np.float64)
    surface_areas = np.full((n_g, n_o), np.nan, dtype=np.float64)

    for g_idx, (geom_key, geom_label, base_func, _, _) in enumerate(GEOMETRY_DEFS):
        print(f"\nGeometry: {geom_label}  (offsets: {all_offsets[g_idx, 0]:.2f} .. {all_offsets[g_idx, -1]:.2f})")

        for o_idx, offset in enumerate(all_offsets[g_idx]):
            shifted_func = qugar.impl.ImplicitFunc(
                qugar.cpp.create_functions_addition(
                    base_func.cpp_object,
                    qugar.impl.create_constant(offset, dim=2).cpp_object,
                )
            )

            # surface area only depends on geometry, not stabilization
            try:
                tmp_mesh = create_unfitted_impl_Cartesian_mesh(
                    MPI.COMM_WORLD,
                    shifted_func,
                    N_CELLS,
                    xmin,
                    xmax,
                    exclude_empty_cells=True,
                    dtype=dtype,
                )
                area = compute_active_area(tmp_mesh)
                surface_areas[g_idx, o_idx] = area
                del tmp_mesh
                gc.collect()
            except Exception as exc:
                print(f"  [WARN] Surface area failed for offset={offset:.2f}: {exc}")

            print(f"  Offset = {offset:.2f}  |  surface area = {surface_areas[g_idx, o_idx]:.4f}")

            for s_idx, stabilization in enumerate(STABILIZATION_VALUES):
                try:
                    mesh, uh = solve_elasticity(
                        impl_func=shifted_func,
                        n_cells=N_CELLS,
                        degree=DEGREE,
                        stabilization=stabilization,
                    )
                    rel_l2_error = compute_normalized_l2_error(uh, mesh, DEGREE)
                    errors[g_idx, s_idx, o_idx] = rel_l2_error
                    print(f"    stab = {stabilization:.0e}  |  rel L2 error = {rel_l2_error:.3e}")
                except Exception as exc:
                    print(f"    [WARN] Solve failed for stab={stabilization:.0e}: {exc}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        RESULTS_FILE,
        geometry_keys=np.array([g[0] for g in GEOMETRY_DEFS]),
        geometry_labels=np.array([g[1] for g in GEOMETRY_DEFS]),
        stabilizations=np.array(STABILIZATION_VALUES, dtype=np.float64),
        levelset_offsets=all_offsets,
        degree=np.int32(DEGREE),
        rel_l2_errors=errors,
        surface_areas=surface_areas,
    )
    print(f"\n[INFO] Results saved to {RESULTS_FILE}")


if RUN_PLOTS:
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Results file not found: {RESULTS_FILE}. Run with RUN_ANALYSIS=True first.")

    data = np.load(RESULTS_FILE)
    geometry_keys = data["geometry_keys"]
    geometry_labels = data["geometry_labels"]
    stabilizations = data["stabilizations"]
    levelset_offsets = data["levelset_offsets"]   # shape (n_geoms, N_OFFSET_POINTS)
    degree = int(data["degree"])
    rel_l2_errors = data["rel_l2_errors"]
    surface_areas = data["surface_areas"]
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

        areas = surface_areas[g_idx, :]
        valid = np.isfinite(areas)

        for s_idx, stabilization in enumerate(stabilizations):
            curve = rel_l2_errors[g_idx, s_idx, :]
            mask = valid & np.isfinite(curve)
            if not np.any(mask):
                continue
            ax.semilogy(
                1.0 - areas[mask],
                curve[mask],
                color=colors[s_idx % len(colors)],
                marker=markers[s_idx % len(markers)],
                linewidth=2.0,
                markersize=6.5,
                markerfacecolor="white",
                markeredgewidth=1.3,
            )

        ax.set_xlabel("Area ratio")
        ax.set_ylabel(r"$L^2$ error")
        ax.set_ylim(1e-9, 1e2)
        ax.grid(True, which="major", alpha=0.25)

        # Top-left: geometry at minimum offset
        min_path = FIGS_DIR / f"stabilization_surfstudy_a1_{geom_key}.png"
        if min_path.exists():
            axins_tl = ax.inset_axes([0.01, 0.73, 0.23, 0.23])
            axins_tl.imshow(plt.imread(str(min_path)))
            axins_tl.axis("off")

        # Top-right: geometry at maximum offset
        max_path = FIGS_DIR / f"stabilization_surfstudy_a2_{geom_key}.png"
        if max_path.exists():
            axins_tr = ax.inset_axes([0.76, 0.73, 0.23, 0.23])
            axins_tr.imshow(plt.imread(str(max_path)))
            axins_tr.axis("off")

        plt.tight_layout()
        out_path = FIGS_DIR / f"stabilization_surfstudy_{geom_key}.pdf"
        plt.savefig(str(out_path), bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"[INFO] Figure saved to {out_path}")

    # Standalone horizontal legend figure
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0],
               color=colors[s_idx % len(colors)],
               marker=markers[s_idx % len(markers)],
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
    plt.savefig(str(FIGS_DIR / "stabilization_surfstudy_legend.pdf"), bbox_inches=bbox, pad_inches=0.08)
    plt.close(fig_leg)
    print("[INFO] Legend figure saved to stabilization_surfstudy_legend.pdf")


if RUN_IMAGES:
    import qugar.impl as _impl
    import qugar.cpp as _cpp
    from matplotlib.colors import ListedColormap

    if not has_matplotlib:
        raise ImportError("matplotlib is required for RUN_IMAGES.")

    N_IMG = 500

    _grey  = [200/255, 197/255, 189/255, 1.0]
    _white = [1.0,     1.0,     1.0,     1.0]
    _cmap  = ListedColormap([_white, _grey])   # 0 → white (trimmed), 1 → grey (active)

    # (key, base_func, offset_min, offset_max)
    _GEOM_IMG = [
        ("schwarz_diamond",
         _impl.ImplicitFunc(_cpp.create_functions_addition(
             _impl.create_Schwarz_Diamond(periods=[1, 1], z=0.0).cpp_object,
             _impl.create_constant(-0.1, dim=2).cpp_object,
         )),
         -1.0, -0.1),
        ("schwarz_primitive",
         _impl.create_Schwarz_Primitive(periods=[1, 1], z=0.5),
         -1.0, 0.5),
        ("schoen_frd",
         _impl.create_negative(_impl.create_Schoen_FRD(periods=[1, 1], z=0)),
         -7.0, 0.8),
        ("schoen_iwp",
         _impl.create_Schoen_IWP(periods=[1, 1], z=0),
         -3.0, 1.5),
    ]

    xi1_v = np.linspace(0.0, 1.0, N_IMG)
    xi2_v = np.linspace(0.0, 1.0, N_IMG)
    XI1, XI2 = np.meshgrid(xi1_v, xi2_v)
    pts_img = np.column_stack([XI1.ravel(), XI2.ravel()])

    def _render_threshold(base_func, offset, out_path):
        shifted = _impl.ImplicitFunc(
            _cpp.create_functions_addition(
                base_func.cpp_object,
                _impl.create_constant(float(offset), dim=2).cpp_object,
            )
        )
        PHI    = shifted.eval(pts_img).reshape(N_IMG, N_IMG)
        ACTIVE = (PHI < 0.0).astype(float)

        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.pcolormesh(XI1, XI2, ACTIVE,
                      cmap=_cmap, vmin=0.0, vmax=1.0, shading="auto", rasterized=True)
        ax.contour(XI1, XI2, PHI, levels=[0.0], colors="black", linewidths=1.2)
        ax.plot([0.0, 1.0], [0.0, 0.0], "k-", linewidth=2.0)   # bottom
        ax.plot([0.0, 1.0], [1.0, 1.0], "k-", linewidth=2.0)   # top
        ax.plot([0.0, 0.0], [0.0, 1.0], "k-", linewidth=1.2)   # left
        ax.plot([1.0, 1.0], [0.0, 1.0], "k-", linewidth=1.2)   # right
        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout(pad=0.1)
        plt.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        print(f"[INFO] Image saved to {out_path}")

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    for geom_key, base_func, offset_min, offset_max in _GEOM_IMG:
        _render_threshold(base_func, offset_min,
                          FIGS_DIR / f"stabilization_surfstudy_a1_{geom_key}.png")
        _render_threshold(base_func, offset_max,
                          FIGS_DIR / f"stabilization_surfstudy_a2_{geom_key}.png")
