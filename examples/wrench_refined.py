"""h-refined wrench runs: k x k children per lattice cell.

Mode "window" keeps the physical lattice identical for every k (h-refinement
reference of the validation study); mode "lattice" makes each child a full
lattice cell (the paper's dense wrench is k = 9). One process invocation per
configuration.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import qugar.cpp
from mpi4py import MPI

from FLASh.utils import Communicators
from FLASh.pde import BDDC, Cholesky
from FLASh.rom import MDEIM

from _paths import FIGS_DIR, ROM_DATA_DIR
from wrench_validation import PAPER_DIR, PAPER_RANGE, PAPER_SEED
from wrench_refined_utils import (
    RefinedWrenchGeometry,
    RefinedWrenchMesh,
    create_refined_gdm,
    setup_solver_with_gdm,
)
from wrench_validation_utils import (
    ROM_FAMILY,
    ROM_MDEIM_N,
    ROM_MDEIM_P,
    ROM_BOX,
    _load_parameter_field,
    _points_in_array,
    _traction,
    _verify_neumann_marker,
    compute_applied_load,
    compute_boundary_mean_displacement,
    compute_compliance,
    compute_consistent_reactions,
    compute_external_resultant,
    compute_point_displacement,
    compute_stress_reactions,
    load_wrench_geometry,
)
from FLASh.pde import Elasticity

dtype = np.float64

FINE_SEED = 20260214

SOLVERS = {"bddc": BDDC, "cholesky": Cholesky}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--mode", choices=("window", "lattice"), default="window")
    parser.add_argument("--solver", choices=SOLVERS, default="cholesky")
    parser.add_argument("--rom", action="store_true")
    parser.add_argument("--stabilization", type=float, default=0.0)
    parser.add_argument("--n-quad", type=int, default=None)
    parser.add_argument("--n-quad-ref", type=int, default=20)
    parser.add_argument("--write-vtu", action="store_true")
    parser.add_argument("--write-png", action="store_true")
    return parser.parse_args()


def run_tag(args: argparse.Namespace) -> str:
    tag = f"{args.mode}_k{args.k}_p{args.degree}_{args.solver}"
    if args.rom:
        tag += "_rom"
    if args.stabilization > 0:
        tag += f"_stab{args.stabilization:g}"
    return tag


def main() -> None:
    args = parse_args()

    assert not (args.rom and args.mode == "window"), "ROM cells must be full lattice periods"

    n_quad = args.n_quad if args.n_quad is not None else max(8, args.degree + 2)

    parent_geometry, edges_dir, edges_neu = load_wrench_geometry(args.degree)
    parent_mesh = parent_geometry.coarse_mesh

    parent_field = _load_parameter_field(
        parent_mesh._n, PAPER_DIR / "threshold_field_coarse.npy", PAPER_RANGE, PAPER_SEED
    )
    parent_mesh.set_parameter_field(parent_field)

    mesh = RefinedWrenchMesh(parent_mesh, parent_geometry, args.k)
    geometry = RefinedWrenchGeometry(parent_geometry, mesh, args.mode)

    if args.mode == "lattice":
        field = _load_parameter_field(
            mesh._n,
            PAPER_DIR / f"threshold_field_fine_k{args.k}.npy",
            PAPER_RANGE,
            FINE_SEED,
        )
        mesh.set_parameter_field(field)

    edges_dir_c = mesh.child_edges_of(edges_dir)
    edges_neu_c = mesh.child_edges_of(edges_neu)

    nodes_dir_c = np.unique(np.asarray(mesh.edge_vertex_conn)[edges_dir_c].flatten())
    points_dir = np.vstack([
        mesh.vertex_coordinates[nodes_dir_c][:, :2],
        mesh.edge_coordinates[edges_dir_c][:, :2],
    ]).T
    points_neu = mesh.edge_coordinates[edges_neu_c][:, :2].T

    def h_bc(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (0 + 0 * X[0], 0 + 0 * X[0])

    def source(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (0.0 + 0.0 * X[0], 0.0 + 0.0 * X[0])

    def neu_marker(x: np.ndarray) -> np.ndarray:
        return _points_in_array(x[:2], points_neu[:2])

    _verify_neumann_marker(mesh, edges_neu_c, neu_marker)

    exterior_bc = [
        (0, h_bc, lambda x: _points_in_array(x[:2], points_dir[:2]), 0),
        (1, _traction, neu_marker, 1),
    ]

    pde_kwargs: dict = {}
    if args.rom:
        assert args.degree == 8, "ROM data is trained at basis_degree = 8"
        p0 = np.array([ROM_BOX[0]] * 4)
        p1 = np.array([ROM_BOX[1]] * 4)
        models = {}
        for name in ("K_core", "M_core", "bM_core"):
            model = MDEIM(ROM_MDEIM_N, ROM_MDEIM_P, p0, p1)
            model.set_up_from_files(str(ROM_DATA_DIR / ROM_FAMILY / name))
            models[name] = model
        pde_kwargs = {
            "K_model": models["K_core"],
            "M_model": models["M_core"],
            "bM_model": models["bM_core"],
            "K_full_core": np.load(str(ROM_DATA_DIR / ROM_FAMILY / "K_core" / "full_array.npy")),
        }

    pde = Elasticity(exterior_bc=exterior_bc, source=source, E=5, nu=0.25, **pde_kwargs)

    # hexagon centroid of the PARENT mesh: identical across k and Phase-B runs
    nodes_dir_p = np.unique(np.asarray(parent_mesh.edge_vertex_conn)[np.atleast_1d(edges_dir)].flatten())
    center = parent_mesh.vertex_coordinates[nodes_dir_p].mean(axis=0)[:2]

    communicators = Communicators()

    sbdmn_opts: dict = {"assemble": True, "n_quad_pts": n_quad}
    if args.stabilization > 0:
        sbdmn_opts["stabilize"] = True
        sbdmn_opts["stabilization"] = args.stabilization

    solver = SOLVERS[args.solver](
        geometry, pde, communicators,
        opts={"global_dofs_manager_opts": {"subdomain_opts": sbdmn_opts}},
    )
    gdm = create_refined_gdm(geometry, pde, communicators, opts={"subdomain_opts": sbdmn_opts})
    setup_solver_with_gdm(solver, gdm)
    solver.solve()

    F_ext, Mz_ext = compute_external_resultant(
        solver, edges_neu_c, _traction, center, n_quad=args.n_quad_ref
    )
    F_appl, Mz_appl = compute_applied_load(solver, center)
    F_cons, Mz_cons = compute_consistent_reactions(solver, center)
    F_stress, Mz_stress = compute_stress_reactions(
        solver, edges_dir_c, center, n_quad=args.n_quad_ref
    )
    compliance = compute_compliance(solver)

    coords = mesh.vertex_coordinates[:, :2]
    left = np.isclose(coords[:, 0], coords[:, 0].min())
    corner_point = coords[left][np.argmin(coords[left, 1])]
    u_corner, corner_info = compute_point_displacement(solver, corner_point)
    u_neu_mean = compute_boundary_mean_displacement(solver, edges_neu_c, n_quad=args.n_quad_ref)

    area_local = 0.0
    for sub in gdm.subdomains:
        quad = qugar.cpp.create_quadrature(sub.create_qugar_mesh(), np.array([0]), 8)
        det = np.abs(sub._map.evaluate_jacobian_determinant(quad.points))
        area_local += float(quad.weights @ det)
    area = communicators.global_comm.allreduce(area_local, op=MPI.SUM)

    n_interior = sum(sub.interior_dofs.size for sub in gdm.subdomains)
    n_interior = communicators.global_comm.allreduce(n_interior)
    n_dofs = int(n_interior + gdm.get_num_boundary_dofs())

    if communicators.global_comm.Get_rank() == 0:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()

        results = {
            "config": {
                "k": args.k, "mode": args.mode, "degree": args.degree,
                "solver": args.solver, "rom": args.rom,
                "stabilization": args.stabilization, "n_quad": n_quad,
                "n_quad_ref": args.n_quad_ref, "commit": commit,
            },
            "center": center.tolist(),
            "n_subdomains": int(mesh._N),
            "n_dofs": n_dofs,
            "material_area": area,
            "F_ext": F_ext.tolist(), "Mz_ext": Mz_ext,
            "F_applied": F_appl.tolist(), "Mz_applied": Mz_appl,
            "F_consistent": F_cons.tolist(), "Mz_consistent": Mz_cons,
            "F_stress": F_stress.tolist(), "Mz_stress": Mz_stress,
            "compliance": compliance,
            "corner_point": corner_point.tolist(),
            "u_corner": u_corner.tolist(),
            "corner_info": corner_info,
            "u_neu_mean": u_neu_mean.tolist(),
            "stats": {k: v for k, v in solver.get_stats().items()},
        }

        out = PAPER_DIR / f"run_refined_{run_tag(args)}.json"
        out.parent.mkdir(exist_ok=True, parents=True)
        out.write_text(json.dumps(results, indent=2, default=float))

        eq_F = np.linalg.norm(F_cons + F_appl) / np.linalg.norm(F_appl)
        print(f"[{run_tag(args)}] n_sub = {mesh._N}, n_dofs = {n_dofs}, area = {area:.10e}")
        print(f"  F_applied   = {F_appl}, Mz_appl = {Mz_appl:.6e}")
        print(f"  equilibrium = {eq_F:.3e}")
        print(f"  compliance  = {compliance:.12e}")
        print(f"  u_corner    = {u_corner} at {corner_point}")
        print(f"  written to {out}")

    if args.write_vtu or args.write_png:
        assert communicators.global_comm.Get_size() == 1, "field output is serial only"
        from wrench_validation_plot_field import build_field_meshes, render_field, save_field_vtu

        meshes = build_field_meshes(solver)
        if args.write_vtu:
            save_field_vtu(meshes, PAPER_DIR / f"field_refined_{run_tag(args)}.vtu")
        if args.write_png:
            FIGS_DIR.mkdir(exist_ok=True, parents=True)
            render_field(meshes, FIGS_DIR / f"wrench_refined_{run_tag(args)}.png")


if __name__ == "__main__":
    main()
