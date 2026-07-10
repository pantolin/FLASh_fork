"""Single-configuration run of the wrench validation study.

Solves the coarse wrench (90 subdomains) at a given basis degree, with or
without the MDEIM ROM, and writes all QoIs to a JSON file. Run one process
invocation per configuration (a class-level cache in `Subdomain` makes mixing
degrees within one process unsafe).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from FLASh.utils import Communicators
from FLASh.pde import BDDC, Cholesky

from _paths import RESULTS_DIR
from wrench_validation_utils import (
    build_case,
    compute_applied_load,
    compute_boundary_mean_displacement,
    compute_compliance,
    compute_consistent_reactions,
    compute_external_resultant,
    compute_point_displacement,
    compute_stress_reactions,
)

VALIDATION_DIR = RESULTS_DIR / "validation_wrench"
PAPER_DIR = RESULTS_DIR / "wrench_paper"

PAPER_RANGE = (-2.5, 2.5)
PAPER_SEED = 20260213

SOLVERS = {"bddc": BDDC, "cholesky": Cholesky}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--solver", choices=SOLVERS, default="cholesky")
    parser.add_argument("--rom", action="store_true")
    parser.add_argument("--stabilization", type=float, default=0.0)
    parser.add_argument("--n-quad", type=int, default=None)
    parser.add_argument("--n-quad-ref", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--write-fields", action="store_true")
    parser.add_argument(
        "--paper-field", action="store_true",
        help="use the paper threshold field (uniform [-2.5, 2.5], its own seed) "
             "and write outputs under wrench_paper/",
    )
    return parser.parse_args()


def run_tag(args: argparse.Namespace) -> str:
    tag = f"p{args.degree}_{args.solver}"
    if args.rom:
        tag += "_rom"
    if args.stabilization > 0:
        tag += f"_stab{args.stabilization:g}"
    return tag


def main() -> None:
    args = parse_args()

    n_quad = args.n_quad if args.n_quad is not None else max(8, args.degree + 2)

    out_dir = PAPER_DIR if args.paper_field else VALIDATION_DIR

    if args.paper_field:
        case = build_case(
            degree=args.degree,
            rom=args.rom,
            params_file=PAPER_DIR / "threshold_field_coarse.npy",
            param_range=PAPER_RANGE,
            seed=PAPER_SEED,
        )
    else:
        case = build_case(
            degree=args.degree,
            rom=args.rom,
            params_file=VALIDATION_DIR / "parameters.npy",
        )

    communicators = Communicators()

    sbdmn_opts = {
        "assemble": True,
        "n_quad_pts": n_quad,
    }
    if args.stabilization > 0:
        sbdmn_opts["stabilize"] = True
        sbdmn_opts["stabilization"] = args.stabilization

    opts = {"global_dofs_manager_opts": {"subdomain_opts": sbdmn_opts}}

    solver = SOLVERS[args.solver](case.geometry, case.pde, communicators, opts=opts)
    solver.setup()
    solver.solve()

    F_ext, Mz_ext = compute_external_resultant(
        solver, case.edges_neu, case.traction, case.center, n_quad=args.n_quad_ref
    )
    F_ext_b, Mz_ext_b = compute_external_resultant(
        solver, case.edges_neu, case.traction, case.center, n_quad=args.n_quad_ref + 5
    )
    F_appl, Mz_appl = compute_applied_load(solver, case.center)

    F_cons, Mz_cons = compute_consistent_reactions(solver, case.center)
    F_stress, Mz_stress = compute_stress_reactions(
        solver, case.edges_dir, case.center, n_quad=args.n_quad_ref
    )
    compliance = compute_compliance(solver)

    coords = case.geometry.coarse_mesh.vertex_coordinates[:, :2]
    left = np.isclose(coords[:, 0], coords[:, 0].min())
    corner_point = coords[left][np.argmin(coords[left, 1])]
    u_corner, corner_info = compute_point_displacement(solver, corner_point)
    u_neu_mean = compute_boundary_mean_displacement(solver, case.edges_neu, n_quad=args.n_quad_ref)

    gdm = solver.gbl_dofs_mngr
    n_interior = sum(sub.interior_dofs.size for sub in gdm.subdomains)
    n_interior = communicators.global_comm.allreduce(n_interior)
    n_dofs = int(n_interior + gdm.get_num_boundary_dofs())

    if args.write_fields:
        solver.write_solution(str(out_dir / f"fields_{run_tag(args)}"))

    if communicators.global_comm.Get_rank() == 0:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()

        results = {
            "config": {
                "degree": args.degree,
                "solver": args.solver,
                "rom": args.rom,
                "stabilization": args.stabilization,
                "n_quad": n_quad,
                "n_quad_ref": args.n_quad_ref,
                "commit": commit,
            },
            "center": case.center.tolist(),
            "n_dofs": n_dofs,
            "F_ext": F_ext.tolist(),
            "Mz_ext": Mz_ext,
            "F_ext_check": F_ext_b.tolist(),
            "Mz_ext_check": Mz_ext_b,
            "F_applied": F_appl.tolist(),
            "Mz_applied": Mz_appl,
            "F_consistent": F_cons.tolist(),
            "Mz_consistent": Mz_cons,
            "F_stress": F_stress.tolist(),
            "Mz_stress": Mz_stress,
            "compliance": compliance,
            "corner_point": corner_point.tolist(),
            "u_corner": u_corner.tolist(),
            "corner_info": corner_info,
            "u_neu_mean": u_neu_mean.tolist(),
            "stats": {k: v for k, v in solver.get_stats().items()},
        }

        out = args.out or out_dir / f"run_{run_tag(args)}.json"
        out.parent.mkdir(exist_ok=True, parents=True)
        out.write_text(json.dumps(results, indent=2, default=float))

        eq_F = np.linalg.norm(F_cons + F_appl) / np.linalg.norm(F_appl)
        eq_M = abs(Mz_cons + Mz_appl) / abs(Mz_appl)
        defect_F = np.linalg.norm(F_stress + F_appl) / np.linalg.norm(F_appl)
        defect_M = abs(Mz_stress + Mz_appl) / abs(Mz_appl)
        load_F = np.linalg.norm(F_appl - F_ext) / np.linalg.norm(F_ext)
        load_M = abs(Mz_appl - Mz_ext) / abs(Mz_ext)
        print(f"[{run_tag(args)}] n_dofs = {n_dofs}")
        print(f"  F_ext (nominal)   = {F_ext}, Mz_ext = {Mz_ext:.6e}")
        print(f"  F_applied         = {F_appl}, Mz_appl = {Mz_appl:.6e}")
        print(f"  load consistency  = {load_F:.3e} (F), {load_M:.3e} (Mz)")
        print(f"  equilibrium err   = {eq_F:.3e} (F), {eq_M:.3e} (Mz)   [consistent reactions]")
        print(f"  flux defect       = {defect_F:.3e} (F), {defect_M:.3e} (Mz)   [stress reactions]")
        print(f"  compliance        = {compliance:.12e}")
        print(f"  u_corner          = {u_corner} at {corner_point} "
              f"(owners {corner_info['n_owners']}, spread {corner_info['spread']:.2e})")
        print(f"  written to {out}")


if __name__ == "__main__":
    main()
