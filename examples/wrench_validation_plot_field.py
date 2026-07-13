"""Off-screen displacement-magnitude figure and VTU export for wrench runs.

`build_field_meshes(solver)` turns a solved wrench solver into per-subdomain
PyVista grids in physical coordinates (point data: `uh`, `|u|`, `|u| [mm]`);
`render_field` / `save_field_vtu` produce the figure / merged VTU. The module
is import-safe (no argparse at import time) so refined drivers can reuse it.
Serial only.

Field rendering for Figure 20(a,b,c,d) (Section 5.2.2, "Lattice wrench").
"""

import argparse

import dolfinx.fem
import numpy as np
import pyvista as pv
import qugar.plot
import qugar.reparam
from mpi4py import MPI

from FLASh.utils import Communicators

from _paths import FIGS_DIR, RESULTS_DIR
from wrench_validation import PAPER_DIR, PAPER_RANGE, PAPER_SEED, SOLVERS, VALIDATION_DIR, run_tag
from wrench_validation_utils import build_case

dtype = np.float64


def build_field_meshes(solver) -> list[pv.UnstructuredGrid]:
    """Per-subdomain PyVista grids of the solution, in physical coordinates."""

    gdm = solver.gbl_dofs_mngr
    us = gdm.transform_to_fenicsx(solver.get_solution())

    dim = 2
    reparam_degree = 3
    meshes = []

    for s_ind, _ in enumerate(gdm.process_subdomains):
        sub = gdm.subdomains[s_ind]

        unf_mesh = sub.create_mesh()
        V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", sub._degree, (dim,)))
        uh = dolfinx.fem.Function(V)
        uh.x.array[:] = us[s_ind]

        reparam = qugar.reparam.create_reparam_mesh(unf_mesh, degree=reparam_degree, levelset=False)
        reparam_mesh = reparam.create_mesh()

        cmap0 = reparam_mesh.topology.index_map(reparam_mesh.topology.dim)
        if cmap0.size_local + cmap0.num_ghosts == 0:
            continue

        V_reparam = dolfinx.fem.functionspace(reparam_mesh, ("CG", reparam_degree, (dim,)))
        uh_reparam = dolfinx.fem.Function(V_reparam)

        cmap = reparam_mesh.topology.index_map(reparam_mesh.topology.dim)
        cells = np.arange(cmap.size_local + cmap.num_ghosts, dtype=np.int32)

        interpolation_data = dolfinx.fem.create_interpolation_data(V_reparam, V, cells, padding=1.0e-14)
        uh_reparam.interpolate_nonmatching(uh, cells, interpolation_data=interpolation_data)

        pv_mesh = qugar.plot.reparam_mesh_to_PyVista(reparam).get("reparam")
        pv_mesh.points = sub._map.evaluate(pv_mesh.points)

        u_plot = uh_reparam.x.array.reshape(-1, dim)
        pv_mesh.point_data["uh"] = np.hstack((u_plot, np.zeros((u_plot.shape[0], 1))))
        pv_mesh.point_data["|u|"] = np.linalg.norm(u_plot, axis=1)
        pv_mesh.point_data["|u| [mm]"] = 1000.0 * pv_mesh.point_data["|u|"]

        meshes.append(pv_mesh)

    return meshes


def save_field_vtu(meshes: list[pv.UnstructuredGrid], path) -> None:
    pv.merge(meshes).save(str(path))
    print(f"written {path}")


def render_field(
    meshes: list[pv.UnstructuredGrid],
    path,
    warp_factor: float = 1.0,
    resolution: tuple[int, int] = (2400, 1200),
    scalars: str = "|u| [mm]",
) -> None:
    warped = [m.warp_by_vector("uh", factor=warp_factor) for m in meshes]
    clim = [0.0, max(m.point_data[scalars].max() for m in warped)]

    plotter = pv.Plotter(off_screen=True, window_size=list(resolution))
    for i, m in enumerate(warped):
        plotter.add_mesh(
            m,
            scalars=scalars,
            clim=clim,
            show_edges=False,
            cmap="viridis",
            show_scalar_bar=(i == 0),
            scalar_bar_args={"title": scalars, "vertical": False, "fmt": "%.2f"},
        )
    plotter.view_xy()
    plotter.screenshot(str(path))
    print(f"written {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--solver", choices=SOLVERS, default="cholesky")
    parser.add_argument("--rom", action="store_true")
    parser.add_argument("--stabilization", type=float, default=0.0)
    parser.add_argument("--warp-factor", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, nargs=2, default=(2400, 1200))
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--write-vtu", action="store_true")
    parser.add_argument("--paper-field", action="store_true")
    return parser.parse_args()


def main() -> None:
    assert MPI.COMM_WORLD.Get_size() == 1, "run serially"

    args = parse_args()

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

    sbdmn_opts: dict = {"assemble": True, "n_quad_pts": max(8, args.degree + 2)}
    if args.stabilization > 0:
        sbdmn_opts["stabilize"] = True
        sbdmn_opts["stabilization"] = args.stabilization

    opts = {"global_dofs_manager_opts": {"subdomain_opts": sbdmn_opts}}

    solver = SOLVERS[args.solver](case.geometry, case.pde, communicators, opts=opts)
    solver.setup()
    solver.solve()

    meshes = build_field_meshes(solver)

    out_dir = PAPER_DIR if args.paper_field else VALIDATION_DIR
    if args.write_vtu:
        out_dir.mkdir(exist_ok=True, parents=True)
        save_field_vtu(meshes, out_dir / f"field_{run_tag(args)}.vtu")

    FIGS_DIR.mkdir(exist_ok=True, parents=True)
    out = args.out or f"wrench_displacement_{run_tag(args)}.png"
    render_field(meshes, FIGS_DIR / out, warp_factor=args.warp_factor,
                 resolution=tuple(args.resolution))


if __name__ == "__main__":
    main()
