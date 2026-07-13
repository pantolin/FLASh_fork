"""k x k subdivision of the wrench coarse mesh for h-refinement studies.

Two modes:
- "window": each child carries a 1/k x 1/k window of the parent cell's TPMS
  period (same physical lattice for every k) -- used as the h-refined
  reference of the validation study.
- "lattice": each child is a full lattice cell with its own threshold corners
  (a k times finer lattice) -- reproduces the paper's dense wrench (k = 9).

Supporting module for wrench_refined.py, used to reproduce Figure 20(c,d)
and Figure 21 (Section 5.2.2, "Lattice wrench").

The subdivided mesh duck-types WrenchCoarseMesh, so GlobalDofsManager and the
QoI functions work unchanged; only the Subdomain construction loop is custom
(each child needs its own level-set callable).
"""

from typing import Callable

import numpy as np
import qugar.cpp
from mpi4py import MPI

from FLASh.mesh import BezierElement
from FLASh.mesh.global_dofs_manager import GlobalDofsManager
from FLASh.mesh.subdomain import Subdomain

from example_5_utils import WrenchCoarseMesh

dtype = np.float64

RHO_VOID = 1e-10


def _blossom_1d(P: np.ndarray, a: float, b: float) -> np.ndarray:
    """Control points of a quadratic Bezier restricted to [a, b] (axis 0)."""

    def blossom(s: float, t: float) -> np.ndarray:
        return ((1 - s) * (1 - t)) * P[0] + ((1 - s) * t + s * (1 - t)) * P[1] + (s * t) * P[2]

    return np.stack([blossom(a, a), blossom(a, b), blossom(b, b)])


def subdivide_bezier(cp: np.ndarray, i: int, j: int, k: int) -> np.ndarray:
    """Restrict a quadratic Bezier patch cp[ix, iy, :] to child block (i, j) of k x k."""

    ax, bx = i / k, (i + 1) / k
    ay, by = j / k, (j + 1) / k

    cp = _blossom_1d(cp, ax, bx)
    cp = _blossom_1d(cp.transpose(1, 0, 2), ay, by).transpose(1, 0, 2)

    return cp


def window_levelset(levelset: Callable, i: int, j: int, k: int) -> Callable:
    """Level-set callable exposing the (i, j) window of the parent cell's period.

    Subdomain calls levelset(params, [0,0], [1,1]); the returned wrapper ignores
    those box arguments and shifts/scales the QUGaR affine so that child-local
    coordinates xi see the parent coordinate (xi + (i, j)) / k. The same affine
    drives the dim_linear threshold interpolation, so passing the parent's four
    corner thresholds reproduces the parent's bilinear field exactly.
    """

    p0 = np.array([-float(i), -float(j)], dtype=dtype)
    p1 = np.array([float(k - i), float(k - j)], dtype=dtype)

    def wrapped(params, xmin, xmax):
        return levelset(params, p0, p1)

    return wrapped


class RefinedWrenchMesh(WrenchCoarseMesh):
    """k x k subdivision of a WrenchCoarseMesh, built from parent connectivity."""

    def __init__(self, parent: WrenchCoarseMesh, parent_geometry, k: int) -> None:

        self.parent = parent
        self.k = k

        n_v = parent._n
        n_e = len(parent.edge_vertex_conn)
        n_c = parent._N

        self._edge_children = np.arange(n_e * k).reshape(n_e, k)
        self._child_parent = [(c, i, j) for c in range(n_c) for j in range(k) for i in range(k)]

        vertex_ids, coords = self._build_vertices(parent, parent_geometry, n_v, n_e, n_c)
        cell_vertex, cell_edge, edge_vertex = self._build_connectivity(parent, vertex_ids, n_e, n_c)

        super().__init__(cell_vertex, cell_edge, edge_vertex, coords)

    def _grid_vertex_ids(self, parent: WrenchCoarseMesh, cell: int, n_v: int, n_e: int) -> np.ndarray:
        """Global child-vertex ids of the (k+1) x (k+1) grid of one parent cell."""

        k = self.k
        ids = np.full((k + 1, k + 1), -1, dtype=np.int64)

        cv = np.asarray(parent.cell_vertex_conn[cell])
        ids[0, 0], ids[k, 0], ids[0, k], ids[k, k] = cv[0], cv[1], cv[2], cv[3]

        edge_slots = {0: ((0, 1), lambda m: (m, 0)),
                      1: ((0, 2), lambda m: (0, m)),
                      2: ((1, 3), lambda m: (k, m)),
                      3: ((2, 3), lambda m: (m, k))}

        ce = np.asarray(parent.cell_edge_conn[cell])
        for be, ((sa, sb), place) in edge_slots.items():
            e = int(ce[be])
            ev = np.asarray(parent.edge_vertex_conn[e])
            forward = ev[0] == cv[sa]
            assert forward or ev[0] == cv[sb], f"edge {e} does not match cell {cell}"
            for m in range(1, k):
                mm = m if forward else k - m
                gi, gj = place(m)
                ids[gi, gj] = n_v + e * (k - 1) + (mm - 1)

        base = n_v + len(parent.edge_vertex_conn) * (k - 1) + cell * (k - 1) ** 2
        for j in range(1, k):
            for i in range(1, k):
                ids[i, j] = base + (j - 1) * (k - 1) + (i - 1)

        assert (ids >= 0).all()
        return ids

    def _build_vertices(self, parent, parent_geometry, n_v: int, n_e: int, n_c: int):
        k = self.k
        n_total = n_v + n_e * (k - 1) + n_c * (k - 1) ** 2
        coords = np.zeros((n_total, 2), dtype=dtype)
        vertex_ids = []

        params = (np.arange(k + 1) / k).astype(dtype)
        gx, gy = np.meshgrid(params, params, indexing="ij")
        grid_pts = np.column_stack([gx.ravel(), gy.ravel()])

        for cell in range(n_c):
            ids = self._grid_vertex_ids(parent, cell, n_v, n_e)
            phys = parent_geometry.get_bezier_element(cell).evaluate(grid_pts)[:, :2]
            coords[ids.ravel()] = phys
            vertex_ids.append(ids)

        self._cell_grid_ids = vertex_ids
        return vertex_ids, coords

    def _build_connectivity(self, parent, vertex_ids, n_e: int, n_c: int):
        k = self.k

        n_child_edges = n_e * k + n_c * 2 * k * (k - 1)
        edge_vertex = np.zeros((n_child_edges, 2), dtype=np.int64)
        edge_set = np.zeros(n_child_edges, dtype=bool)

        cell_vertex = np.zeros((n_c * k * k, 4), dtype=np.int64)
        cell_edge = np.zeros((n_c * k * k, 4), dtype=np.int64)

        for cell in range(n_c):
            ids = vertex_ids[cell]
            cv = np.asarray(parent.cell_vertex_conn[cell])
            ce = np.asarray(parent.cell_edge_conn[cell])

            def parent_edge_child(be: int, m: int) -> int:
                e = int(ce[be])
                ev = np.asarray(parent.edge_vertex_conn[e])
                slots = {0: (0, 1), 1: (0, 2), 2: (1, 3), 3: (2, 3)}[be]
                forward = ev[0] == cv[slots[0]]
                mm = m if forward else k - 1 - m
                return int(self._edge_children[e, mm])

            base = n_e * k + cell * 2 * k * (k - 1)
            # interior horizontal edges: (i, j)-(i+1, j), j = 1..k-1
            def h_edge(i: int, j: int) -> int:
                if j == 0:
                    return parent_edge_child(0, i)
                if j == k:
                    return parent_edge_child(3, i)
                return base + (j - 1) * k + i

            # interior vertical edges: (i, j)-(i, j+1), i = 1..k-1
            def v_edge(i: int, j: int) -> int:
                if i == 0:
                    return parent_edge_child(1, j)
                if i == k:
                    return parent_edge_child(2, j)
                return base + k * (k - 1) + (i - 1) * k + j

            for j in range(k):
                for i in range(k):
                    child = cell * k * k + j * k + i
                    cell_vertex[child] = [ids[i, j], ids[i + 1, j], ids[i, j + 1], ids[i + 1, j + 1]]
                    cell_edge[child] = [h_edge(i, j), v_edge(i, j), v_edge(i + 1, j), h_edge(i, j + 1)]

                    pairs = {
                        h_edge(i, j): (ids[i, j], ids[i + 1, j]),
                        v_edge(i, j): (ids[i, j], ids[i, j + 1]),
                        v_edge(i + 1, j): (ids[i + 1, j], ids[i + 1, j + 1]),
                        h_edge(i, j + 1): (ids[i, j + 1], ids[i + 1, j + 1]),
                    }
                    for e, (a, b) in pairs.items():
                        if edge_set[e]:
                            assert {int(edge_vertex[e, 0]), int(edge_vertex[e, 1])} == {int(a), int(b)}
                        else:
                            edge_vertex[e] = [a, b]
                            edge_set[e] = True

        assert edge_set.all()
        return cell_vertex, cell_edge, edge_vertex

    def child_edges_of(self, parent_edges: np.ndarray) -> np.ndarray:
        return self._edge_children[np.atleast_1d(parent_edges)].ravel()


class RefinedWrenchGeometry:
    """Duck-typed geometry over a RefinedWrenchMesh.

    In "window" mode `get_cell_parameters` returns the parent's corner
    thresholds (the window wrapper makes dim_linear reproduce the parent's
    bilinear field); in "lattice" mode the standard per-child corner values of
    the child-vertex field set via `coarse_mesh.set_parameter_field`.
    """

    def __init__(self, parent_geometry, coarse_mesh: RefinedWrenchMesh, mode: str) -> None:

        assert mode in ("window", "lattice")

        self.parent_geometry = parent_geometry
        self.coarse_mesh = coarse_mesh
        self.mode = mode

        self.degree = parent_geometry.degree
        self.basis_degree = parent_geometry.basis_degree
        self.levelset = parent_geometry.levelset

        self._parent_cp = {}

    def get_bezier_element(self, child: int) -> BezierElement:

        cell, i, j = self.coarse_mesh._child_parent[child]

        if cell not in self._parent_cp:
            self._parent_cp[cell] = self.parent_geometry.get_bezier_element(cell).control_points

        cp = subdivide_bezier(self._parent_cp[cell], i, j, self.coarse_mesh.k)
        return BezierElement(self.degree, cp)

    def get_cell_levelset(self, child: int) -> Callable:

        if self.mode == "lattice":
            return self.levelset

        cell, i, j = self.coarse_mesh._child_parent[child]
        return window_levelset(self.levelset, i, j, self.coarse_mesh.k)

    def get_cell_parameters(self, child: int) -> np.ndarray:

        if self.mode == "lattice":
            return self.coarse_mesh.get_cell_parameters(child)

        cell, _, _ = self.coarse_mesh._child_parent[child]
        return self.parent_geometry.coarse_mesh.get_cell_parameters(cell)


def create_refined_gdm(
    geometry: RefinedWrenchGeometry,
    linear_pde,
    communicators,
    opts: dict | None = None,
) -> GlobalDofsManager:
    """Custom Subdomain construction loop (per-child level-set callables)."""

    opts = opts or {}
    subdomain_opts = opts.get("subdomain_opts", None)

    coarse_mesh = geometry.coarse_mesh
    N = coarse_mesh._N
    degree = geometry.basis_degree

    size = communicators.global_comm.Get_size()
    rank = communicators.global_comm.Get_rank()
    counts = [N // size + (1 if r < N % size else 0) for r in range(size)]
    starts = np.cumsum([0] + counts[:-1])
    process_subdomains = np.arange(starts[rank], starts[rank] + counts[rank])

    n_void = 0
    subdomains = []
    for s_id in process_subdomains:
        pts = coarse_mesh.get_cell_vertex_points(int(s_id))
        params = geometry.get_cell_parameters(int(s_id))
        levelset = geometry.get_cell_levelset(int(s_id))

        opts_child = subdomain_opts
        if _is_void(levelset, params):
            # A fully void child has zero local operators, which makes the
            # local/global solves singular. A negligible stabilization gives
            # it fictitious stiffness RHO_VOID * K_full; the perturbation of
            # the material solution is O(RHO_VOID).
            opts_child = dict(subdomain_opts or {})
            if not opts_child.get("stabilize", False):
                opts_child["stabilize"] = True
                opts_child["stabilization"] = RHO_VOID
            n_void += 1

        subdomains.append(
            Subdomain(
                [1, 1],
                degree,
                2,
                pts[0],
                pts[3],
                params,
                levelset,
                linear_pde,
                geometry.get_bezier_element(int(s_id)),
                opts=opts_child,
            )
        )

    n_void = communicators.global_comm.allreduce(n_void, op=MPI.SUM)
    if rank == 0 and n_void:
        print(f"void children regularized with rho = {RHO_VOID}: {n_void} / {N}")

    return GlobalDofsManager(geometry, subdomains, linear_pde, communicators)


def _is_void(levelset: Callable, params: np.ndarray) -> bool:
    grid = qugar.cpp.create_cart_grid([np.array([0.0, 1.0], dtype=dtype)] * 2)
    impl_func = levelset(params, np.array([0.0, 0.0]), np.array([1.0, 1.0])).cpp_object
    unf = qugar.cpp.create_unfitted_impl_domain(impl_func, grid)
    quad = qugar.cpp.create_quadrature(unf, np.array([0]), 4)
    return quad.weights.size == 0


def setup_solver_with_gdm(solver, gdm: GlobalDofsManager) -> None:
    """Replacement for BaseSolver.setup() when the gdm is built externally."""

    start = MPI.Wtime()
    solver.gbl_dofs_mngr = gdm
    solver.stats["setup time"] = MPI.Wtime() - start
    solver._setup_extra()
