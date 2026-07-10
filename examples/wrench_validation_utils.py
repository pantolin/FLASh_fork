"""Shared setup and QoI computations for the wrench validation study.

QoIs: total compliance, consistent (residual-based) reactions on the Dirichlet
boundary, stress-integrated reactions on the Dirichlet boundary, and the
resultant of the applied Neumann traction (used as exact equilibrium reference).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import qugar.cpp
from mpi4py import MPI
from scipy.io import loadmat

from FLASh.mesh import gyroid
from FLASh.pde import Elasticity
from FLASh.pde.linear_pde import create_facet_quadrature
from FLASh.rom import MDEIM

from _paths import DATA_ROOT, ROM_DATA_DIR
from example_5_utils import WrenchGeometry

dtype = np.float64

TRACTION_VALUE: tuple[float, float] = (0.0, -0.001)
PARAM_SEED = 20260707
ROM_FAMILY = "schoen_iwp_3"

# Verified against the QUGaR-assembled cores: the schoen_iwp_3 MDEIM data was
# trained on the threshold box [-2.5, 2.5]^4 (core errors ~1e-4 with this box,
# ~1e-1 with the [0.1, 0.9]^4 box used in example_5.py).
ROM_MDEIM_N = 2
ROM_MDEIM_P = 6
ROM_BOX: tuple[float, float] = (-2.5, 2.5)

_FACE_PARAM_ENDPOINTS = {
    0: np.array([[0.0, 0.0], [0.0, 1.0]]),
    1: np.array([[1.0, 0.0], [1.0, 1.0]]),
    2: np.array([[0.0, 0.0], [1.0, 0.0]]),
    3: np.array([[0.0, 1.0], [1.0, 1.0]]),
}

_FACE_TANGENT_INDEX = [1, 1, 0, 0]

_FACE_CORNER_SLOTS = {0: (0, 2), 1: (1, 3), 2: (0, 1), 3: (2, 3)}

_QUGAR_TO_BASIX = {2: 0, 0: 1, 1: 2, 3: 3}


@dataclass
class WrenchCase:
    geometry: WrenchGeometry
    pde: Elasticity
    edges_dir: np.ndarray
    edges_neu: np.ndarray
    center: np.ndarray
    traction: Callable


def _points_in_array(x: np.ndarray, y: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(2, 1)
        single_point = True
    else:
        single_point = False

    diff = x[:, :, None] - y[:, None, :]
    dist = np.linalg.norm(diff, axis=0)
    mask = np.any(dist < tol, axis=1)

    return mask[0] if single_point else mask


def _traction(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (TRACTION_VALUE[0] + 0 * X[0], TRACTION_VALUE[1] + 0 * X[0])


def _chord_edge_centers(pts: np.ndarray) -> list[np.ndarray]:
    """Physical chord midpoints per basix edge, as the fixed Subdomain computes them."""

    v = pts[:, :2]
    return [0.5 * (v[0] + v[1]), 0.5 * (v[0] + v[2]), 0.5 * (v[1] + v[3]), 0.5 * (v[2] + v[3])]


def _verify_neumann_marker(coarse_mesh, edges_neu: np.ndarray, marker: Callable) -> None:
    """Assert the marker hits exactly the faces descending from `edges_neu`."""

    expected: dict[int, set[int]] = {}
    for edge in np.atleast_1d(edges_neu):
        cell, face = _edge_to_cell_face(coarse_mesh, int(edge))
        expected.setdefault(cell, set()).add(_QUGAR_TO_BASIX[face])

    for cell in range(coarse_mesh._N):
        centers = _chord_edge_centers(coarse_mesh.get_cell_vertex_points(cell))
        hits = {k for k, c in enumerate(centers) if marker(np.asarray(c))}
        assert hits == expected.get(cell, set()), (
            f"Neumann marker mismatch on cell {cell}: "
            f"hits {hits}, expected {expected.get(cell, set())}"
        )


def _load_parameter_field(
    n_vertices: int,
    params_file: Path,
    param_range: tuple[float, float],
    seed: int,
) -> np.ndarray:
    comm = MPI.COMM_WORLD
    parameter_array = None

    if comm.Get_rank() == 0:
        if params_file.exists():
            parameter_array = np.load(params_file)
            assert parameter_array.shape == (n_vertices,)
        else:
            rng = np.random.default_rng(seed)
            parameter_array = rng.uniform(param_range[0], param_range[1], n_vertices)
            params_file.parent.mkdir(exist_ok=True, parents=True)
            np.save(params_file, parameter_array)

    return comm.bcast(parameter_array, root=0)


def load_wrench_geometry(degree: int) -> tuple[WrenchGeometry, np.ndarray, np.ndarray]:
    """Load Wrench.mat and build the 90-cell geometry; returns (geometry, edges_dir, edges_neu)."""

    data = loadmat(str(DATA_ROOT / "wrench" / "Wrench.mat"))

    nodes = data["nodes"]
    eleme_coefs = data["eleme_coefs"]

    conn_eleme_nodes = (data["conn_eleme_nodes"] - 1)[:, [0, 2, 1, 3]]
    conn_eleme_edges = (data["conn_eleme_edges"] - 1)[:, [2, 0, 1, 3]]
    conn_edges_nodes = data["conn_edges_nodes"] - 1

    edges_neu = np.squeeze(data["edges_neu"] - 1)
    edges_dir = np.squeeze(data["edges_dir"] - 1)

    geometry = WrenchGeometry(
        conn_eleme_nodes,
        conn_eleme_edges,
        conn_edges_nodes,
        nodes,
        eleme_coefs,
        gyroid.SchoenIWP().make_function(),
        {"basis_degree": degree, "spline_degree": 2},
    )

    return geometry, edges_dir, edges_neu


def build_case(
    degree: int,
    rom: bool,
    params_file: Path,
    param_range: tuple[float, float] = (0.1, 0.9),
    seed: int = PARAM_SEED,
) -> WrenchCase:
    """Build the coarse wrench geometry, PDE, and boundary data.

    The per-vertex threshold parameter field is loaded from `params_file`
    (created with a fixed seed on first use), so that every run of the study
    uses the identical geometry.
    """

    geometry, edges_dir, edges_neu = load_wrench_geometry(degree)

    coarse_mesh = geometry.coarse_mesh
    parameter_array = _load_parameter_field(coarse_mesh._n, params_file, param_range, seed)
    coarse_mesh.set_parameter_field(parameter_array)

    nodes_dir = np.unique(np.array(coarse_mesh.edge_vertex_conn)[edges_dir].flatten())
    points_dir = np.vstack([
        coarse_mesh.vertex_coordinates[nodes_dir],
        coarse_mesh.edge_coordinates[edges_dir],
    ]).T

    points_neu = coarse_mesh.edge_coordinates[edges_neu][:, :2].T

    def h_bc(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (0 + 0 * X[0], 0 + 0 * X[0])

    def source(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (0.0 + 0.0 * X[0], 0.0 + 0.0 * X[0])

    def neu_marker(x: np.ndarray) -> np.ndarray:
        return _points_in_array(x[:2], points_neu[:2])

    _verify_neumann_marker(coarse_mesh, edges_neu, neu_marker)

    exterior_bc = [
        (0, h_bc, lambda x: _points_in_array(x[:2], points_dir[:2]), 0),
        (1, _traction, neu_marker, 1),
    ]

    pde_kwargs: dict = {}
    if rom:
        assert degree == 8, "ROM data is trained at basis_degree = 8"

        d_rom = 4
        p0 = np.array([ROM_BOX[0]] * d_rom)
        p1 = np.array([ROM_BOX[1]] * d_rom)

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

    pde = Elasticity(
        exterior_bc=exterior_bc,
        source=source,
        E=5,
        nu=0.25,
        **pde_kwargs,
    )

    center = coarse_mesh.vertex_coordinates[nodes_dir].mean(axis=0)[:2]

    return WrenchCase(
        geometry=geometry,
        pde=pde,
        edges_dir=edges_dir,
        edges_neu=edges_neu,
        center=center,
        traction=_traction,
    )


def compute_compliance(solver) -> float:
    """Total compliance C = f^T u (valid for homogeneous Dirichlet data)."""

    gdm = solver.gbl_dofs_mngr
    us = solver.get_solution()

    c_local = sum(
        float(us[s_ind] @ gdm.subdomains[s_ind].f)
        for s_ind in range(len(gdm.subdomains))
    )

    return solver.communicators.global_comm.allreduce(c_local, op=MPI.SUM)


def compute_consistent_reactions(solver, center: np.ndarray) -> tuple[np.ndarray, float]:
    """Residual-based reactions (K u - f) on the Dirichlet DOFs.

    Satisfies equilibrium with the applied load to solver precision at any
    degree; used as an implementation sanity check.
    """

    gdm = solver.gbl_dofs_mngr
    us = solver.get_solution()
    dir_dofs = gdm.get_dirichlet_boundary_dofs()

    F = np.zeros(2, dtype=dtype)
    Mz = 0.0

    for s_ind, s_id in enumerate(gdm.process_subdomains):
        sub = gdm.subdomains[s_ind]
        r = sub.K @ us[s_ind] - sub.f

        R = gdm.create_R(s_id)
        mask = np.isin(R, dir_dofs)
        if not mask.any():
            continue

        loc = np.asarray(sub.boundary_dofs)[mask]
        r_c = r[loc]
        node, comp = loc // 2, loc % 2
        X = sub._map.evaluate(sub._basis.get_nodes())[:, :2][node]

        np.add.at(F, comp, r_c)
        arm = np.where(comp == 1, X[:, 0] - center[0], -(X[:, 1] - center[1]))
        Mz += float(arm @ r_c)

    comm = solver.communicators.global_comm
    F = comm.allreduce(F, op=MPI.SUM)
    Mz = comm.allreduce(Mz, op=MPI.SUM)

    return F, Mz


def _edge_to_cell_face(coarse_mesh, edge: int) -> tuple[int, int]:
    """Owning cell and QUGaR face id (0: x=0, 1: x=1, 2: y=0, 3: y=1) of a boundary edge."""

    owners = coarse_mesh.edge_cell_conn[edge]
    assert len(owners) == 1, f"edge {edge} is not a boundary edge"
    cell = int(owners[0])

    edge_vertices = set(np.asarray(coarse_mesh.edge_vertex_conn[edge]).tolist())
    cell_vertices = np.asarray(coarse_mesh.cell_vertex_conn[cell])

    for face, (a, b) in _FACE_CORNER_SLOTS.items():
        if {int(cell_vertices[a]), int(cell_vertices[b])} == edge_vertices:
            return cell, face

    raise ValueError(f"edge {edge} does not match any face of cell {cell}")


def _local_index(gdm, cell: int) -> int | None:
    proc = gdm.process_subdomains
    if proc[0] <= cell < proc[0] + len(proc):
        return int(cell - proc[0])
    return None


def _check_face_geometry(sub, coarse_mesh, edge: int, face: int) -> None:
    endpoints = sub._map.evaluate(_FACE_PARAM_ENDPOINTS[face])[:, :2]
    vertices = coarse_mesh.vertex_coordinates[coarse_mesh.edge_vertex_conn[edge]][:, :2]

    direct = np.linalg.norm(endpoints - vertices, axis=1).max()
    swapped = np.linalg.norm(endpoints - vertices[::-1], axis=1).max()
    assert min(direct, swapped) < 1e-8, f"face/edge mismatch for edge {edge}"


def _facet_quadrature(sub, face: int, n_quad: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unf_domain = sub.create_qugar_mesh()
    facet_quad = qugar.cpp.create_facets_quadrature_exterior_integral(
        unf_domain, np.array([0]), np.array([face]), n_quad
    )
    return create_facet_quadrature(facet_quad.points, facet_quad.weights, face)


def compute_stress_reactions(
    solver,
    coarse_edges: np.ndarray,
    center: np.ndarray,
    n_quad: int = 20,
) -> tuple[np.ndarray, float]:
    """Reactions on `coarse_edges` by boundary integration of sigma(u_h) . n."""

    gdm = solver.gbl_dofs_mngr
    coarse_mesh = solver.geometry.coarse_mesh
    us = solver.get_solution()

    lam = solver.linear_pde.lambda_
    mu = solver.linear_pde.mu

    F = np.zeros(2, dtype=dtype)
    Mz = 0.0

    for edge in np.atleast_1d(coarse_edges):
        cell, face = _edge_to_cell_face(coarse_mesh, int(edge))
        s_ind = _local_index(gdm, cell)
        if s_ind is None:
            continue

        sub = gdm.subdomains[s_ind]
        _check_face_geometry(sub, coarse_mesh, int(edge), face)

        pts, w, n_ref = _facet_quadrature(sub, face, n_quad)
        if w.size == 0:
            continue

        d = np.array(sub._basis.evaluate_derivative(pts))
        u = us[s_ind]
        g = np.einsum("imj,kj->kim", d, np.stack([u[0::2], u[1::2]]))

        ijf = sub._map.evaluate_jacobian_inverse(pts)
        Gu = np.einsum("kim,mid->mkd", g, ijf)

        eps = 0.5 * (Gu + Gu.transpose(0, 2, 1))
        tr = np.trace(eps, axis1=1, axis2=2)
        sig = 2.0 * mu * eps
        sig[:, 0, 0] += lam * tr
        sig[:, 1, 1] += lam * tr

        det = sub._map.evaluate_jacobian_determinant(pts)
        v = np.abs(det)[:, None] * np.einsum("mid,i->md", ijf, n_ref[0])

        t_vec = np.einsum("mkd,md->mk", sig, v)
        F += w @ t_vec

        x = sub._map.evaluate(pts)[:, :2]
        Mz += float(w @ ((x[:, 0] - center[0]) * t_vec[:, 1] - (x[:, 1] - center[1]) * t_vec[:, 0]))

    comm = solver.communicators.global_comm
    F = comm.allreduce(F, op=MPI.SUM)
    Mz = comm.allreduce(Mz, op=MPI.SUM)

    return F, Mz


def compute_external_resultant(
    solver,
    coarse_edges: np.ndarray,
    traction: Callable,
    center: np.ndarray,
    n_quad: int = 20,
) -> tuple[np.ndarray, float]:
    """Resultant force and moment of the traction applied on `coarse_edges`."""

    gdm = solver.gbl_dofs_mngr
    coarse_mesh = solver.geometry.coarse_mesh

    F = np.zeros(2, dtype=dtype)
    Mz = 0.0

    for edge in np.atleast_1d(coarse_edges):
        cell, face = _edge_to_cell_face(coarse_mesh, int(edge))
        s_ind = _local_index(gdm, cell)
        if s_ind is None:
            continue

        sub = gdm.subdomains[s_ind]
        _check_face_geometry(sub, coarse_mesh, int(edge), face)

        pts, w, _ = _facet_quadrature(sub, face, n_quad)
        if w.size == 0:
            continue

        ds = sub._map.evaluate_arclen(pts)[:, _FACE_TANGENT_INDEX[face]]
        x = sub._map.evaluate(pts)[:, :2]
        t_vals = np.array(traction(x.T))

        F += (t_vals * (w * ds)).sum(axis=1)
        Mz += float(
            (w * ds)
            @ ((x[:, 0] - center[0]) * t_vals[1] - (x[:, 1] - center[1]) * t_vals[0])
        )

    comm = solver.communicators.global_comm
    F = comm.allreduce(F, op=MPI.SUM)
    Mz = comm.allreduce(Mz, op=MPI.SUM)

    return F, Mz


def compute_point_displacement(solver, point: np.ndarray) -> tuple[np.ndarray, dict]:
    """Displacement at a coarse-mesh vertex, plus diagnostic info.

    Returns the averaged (ux, uy) over all subdomains owning the vertex and a
    dict with the spread across owners and the active-area fraction of each
    owning cell (if the corner region is void, the value is the
    fictitious-domain extension of the solution).
    """

    gdm = solver.gbl_dofs_mngr
    coarse_mesh = solver.geometry.coarse_mesh
    us = solver.get_solution()

    verts = coarse_mesh.locate_vertices(
        lambda x: _points_in_array(x[:2], np.asarray(point).reshape(2, 1))
    )
    assert len(verts) == 1, f"point {point} matches {len(verts)} vertices"
    vertex = int(verts[0])

    values: list[list[float]] = []
    fractions: list[float] = []

    for cell in coarse_mesh.vertex_cell_conn[vertex]:
        s_ind = _local_index(gdm, int(cell))
        if s_ind is None:
            continue

        sub = gdm.subdomains[s_ind]
        slot = int(np.where(np.asarray(coarse_mesh.cell_vertex_conn[int(cell)]) == vertex)[0][0])
        dofs = sub.vertices_dofs[slot]
        values.append([float(v) for v in us[s_ind][dofs]])

        quad = qugar.cpp.create_quadrature(sub.create_qugar_mesh(), np.array([0]), 8)
        fractions.append(float(np.sum(quad.weights)))

    comm = solver.communicators.global_comm
    values = [v for rank_vals in comm.allgather(values) for v in rank_vals]
    fractions = [f for rank_vals in comm.allgather(fractions) for f in rank_vals]

    material = np.array([v for v, f in zip(values, fractions) if f > 1e-12])
    arr = material if material.size else np.array(values)
    assert arr.size > 0, f"no subdomain owns vertex {vertex}"
    u = arr.mean(axis=0)
    spread = float(np.abs(arr - u).max()) if len(arr) > 1 else 0.0

    return u, {
        "vertex": vertex,
        "n_owners": len(arr),
        "spread": spread,
        "material": bool(material.size),
        "cell_active_fractions": fractions,
    }


def compute_boundary_mean_displacement(
    solver,
    coarse_edges: np.ndarray,
    n_quad: int = 20,
) -> np.ndarray:
    """Mean displacement over the active part of the given boundary edges."""

    gdm = solver.gbl_dofs_mngr
    coarse_mesh = solver.geometry.coarse_mesh
    us = solver.get_solution()

    u_int = np.zeros(2, dtype=dtype)
    length = 0.0

    for edge in np.atleast_1d(coarse_edges):
        cell, face = _edge_to_cell_face(coarse_mesh, int(edge))
        s_ind = _local_index(gdm, cell)
        if s_ind is None:
            continue

        sub = gdm.subdomains[s_ind]
        pts, w, _ = _facet_quadrature(sub, face, n_quad)
        if w.size == 0:
            continue

        ds = sub._map.evaluate_arclen(pts)[:, _FACE_TANGENT_INDEX[face]]
        basis_vals = sub._basis.evaluate(pts)
        u = us[s_ind]
        u_vals = np.stack([basis_vals @ u[0::2], basis_vals @ u[1::2]])

        u_int += (u_vals * (w * ds)).sum(axis=1)
        length += float(w @ ds)

    comm = solver.communicators.global_comm
    u_int = comm.allreduce(u_int, op=MPI.SUM)
    length = comm.allreduce(length, op=MPI.SUM)

    return u_int / length


def compute_applied_load(solver, center: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact resultant force and moment of the assembled (discrete) load.

    Obtained by pairing the load vectors with the nodal rigid-body fields;
    both fields lie in the discretization space, so this is the load the
    discrete solution is actually in equilibrium with. It differs from the
    nominal traction resultant by the corner spillover of the boundary-mass
    Neumann assembly.
    """

    gdm = solver.gbl_dofs_mngr

    F_local = np.zeros(2, dtype=dtype)
    Mz_local = 0.0

    for sub in gdm.subdomains:
        fx, fy = sub.f[0::2], sub.f[1::2]
        X = sub._map.evaluate(sub._basis.get_nodes())[:, :2]

        F_local[0] += fx.sum()
        F_local[1] += fy.sum()
        Mz_local += float((X[:, 0] - center[0]) @ fy - (X[:, 1] - center[1]) @ fx)

    comm = solver.communicators.global_comm
    return comm.allreduce(F_local, op=MPI.SUM), comm.allreduce(Mz_local, op=MPI.SUM)
