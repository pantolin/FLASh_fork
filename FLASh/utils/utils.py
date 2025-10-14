import numpy as np
import numpy.typing as npt

from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.fem
import dolfinx.mesh
import dolfinx.io
import ufl
import basix
import dolfinx

class Communicators:

    def __init__(self):
        self.petsc_comm = PETSc.COMM_WORLD
        self.global_comm = self.petsc_comm.tompi4py()
        self.self_comm = MPI.COMM_SELF

def write_solution(u: dolfinx.fem.Function, folder: str, filename_str: str) -> None:
    """Writes the given function to a VTX folder with extension ".pb".
    It can visualized by importing it in ParaView.

    Args:
        u (dolfinx.fem.Function): Function to dump into a VTX file.
        folder (str): Folder in which files are placed.
            It is created if it doesn't exist.
        filename_str (str): Name (without extension) of the folder containing
            the function.
    """

    results_folder = Path(folder)
    results_folder.mkdir(exist_ok=True, parents=True)

    filename = results_folder / filename_str
    filename = filename.with_suffix(".bp")

    V = u.function_space
    comm = V.mesh.comm

    with dolfinx.io.VTXWriter(comm, filename, [u]) as vtx:
        vtx.write(0.0)

def create_Cartesian_mesh_nodes(
    pts_1D: list[npt.NDArray[np.float64]],
    periodic: bool = False
) -> npt.NDArray[np.float64]:
    """Creates the coordinates matrix of the nodes of a 2D tensor-product
    Cartesian mesh.
    The ordering of the generated nodes follow the lexicographical
    ordering convetion.

    Args:
        pts_1D: Points coordinates along the two parametric directions.

    Returns:
        nodes: Coordinates of the nodes stored in a 2D np.ndarray.
            The rows correspond to the different points and columns
            to the coordinates.
    """

    assert len(pts_1D) == 2

    pts_x = pts_1D[0]
    pts_y = pts_1D[1]

    if periodic:
        pts_x = pts_x[:-1]

    x = np.meshgrid(pts_x, pts_y, indexing="xy")

    nodes = np.zeros((x[0].size, 2), dtype=x[0].dtype)
    for dir in range(2):
        nodes[:, dir] = x[dir].ravel()

    return nodes

def create_2D_tensor_prod_mesh_conn(
        n_cells: list[int],
        periodic: bool = False
    ) -> list[list[int]]:
    """Creates the cells' connectivity of a 2D Cartesian mesh made of
    linear quadrilaterals.

    Args:
        n_cells: Number of cells per direction in the Cartesian mesh.

    Returns:
        conn: Generated connectivity. It is a list, where
            every entry is a list of nodes ids.
            The connectivity of each cells follows the DOLFINx convention.
            See https://docs.fenicsproject.org/basix/v0.8.0/index.html.
    """

    assert len(n_cells) == 2 and n_cells[0] > 0 and n_cells[1] > 0

    if periodic:

        ix = np.arange(n_cells[0])
        iy = np.arange(n_cells[1])

        ix, iy = np.meshgrid(ix, iy)
        ix = ix.ravel()
        iy = iy.ravel()

        v1 = iy * n_cells[0] + ix
        v2 = iy * n_cells[0] + (ix + 1) % n_cells[0]
        v3 = (iy + 1) * n_cells[0] + ix
        v4 = (iy + 1) * n_cells[0] + (ix + 1) % n_cells[0]

        conn = np.array([v1, v2, v3, v4]).T

        return conn.tolist()

    else: 

        n_cells = np.array(n_cells)
        n_pts = n_cells + 1

        # First cell.
        first_cell = np.array([0, 1, n_pts[0], n_pts[0] + 1])

        # First line of cells.
        conn = first_cell + np.arange(0, n_cells[0]).reshape(-1, 1)

        # Full connecitivity
        conn = conn.ravel() + np.arange(0, n_pts[0] * n_cells[1], n_pts[0]).reshape(-1, 1)

        conn = conn.reshape(np.prod(n_cells), len(first_cell))
        return conn.tolist()

def create_2D_tensor_prod_mesh_element_to_edge_conn(
        n_cells: list[int],
        periodic: bool = False
    ) -> list[list[int]]:

    if periodic:

        ix = np.arange(n_cells[0])
        iy = np.arange(n_cells[1])

        ix, iy = np.meshgrid(ix, iy)
        ix = ix.ravel()
        iy = iy.ravel()

        f1 = iy * n_cells[0] + ix
        f2 = n_cells[0] * (n_cells[1] + 1) + iy * n_cells[0] + ix
        f3 = n_cells[0] * (n_cells[1] + 1) + iy * n_cells[0] + (ix + 1) % n_cells[0]
        f4 = (iy + 1) * n_cells[0] + ix

        conn = np.array([f1, f2, f3, f4]).T

        return conn.tolist()

    else: 

        ix = np.arange(n_cells[0])
        iy = np.arange(n_cells[1])

        ix, iy = np.meshgrid(ix, iy)
        ix = ix.ravel()
        iy = iy.ravel()

        f1 = iy * n_cells[0] + ix
        f2 = n_cells[0] * (n_cells[1] + 1) + iy * (n_cells[0] + 1) + ix
        f3 = n_cells[0] * (n_cells[1] + 1) + iy * (n_cells[0] + 1) + ix + 1
        f4 = (iy + 1) * n_cells[0] + ix

        conn = np.array([f1, f2, f3, f4]).T

        return conn.tolist()

def create_2D_tensor_prod_mesh_edge_to_node_conn(
        n_cells: list[int],
        periodic: bool = False
    ) -> list[list[int]]:

    if periodic:

        n1 = []
        n2 = []

        ix = np.arange(n_cells[0])
        iy = np.arange(n_cells[1]+1)

        ix, iy = np.meshgrid(ix, iy)
        ix = ix.ravel()
        iy = iy.ravel()

        n1.append(iy * n_cells[0] + ix)
        n2.append(iy * n_cells[0] + (ix +1) % n_cells[0])

        ix = np.arange(n_cells[0])
        iy = np.arange(n_cells[1])

        ix, iy = np.meshgrid(ix, iy)
        ix = ix.ravel()
        iy = iy.ravel()

        n1.append(iy * n_cells[0] + ix)
        n2.append((iy+1) * n_cells[0] + ix)

        n1 = np.hstack(n1)
        n2 = np.hstack(n2)

        conn = np.array([n1, n2]).T

        return conn.tolist()

    else: 

        n1 = []
        n2 = []

        ix = np.arange(n_cells[0])
        iy = np.arange(n_cells[1]+1)

        ix, iy = np.meshgrid(ix, iy)
        ix = ix.ravel()
        iy = iy.ravel()

        n1.append(iy * (n_cells[0] + 1) + ix)
        n2.append(iy * (n_cells[0] + 1) + ix +1)

        ix = np.arange(n_cells[0]+1)
        iy = np.arange(n_cells[1])

        ix, iy = np.meshgrid(ix, iy)
        ix = ix.ravel()
        iy = iy.ravel()

        n1.append(iy * (n_cells[0] + 1) + ix)
        n2.append((iy+1) * (n_cells[0] + 1) + ix)

        n1 = np.hstack(n1)
        n2 = np.hstack(n2)

        conn = np.array([n1, n2]).T

        return conn.tolist()

def create_2D_mesh(
    n: list[int],
    comm,
    p0: list[float] = [0.0, 0.0],
    p1: list[float] = [1.0, 1.0],
) -> dolfinx.mesh.Mesh:
    """Creates a 2D mesh of rectangular domain, using linear quadrilaterals,
    with n[0] and n[1] elements per direction, respectively. The rectangular
    domain is defined its left-bottom and rigth-top corners, p0, and p1,
    respectively.

    Args:
        n (list[int]): Number of elements per direction.
        p0 (list[float], optional): Left-bottom corner of the rectangular
            domain.  Defaults to [0.0, 0.0].
        p1 (list[float], optional): Right-top corner of the rectangular domain.
            Defaults to [1.0, 1.0].

    Returns:
        dolfinx.mesh.Mesh: Generated mesh.
    """

    assert len(n) == 2 and n[0] > 0 and n[1] > 0

    pts_0 = np.linspace(p0[0], p1[0], n[0] + 1)
    pts_1 = np.linspace(p0[1], p1[1], n[1] + 1)
    coords = create_Cartesian_mesh_nodes([pts_0, pts_1])

    conn = create_2D_tensor_prod_mesh_conn(n)

    domain = ufl.Mesh(
        basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2,), dtype=np.float64)
    )

    mesh = dolfinx.mesh.create_mesh(comm, conn, coords, domain)

    return mesh

def create_Cartesian_mesh_edges( 
    pts_1D: list[npt.NDArray[np.float64]],
    periodic: bool = False
) -> npt.NDArray[np.float64]:
    
    pts_x = pts_1D[0]
    pts_y = pts_1D[1]

    x = np.meshgrid(0.5*(pts_x[:-1]+pts_x[1:]), pts_y, indexing="xy")

    nodes_1 = np.zeros((x[0].size, 2), dtype=x[0].dtype)
    for dir in range(2):
        nodes_1[:, dir] = x[dir].ravel()

    if periodic:
        pts_x = pts_x[:-1]

    x = np.meshgrid(pts_x, 0.5*(pts_y[:-1]+pts_y[1:]), indexing="xy")

    nodes_2 = np.zeros((x[0].size, 2), dtype=x[0].dtype)
    for dir in range(2):
        nodes_2[:, dir] = x[dir].ravel()

    nodes = np.concatenate((nodes_1, nodes_2), axis = 0)

    return nodes
