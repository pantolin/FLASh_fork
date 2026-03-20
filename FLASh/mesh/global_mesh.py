"""Mesh utilities for coarse and parametric Cartesian meshes.

This module provides simple mesh representations and helpers for building
regular tensor-product meshes and extracting connectivity information.
"""

import numpy as np
import numpy.typing as npt

import mpi4py.MPI as MPI

from typing import Self, Callable

from FLASh.utils.utils import (
    create_2D_tensor_prod_mesh_conn,
    create_2D_tensor_prod_mesh_element_to_edge_conn,
    create_2D_tensor_prod_mesh_edge_to_node_conn,
    create_Cartesian_mesh_nodes,
    create_Cartesian_mesh_edges
)

def create_inverse_conn(conn):

    conn = np.asarray(conn)

    n = int(conn.max()) + 1

    inv_conn = [[] for _ in range(n)]

    for e, vs in enumerate(conn):
        for v in vs:
            inv_conn[v].append(e)

    inv_conn_arrays = [np.array(es) for es in inv_conn]

    return inv_conn_arrays


class CoarseMesh:
    """Simple representation of a coarse Cartesian mesh.

    This class stores the connectivity and coordinates for a Cartesian mesh
    (cells, edges, vertices) and provides utilities to set parameter fields and
    query cell/vertex relations.
    """

    def __init__(
        self,
        cell_vertex_connectivity,
        cell_edge_connectivity,
        edge_vertex_connectivity,
        vertex_coordinates,
        edge_coordinates,
        elem_2_verts,
        verts_coords
    ) -> None:

        self.cell_vertex_conn = list(np.array(cell_vertex_connectivity))
        self.cell_edge_conn = list(np.array(cell_edge_connectivity))
        self.edge_vertex_conn = list(np.array(edge_vertex_connectivity))
        self.vertex_coordinates = vertex_coordinates
        self.edge_coordinates = edge_coordinates

        self.elem_2_verts = elem_2_verts
        self.verts_coords = verts_coords

        self._N = len(self.cell_vertex_conn)
        self._n = vertex_coordinates.shape[0]

        self._create_inverse_connectivities()

        parameters = np.zeros((self._n))
        self.set_parameter_field(parameters)

    @staticmethod
    def create_cartesian_mesh(
        pts_1D: list[npt.NDArray[np.float64]],
        periodic: bool = False
    ) -> Self:

        n_cells = [pts.size - 1 for pts in pts_1D]

        vertex_coords = create_Cartesian_mesh_nodes(pts_1D, periodic)
        edge_coords = create_Cartesian_mesh_edges(pts_1D, periodic)

        c_2_v = create_2D_tensor_prod_mesh_conn(n_cells, periodic)
        c_2_e = create_2D_tensor_prod_mesh_element_to_edge_conn(n_cells, periodic)
        e_2_v = create_2D_tensor_prod_mesh_edge_to_node_conn(n_cells, periodic)

        elem_2_verts = create_2D_tensor_prod_mesh_conn(n_cells)
        verts_coords = create_Cartesian_mesh_nodes(pts_1D)

        return CoarseMesh(c_2_v, c_2_e, e_2_v, vertex_coords, edge_coords, elem_2_verts, verts_coords)
    
    def _create_inverse_connectivities(self) -> None:

        self.vertex_cell_conn = create_inverse_conn(self.cell_vertex_conn)
        self.edge_cell_conn = create_inverse_conn(self.cell_edge_conn)
        self.vertex_edge_conn = create_inverse_conn(self.edge_vertex_conn)

    def _create_edge_coordinates(self) -> None:

        coords = []
        v_coords = self.vertex_coordinates

        for vertices in self.edge_vertex_conn:
            
            v0 = v_coords[vertices[0]]
            v1 = v_coords[vertices[1]]

            coords.append(0.5*(v0+v1))

        self.edge_coordinates = np.array(coords)

    def set_parameter_field_from_function(self, fun: Callable) -> None:

        comm = MPI.COMM_WORLD

        if comm.Get_rank() == 0:
            parameter_array = fun(self.vertex_coordinates.T)
        else:
            parameter_array = None  

        parameter_array = comm.bcast(parameter_array, root=0)
        
        self.set_parameter_field(parameter_array)

    def set_parameter_field(self, parameter_array: np.ndarray[np.float64]) -> None:

        self._parameters = np.zeros((self._n, 4))

        for idx, vertices in enumerate(self.cell_vertex_conn):

            self._parameters[idx] = parameter_array[vertices]

    def set_parameter_field_values(self, cells_id: np.ndarray, values: list[np.ndarray]) -> None:

        for cell_id, value in zip(cells_id, values):

            self._parameters[cell_id] = value

    def get_cell_vertex_points(self, cell_id: int) -> np.ndarray:

        vertices = self.elem_2_verts[cell_id]

        return self.verts_coords[vertices]

    def get_cell_parameters(self, cell_id: int) -> np.ndarray:

        return self._parameters[cell_id]

    def locate_vertices(self, marker) -> np.ndarray:

        return np.where(marker(self.vertex_coordinates.T))[0]
    
    def locate_edges(self, marker) -> np.ndarray:

        return np.where(marker(self.edge_coordinates.T))[0]


class ParametricMesh:
    """Minimal parametric mesh representation used for interpolation and mapping.

    The mesh is defined by connectivity (elements to vertex indices) and the
    vertex coordinates.
    """

    def __init__(
      self,
      elem_2_verts,
      verts_coords      
    ) -> None:
        
        self.elem_2_verts = elem_2_verts
        self.vert_coords = verts_coords

    @staticmethod
    def create_rectangular_mesh(
        points: list[np.ndarray[np.float64]]
    ) -> Self:
        
        n_cells = [pts.size - 1 for pts in points]

        elem_2_verts = create_2D_tensor_prod_mesh_conn(n_cells)
        verts_coords = create_Cartesian_mesh_nodes(points)

        return ParametricMesh(elem_2_verts, verts_coords)
        


        
