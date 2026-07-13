import numpy as np

import mpi4py.MPI as MPI

from typing import Self, Callable

from FLASh.mesh import (
    BezierElement
)

import numpy as np

def create_inverse_conn(conn):

    conn = np.asarray(conn)

    n = int(conn.max()) + 1

    inv_conn = [[] for _ in range(n)]

    for e, vs in enumerate(conn):
        for v in vs:
            inv_conn[v].append(e)

    inv_conn_arrays = [np.array(es) for es in inv_conn]

    return inv_conn_arrays


class WrenchCoarseMesh:

    def __init__(
        self,
        cell_vertex_connectivity,
        cell_edge_connectivity,
        edge_vertex_connectivity,
        vertex_coordinates
    ) -> None:

        self.cell_vertex_conn = list(np.array(cell_vertex_connectivity))
        self.cell_edge_conn = list(np.array(cell_edge_connectivity))
        self.edge_vertex_conn = list(np.array(edge_vertex_connectivity))
        self.vertex_coordinates = vertex_coordinates

        self._N = len(self.cell_vertex_conn)
        self._n = vertex_coordinates.shape[0]

        self._create_inverse_connectivities()
        self._create_edge_coordinates()

        parameters = np.zeros((self._n))
        self.set_parameter_field(parameters)
    
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

    def set_custom_parameter_field(self, fun: Callable, nodes_id: np.ndarray, values: np.ndarray) -> None:

        comm = MPI.COMM_WORLD

        if comm.Get_rank() == 0:
            parameter_array = fun(self.vertex_coordinates.T)
            parameter_array[nodes_id] = values
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

        vertices = self.cell_vertex_conn[cell_id]

        return self.vertex_coordinates[vertices]

    def get_cell_parameters(self, cell_id: int) -> np.ndarray:

        return self._parameters[cell_id]

    def locate_vertices(self, marker) -> np.ndarray:

        return np.where(marker(self.vertex_coordinates.T))[0]
    
    def locate_edges(self, marker) -> np.ndarray:

        return np.where(marker(self.edge_coordinates.T))[0]

class WrenchGeometry:

    def __init__(
        self,
        cell_vertex_connectivity,
        cell_edge_connectivity,
        edge_vertex_connectivity,
        vertex_coordinates,
        map_coefficients,
        levelset,
        opts: dict = None
    ) -> None:
        
        opts = opts or {}
    
        self.degree = opts.get("spline_degree", 2)
        self.basis_degree = opts.get("basis_degree", 8)
        self.map_coefficients = map_coefficients
        self.levelset = levelset

        self.coarse_mesh = WrenchCoarseMesh(
            cell_vertex_connectivity,
            cell_edge_connectivity,
            edge_vertex_connectivity,
            vertex_coordinates
        )

    def get_bezier_element(self, cell_id) -> BezierElement:

        control_points = self.map_coefficients[cell_id].reshape(self.degree+1, self.degree+1, -1)
        control_points = control_points.transpose(1, 0, 2)
        control_points = np.pad(control_points, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)

        return BezierElement(self.degree, control_points)
    