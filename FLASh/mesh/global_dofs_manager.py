import numpy as np
import numpy.typing as npt
import scipy.sparse

from typing import Self, Callable

from mpi4py import MPI
import sys 

import dolfinx.fem
import pyvista as pv

from FLASh.mesh.subdomain import Subdomain
from FLASh.mesh.geometry import SplineGeometry

from itertools import chain

import qugar
from qugar.mesh import create_unfitted_impl_Cartesian_mesh

from tqdm import tqdm

import line_profiler
from ufl import sym, grad, Identity, tr, sqrt, inner
import dolfinx.fem.petsc


type SparseMatrix = scipy.sparse._csr.csr_matrix
type Marker = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.bool]]

class GlobalDofsManager:
    """Class for managing operations related to the global primal and dual
    degrees-of-freedom of the coarse domain.
    """

    def __init__(self, geometry: SplineGeometry, subdomains: list[Subdomain], linear_pde, communicators):
        """Initializes the class.

        Args:
            coarse_mesh (dolfinx.mesh.Mesh): Coarse mesh describing the
                subdomains partition.
            subdomain (SubDomain): Reference subdomain. All the subdomains are
                considered to be equal to this one, but a different positions.
        """
        self.geometry = geometry
        self.subdomains = subdomains
        self.coarse_mesh = geometry.coarse_mesh
        self.communicators = communicators
        self.linear_pde = linear_pde

        self._create_local_map()
        self._create_boundary_dofs()
        self._create_primal_dofs()
        self._create_dirichlet_boundary_conditions()
        self._create_boundary_scaling()

    @staticmethod
    def create_rectangle(
        geometry: SplineGeometry,
        linear_pde,
        communicators,
        opts: dict = None
    ) -> Self:

        """Creates a new GlobalDofsManager in a unit square.

        Args:
            n (list[int]): Number of elements per direction of the reference
                subdomain.
            degree (int): Discretization degree of the reference subdomain.
            N (list[int]): Number of subdomains per direction in the domain.

        Returns:
            Self: Newly generated GlobalDofsManager.
        """
        opts = opts or {}
        subdomain_opts = opts.get("subdomain_opts", None)

        n = [1,1]
        degree = geometry.basis_degree
        dim = 2
        levelset = geometry.levelset

        subdomains = []

        size = communicators.global_comm.Get_size()
        rank = communicators.global_comm.Get_rank()

        N = geometry.coarse_mesh._N

        counts = [N // size + (1 if i < N % size else 0) for i in range(size)]
        starts = np.cumsum([0] + counts[:-1])
        process_subdomains = np.arange(starts[rank], starts[rank] + counts[rank])

        if rank == 0:
            print("\nSubdomains assembly.")

        for s_ind, s_id in enumerate(process_subdomains):

            pts = geometry.coarse_mesh.get_cell_vertex_points(s_id)

            param = geometry.coarse_mesh.get_cell_parameters(s_id)

            bezier_element = geometry.get_bezier_element(s_id)

            subdomains.append(
                Subdomain(
                    n, 
                    degree, 
                    dim,
                    pts[0], 
                    pts[3], 
                    param,
                    levelset,
                    linear_pde,
                    bezier_element,
                    opts = subdomain_opts
                )
            )

            ### Assembly progress bar (bug if the number of subdomains is not the samme for al process) 
            # locking communication!!! ###

            done_local = np.array(s_ind + 1, dtype="i")
            done_global = np.array(0, dtype="i")
            communicators.global_comm.Reduce(done_local, done_global, op=MPI.SUM, root=0)

            if rank == 0:
                progress = done_global / N
                sys.stdout.write(f"\033[F\033[KSubdomains assembly: {progress:.1%}\n")
                sys.stdout.flush()


        return GlobalDofsManager(geometry, subdomains, linear_pde, communicators)

    @staticmethod
    def plot(
        geometry: SplineGeometry,
        communicators
    ) -> None:
        
        if communicators.global_comm.Get_rank() == 0:
        
            n = [1, 1]
            levelset = geometry.levelset

            N = geometry.coarse_mesh._N

            pl = pv.Plotter(shape=(1, 1))

            for s_id in range(N):

                p0 = np.array([0.0, 0.0])
                p1 = np.array([1.0, 1.0])

                comm = MPI.COMM_SELF

                param = geometry.coarse_mesh.get_cell_parameters(s_id)

                impl_func = levelset(param, p0, p1)

                unf_mesh = create_unfitted_impl_Cartesian_mesh(
                    comm, impl_func, n, p0, p1, exclude_empty_cells=False
                )

                reparam_degree = 3
                reparam = qugar.reparam.create_reparam_mesh(unf_mesh, degree=reparam_degree, levelset=False)

                reparam_pv = qugar.plot.reparam_mesh_to_PyVista(reparam)

                pv_mesh = reparam_pv.get("reparam")
                pv_wirebasket = reparam_pv.get("wirebasket")

                bezier_element = geometry.get_bezier_element(s_id)

                pv_mesh.points = bezier_element.evaluate(pv_mesh.points)
                pv_wirebasket.points = bezier_element.evaluate(pv_wirebasket.points)

                pl.add_mesh(pv_mesh, color="white", show_edges=False)
                pl.add_mesh(pv_wirebasket, color="blue", line_width=2)

            pl.view_xy()
            pl.show_axes()
            pl.show()

    def _create_local_map(self) -> None:

        size = self.communicators.global_comm.Get_size()
        rank = self.communicators.global_comm.Get_rank()

        N = self.get_num_subdomains()

        counts = [N // size + (1 if i < N % size else 0) for i in range(size)]
        starts = np.cumsum([0] + counts[:-1])
        process_subdomains = np.arange(starts[rank], starts[rank] + counts[rank])

        local_map = np.zeros((N+1,), dtype = np.int32)

        for s_ind, s_id in enumerate(process_subdomains):

            local_map[s_id] = s_ind

        self.process_subdomains = process_subdomains
        self.local_map = local_map

    def _create_boundary_dofs(self) -> None:

        N = self.get_num_subdomains()
        num_vertices_local = np.zeros(shape = (N, 4), dtype=np.int32)

        for s_ind, s_id in enumerate(self.process_subdomains):

            vertices_dofs = self.subdomains[s_ind].vertices_dofs
            for i, dofs in enumerate(vertices_dofs):
                num_vertices_local[s_id, i] = len(dofs)


        num_vertices = np.zeros(shape = num_vertices_local.shape, dtype=np.int32)
        self.communicators.global_comm.Allreduce(num_vertices_local, num_vertices, MPI.SUM)

        num_edges_local = np.zeros(shape = (N, 4), dtype=np.int32)

        for s_ind, s_id in enumerate(self.process_subdomains):

            edges_dofs = self.subdomains[s_ind].interior_edges_dofs
            for i, dofs in enumerate(edges_dofs):
                num_edges_local[s_id, i] = len(dofs)


        num_edges = np.zeros(shape = num_edges_local.shape, dtype=np.int32)
        self.communicators.global_comm.Allreduce(num_edges_local, num_edges, MPI.SUM)

        v_2_c = self.coarse_mesh.vertex_cell_conn
        c_2_v = self.coarse_mesh.cell_vertex_conn

        n_vertices = len(v_2_c)

        self._vertices_dofs_ranges = np.empty((n_vertices, 2), dtype=np.int32)
        active_vertices = []

        counter = 0

        self._vertex_to_subdomain = np.zeros(n_vertices, dtype=np.int32) 

        for vertex_id in range(n_vertices):
            cells = v_2_c[vertex_id]
            
            local_ids = [np.where(c_2_v[cell] == vertex_id)[0][0] for cell in cells]
            values = [num_vertices[cell, local_id] for cell, local_id in zip(cells, local_ids)]
            
            max_val = max(values)
            max_index = values.index(max_val)
            self._vertex_to_subdomain[vertex_id] = cells[max_index]
            
            self._vertices_dofs_ranges[vertex_id, 0] = counter
            counter += max_val
            self._vertices_dofs_ranges[vertex_id, 1] = counter

            if max_val > 0:
                active_vertices.append(vertex_id)

        self._active_vertices = np.array(active_vertices)

        e_2_c = self.coarse_mesh.edge_cell_conn
        c_2_e = self.coarse_mesh.cell_edge_conn

        n_edges = len(e_2_c)

        self._edges_dofs_ranges = np.empty((n_edges, 2), dtype=np.int32)

        for edge_id in range(n_edges):
            cells = e_2_c[edge_id]
            local_edge_id  = np.where(c_2_e[cells[0]] == edge_id)[0]

            if len(cells) > 1:
                reference = num_edges[cells[0], local_edge_id]
                for cell in cells[1:]:
                    local_id = np.where(c_2_e[cell] == edge_id)[0]
                    if num_edges[cell, local_id] != reference:
                        raise ValueError(f"Inconsistent num_vertices for edge {edge_id} "
                                        f"between cells {cells[0]} and {cell}: "
                                        f"{reference} vs {num_edges[cell, local_id]}")
            
            self._edges_dofs_ranges[edge_id, 0] = counter
            counter += num_edges[cells[0], local_edge_id].item()
            self._edges_dofs_ranges[edge_id, 1] = counter

    def _create_primal_dofs(self) -> None:
        """Creates the global primal degrees-of-freedom of every subdomain
        and stores them in self._primal_dofs.
        """

        N = self.get_num_subdomains()

        num_vertices_primals_local = np.zeros(shape = (N, 4), dtype=np.int32)
        num_edge_primals_local = np.zeros(shape = (N, 4), dtype=np.int32)
        
        for s_ind, s_id in enumerate(self.process_subdomains):

            vertices_range, edges_range = self.subdomains[s_ind].get_primal_ranges()

            for i, limit in enumerate(vertices_range):
                num_vertices_primals_local[s_id, i] = limit[1]-limit[0]

            for i, limit in enumerate(edges_range):
                num_edge_primals_local[s_id, i ] = limit[1]-limit[0]

        num_vertices_primals = np.zeros(shape = num_vertices_primals_local.shape, dtype=np.int32)
        self.communicators.global_comm.Allreduce(num_vertices_primals_local, num_vertices_primals, MPI.SUM)

        num_edge_primals = np.zeros(shape = num_edge_primals_local.shape, dtype=np.int32)
        self.communicators.global_comm.Allreduce(num_edge_primals_local, num_edge_primals, MPI.SUM)


        v_2_c = self.coarse_mesh.vertex_cell_conn
        c_2_v = self.coarse_mesh.cell_vertex_conn

        n_vertices = len(v_2_c)

        self._vertices_primal_ranges = np.empty((n_vertices, 2), dtype=np.int32)

        counter = 0
        for vertex_id in range(n_vertices):
            cells = v_2_c[vertex_id]
            
            local_ids = [np.where(c_2_v[cell] == vertex_id)[0][0] for cell in cells]
            values = [num_vertices_primals[cell, local_id] for cell, local_id in zip(cells, local_ids)]
            
            max_val = max(values)
            
            self._vertices_primal_ranges[vertex_id, 0] = counter
            counter += max_val
            self._vertices_primal_ranges[vertex_id, 1] = counter

        e_2_c = self.coarse_mesh.edge_cell_conn
        c_2_e = self.coarse_mesh.cell_edge_conn

        n_edges = len(e_2_c)

        active_edges = []
        self._edge_primal_ranges = np.empty((n_edges, 2), dtype=np.int32)

        for edge_id in range(n_edges):
            cells = e_2_c[edge_id]
            local_edge_id  = np.where(c_2_e[cells[0]] == edge_id)[0]

            if len(cells) > 1:
                reference = num_edge_primals[cells[0], local_edge_id]
                for cell in cells[1:]:
                    local_id = np.where(c_2_e[cell] == edge_id)[0]
                    if num_edge_primals[cell, local_id] != reference:
                        raise ValueError(f"Inconsistent average for edge {edge_id} "
                                        f"between cells {cells[0]} and {cell}: "
                                        f"{reference} vs {num_edge_primals[cell, local_id]}")
            
            self._edge_primal_ranges[edge_id, 0] = counter
            counter += num_edge_primals[cells[0], local_edge_id].item()
            self._edge_primal_ranges[edge_id, 1] = counter     

            if num_edge_primals[cells[0], local_edge_id] > 0:
                active_edges.append(edge_id)

        self._active_edges = np.array(active_edges)
        self._total_primals = counter

    def _create_dirichlet_boundary_conditions(self) -> None:

        v_2_c = self.coarse_mesh.vertex_cell_conn
        c_2_v = self.coarse_mesh.cell_vertex_conn

        n_vertices = len(v_2_c)

        vertices_dirichlet_dofs = []
        vdd_tag = []
        vdd_subdomain = []
        vdd_local_index = []

        for (type_, fun, marker, tag) in self.linear_pde.exterior_bc:

            if type_ == 0:

                vertices = self.coarse_mesh.locate_vertices(marker)
                vertices_dirichlet_dofs.append(vertices)
                vdd_tag.append(np.full(vertices.shape, tag, dtype = np.int32))

                subdomains = [self._vertex_to_subdomain[v] for v in vertices]
                local_indices = [np.where(c_2_v[s] == v)[0][0] for v, s in zip(vertices, subdomains)]

                vdd_subdomain.extend(subdomains)
                vdd_local_index.extend(local_indices)

        vertices_dirichlet_dofs = np.hstack(vertices_dirichlet_dofs)
        vdd_tag = np.hstack(vdd_tag)
        vdd_subdomain = np.array(vdd_subdomain, dtype=np.int32)
        vdd_local_index = np.array(vdd_local_index, dtype=np.int32)

        # I think that it is no longer necessary to find unique ones because they are unique by construction

        vertices_dirichlet_dofs, idx = np.unique(vertices_dirichlet_dofs, return_index=True)
        vdd_tag = vdd_tag[idx]
        vdd_subdomain = vdd_subdomain[idx]
        vdd_local_index = vdd_local_index[idx]

        vertices_dirichlet_dofs, i_idx, a_idx = np.intersect1d(
            vertices_dirichlet_dofs, 
            self._active_vertices, 
            return_indices=True
        )

        vdd_tag = vdd_tag[i_idx]
        vdd_subdomain = vdd_subdomain[i_idx]
        vdd_local_index = vdd_local_index[i_idx]

        self._vertices_dirichlet_dofs = vertices_dirichlet_dofs
        self._vertices_dirichlet_dofs_tags = vdd_tag

        ####

        e_2_c = self.coarse_mesh.edge_cell_conn
        c_2_e = self.coarse_mesh.cell_edge_conn

        n_edges = len(e_2_c)

        edges_dirichlet_dofs = []
        edd_tag = []
        edd_subdomain = []
        edd_local_index = []

        for (type_, fun, marker, tag) in self.linear_pde.exterior_bc:

            if type_ == 0:

                edges = self.coarse_mesh.locate_edges(marker)
                edges_dirichlet_dofs.append(edges)
                edd_tag.append(np.full(edges.shape, tag, dtype = np.int32))

                subdomains = [e_2_c[e][0] for e in edges]
                local_indices = [np.where(c_2_e[s] == e)[0][0] for e, s in zip(edges, subdomains)]

                edd_subdomain.extend(subdomains)
                edd_local_index.extend(local_indices)

        edges_dirichlet_dofs = np.hstack(edges_dirichlet_dofs)
        edd_tag = np.hstack(edd_tag)
        edd_subdomain = np.array(edd_subdomain, dtype=np.int32)
        edd_local_index = np.array(edd_local_index, dtype=np.int32)

        edges_dirichlet_dofs, idx = np.unique(edges_dirichlet_dofs, return_index=True)
        edd_tag = edd_tag[idx]
        edd_subdomain = edd_subdomain[idx]
        edd_local_index = edd_local_index[idx]

        edges_dirichlet_dofs, i_idx, a_idx = np.intersect1d(
            edges_dirichlet_dofs, 
            self._active_edges, 
            return_indices=True
        )

        edd_tag = edd_tag[i_idx]
        edd_subdomain = edd_subdomain[i_idx]
        edd_local_index = edd_local_index[i_idx]

        self._edges_dirichlet_dofs = edges_dirichlet_dofs
        self._edges_dirichlet_dofs_tagss = edd_tag

        #### 

        vertices_dirichlet_values = []
        edges_dirichlet_values = []        

        vertices_ordering = []
        edges_ordering = []

        N = self.get_num_subdomains()

        for s_ind, s_id in enumerate(self.process_subdomains):

            subdomain = self.subdomains[s_ind]

            local_vertices = np.where(vdd_subdomain == s_id)[0]
            local_edges = np.where(edd_subdomain == s_id)[0]

            local_vertices_tag = vdd_tag[local_vertices]
            local_vertices_index = vdd_local_index[local_vertices]

            local_edges_tag = edd_tag[local_edges]
            local_edges_index = edd_local_index[local_edges]
            
            max_vertex_tag = np.max(local_vertices_tag) if local_vertices_tag.size > 0 else -1
            max_edge_tag = np.max(local_edges_tag) if local_edges_tag.size > 0 else -1
            max_tag = max(max_vertex_tag, max_edge_tag)

            for tag in range(max_tag + 1):

                (_, fun, _, _) = self.linear_pde.exterior_bc[tag]

                u = subdomain.get_projected_function(fun)
                v_dofs = subdomain.vertices_dofs
                e_dofs = subdomain.interior_edges_dofs

                vertices = np.where(local_vertices_tag == tag)[0]
                edges = np.where(local_edges_tag == tag)[0]

                for vertex in vertices:
                    local_vertex = local_vertices_index[vertex]
                    global_vertex = local_vertices[vertex]

                    vertices_dirichlet_values.append(u[v_dofs[local_vertex]])
                    vertices_ordering.append(global_vertex)

                for edge in edges:
                    local_edge = local_edges_index[edge]
                    global_edge = local_edges[edge]

                    edges_dirichlet_values.append(u[e_dofs[local_edge]])
                    edges_ordering.append(global_edge)

        vertices_dirichlet_values = self.communicators.global_comm.allgather(vertices_dirichlet_values)
        edges_dirichlet_values = self.communicators.global_comm.allgather(edges_dirichlet_values)

        vertices_ordering = self.communicators.global_comm.allgather(vertices_ordering)
        edges_ordering = self.communicators.global_comm.allgather(edges_ordering)

        unordered_vertices_dirichlet_values = list(chain.from_iterable(vertices_dirichlet_values))      
        unordered_edges_dirichlet_values = list(chain.from_iterable(edges_dirichlet_values)) 

        vertices_ordering = list(chain.from_iterable(vertices_ordering))
        edges_ordering = list(chain.from_iterable(edges_ordering))

        vertices_dirichlet_values = [None] * len(vertices_ordering)
        edges_dirichlet_values = [None] * len(edges_ordering)

        for i, idx in enumerate(vertices_ordering):
            vertices_dirichlet_values[idx] = unordered_vertices_dirichlet_values[i]

        for i, idx in enumerate(edges_ordering):
            edges_dirichlet_values[idx] = unordered_edges_dirichlet_values[i]

        vertices_dirichlet_values = np.hstack(vertices_dirichlet_values) if vertices_dirichlet_values else np.array([])
        edges_dirichlet_values = np.hstack(edges_dirichlet_values)if edges_dirichlet_values else np.array([])

        ##### 

        primal_vertices_dirichlet_dofs = []
        boundary_vertices_dirichlet_dofs = []

        for vertex in vertices_dirichlet_dofs:

            start = self._vertices_primal_ranges[vertex, 0]
            end = self._vertices_primal_ranges[vertex, 1]
            primal_vertices_dirichlet_dofs.append(np.arange(start, end))

            start = self._vertices_dofs_ranges[vertex, 0]
            end = self._vertices_dofs_ranges[vertex, 1]
            boundary_vertices_dirichlet_dofs.append(np.arange(start, end))

        primal_edges_dirichlet_dofs = []
        boundary_edges_dirichlet_dofs = []

        for edge in edges_dirichlet_dofs:

            start = self._edge_primal_ranges[edge, 0]
            end = self._edge_primal_ranges[edge, 1]
            primal_edges_dirichlet_dofs.append(np.arange(start, end))

            start = self._edges_dofs_ranges[edge, 0]
            end = self._edges_dofs_ranges[edge, 1]
            boundary_edges_dirichlet_dofs.append(np.arange(start, end))


        primal_vertices_dirichlet_dofs = np.hstack(primal_vertices_dirichlet_dofs) if primal_vertices_dirichlet_dofs else np.array([])
        boundary_vertices_dirichlet_dofs = np.hstack(boundary_vertices_dirichlet_dofs) if boundary_vertices_dirichlet_dofs else np.array([])
        primal_edges_dirichlet_dofs = np.hstack(primal_edges_dirichlet_dofs) if primal_edges_dirichlet_dofs else np.array([])
        boundary_edges_dirichlet_dofs = np.hstack(boundary_edges_dirichlet_dofs) if boundary_edges_dirichlet_dofs else np.array([])

        #####

        self._primal_dirichlet_dofs = np.hstack([
            primal_vertices_dirichlet_dofs,
            primal_edges_dirichlet_dofs
        ]).astype(np.int32)

        self._boundary_dirichlet_dofs = np.hstack([
            boundary_vertices_dirichlet_dofs,
            boundary_edges_dirichlet_dofs
        ]).astype(np.int32)

        self._boundary_dirichlet_values = np.hstack([
            vertices_dirichlet_values,
            edges_dirichlet_values,
        ])

    def _create_boundary_scaling(self) -> None:

        c_2_v = self.coarse_mesh.cell_vertex_conn
        c_2_e = self.coarse_mesh.cell_edge_conn

        local_weights = np.zeros(self.get_num_boundary_dofs())

        for s_ind, s_id in enumerate(self.process_subdomains):

            vertices = c_2_v[s_id]
            edges = c_2_e[s_id]

            subdomain = self.subdomains[s_ind]
            local_vertices = subdomain.vertices_dofs
            local_edges = subdomain.interior_edges_dofs

            diagonal = subdomain.K.diagonal()

            for vertex, local_vertex in zip(vertices, local_vertices):
                if local_vertex.size > 0:
                    start, end = self._vertices_dofs_ranges[vertex]
                    local_weights[start:end] += diagonal[local_vertex]

            for edge, local_edge in zip(edges, local_edges):
                if local_edge.size > 0:
                    start, end = self._edges_dofs_ranges[edge]
                    local_weights[start:end] += diagonal[local_edge]

        self._weights = np.zeros(shape = local_weights.shape, dtype=np.float64)
        self.communicators.global_comm.Allreduce(local_weights, self._weights, MPI.SUM)

    def get_dirichlet_boundary_values(self) -> npt.NDArray[np.float64]:
        return self._boundary_dirichlet_values
    
    def get_active_primal_dofs(self) -> npt.NDArray[np.int32]:
        """Gets the ids of the global primal dofs that are not under Dirichlet
        boundary conditions.

        Returns:
            npt.NDArray[np.int32]: Global primal dofs on which Dirichlet
                boundary conditions are not applied. This array is unique and
                sorted.
        """

        all_primal_dofs = np.arange(self.get_num_primals())
        return np.setdiff1d(all_primal_dofs, self._primal_dirichlet_dofs)
    
    def get_dirichlet_boundary_dofs(self) -> npt.NDArray[np.int32]:
        return self._boundary_dirichlet_dofs
    
    def get_active_boundary_dofs(self) -> npt.NDArray[np.int32]:

        all_boundary_dofs = np.arange(self.get_num_boundary_dofs())
        return np.setdiff1d(all_boundary_dofs, self._boundary_dirichlet_dofs)
    
    def create_R(self, subdomain_id: int) -> SparseMatrix:

        assert 0 <= subdomain_id < self.get_num_subdomains()

        subdomain = self.subdomains[self.local_map[subdomain_id]]

        c_2_v = self.coarse_mesh.cell_vertex_conn
        c_2_e = self.coarse_mesh.cell_edge_conn

        vertices = c_2_v[subdomain_id]
        edges = c_2_e[subdomain_id]

        local_vertices, local_edges = subdomain.get_boundary_ranges()

        cols = []

        for vertex, local_vertex in zip(vertices, local_vertices):
            if local_vertex.size > 0:
                start, end = self._vertices_dofs_ranges[vertex]
                cols.append(np.arange(start, end))

        for edge, local_edge in zip(edges, local_edges):
            start, end = self._edges_dofs_ranges[edge]
            cols.append(np.arange(start, end))

        cols = np.hstack(cols)

        return cols
    
    def create_Rc(self, subdomain_id: int) -> SparseMatrix:

        assert 0 <= subdomain_id < self.get_num_subdomains()

        subdomain = self.subdomains[self.local_map[subdomain_id]]

        c_2_v = self.coarse_mesh.cell_vertex_conn
        c_2_e = self.coarse_mesh.cell_edge_conn

        vertices = c_2_v[subdomain_id]
        edges = c_2_e[subdomain_id]

        local_vertices_range, local_edges_range = subdomain.get_primal_ranges()

        cols = []

        for vertex, local_range in zip(vertices, local_vertices_range):
            if local_range[1] > local_range[0]:
                start, end = self._vertices_primal_ranges[vertex]
                cols.append(np.arange(start, end))

        for edge, local_range in zip(edges, local_edges_range):
            start, end = self._edge_primal_ranges[edge]
            cols.append(np.arange(start, end))

        cols = np.hstack(cols)

        return cols
    
    def create_D(self, subdomain_id: int) -> SparseMatrix:

        assert 0 <= subdomain_id < self.get_num_subdomains()

        subdomain = self.subdomains[self.local_map[subdomain_id]]

        c_2_v = self.coarse_mesh.cell_vertex_conn
        c_2_e = self.coarse_mesh.cell_edge_conn

        vertices = c_2_v[subdomain_id]
        edges = c_2_e[subdomain_id]

        local_vertices = subdomain.vertices_dofs
        local_edges = subdomain.interior_edges_dofs

        values = []

        for vertex, local_vertex in zip(vertices, local_vertices):
            if local_vertex.size > 0:
                start, end = self._vertices_dofs_ranges[vertex]
                values.append(subdomain.K.diagonal()[local_vertex]/self._weights[start:end])

        for edge, local_range in zip(edges, local_edges):
            start, end = self._edges_dofs_ranges[edge]
            values.append(subdomain.K.diagonal()[local_range]/self._weights[start:end])

        values = np.hstack(values)
        n = values.size
        rows = np.arange(n)

        D = scipy.sparse.csr_matrix((values, (rows, rows)), shape=(n, n))

        return D

    def get_num_subdomains(self) -> int:
        """Gets the number of subdomains.

        Returns:
            int: Number of subdomains.
        """
        return len(self.coarse_mesh.cell_vertex_conn)

    def get_num_primals(self) -> int:
        """Gets the number of global primal degrees-of-freedom.

        Returns:
            int: Number of primal degrees-of-freedom.
        """
        return self._total_primals
    
    def get_num_boundary_dofs(self) -> int:

        return self._edges_dofs_ranges[-1, 1]
    
    def get_boundary_weights(self) -> npt.NDArray[np.float64]:
        return self._weights
    
    def transform_to_fenicsx(self, us) -> list[npt.NDArray[np.float64]]:

        l_us = []

        for u, subdomain in zip(us, self.subdomains):
            l_us.append(subdomain.get_fenicsx_function(u))
        
        return l_us
    
    def get_fs(self) -> list[npt.NDArray[np.float64]]:

        fs = []

        for subdomain in self.subdomains:
            fs.append(subdomain.f)
        
        return fs

    def compute_error(self, us, us_ex) -> float: 

        error_local = 0
        norm_local = 0

        for u, u_ex, subdomain in zip(us, us_ex, self.subdomains):

            M = subdomain.M
            error_local += (u_ex - u).T @ M @ (u_ex - u)
            norm_local += u_ex.T @ M @ u_ex

        error = np.sqrt(self.communicators.global_comm.allreduce(error_local, op=MPI.SUM))
        norm = np.sqrt(self.communicators.global_comm.allreduce(norm_local, op=MPI.SUM))
        
        return error/norm

    def plot_solution(self, us) -> None:

        s_inds = self.process_subdomains
        us = self.transform_to_fenicsx(us)
        us = self.communicators.global_comm.gather(us, root = 0)
        s_inds = self.communicators.global_comm.gather(s_inds, root = 0)

        if self.communicators.global_comm.Get_rank() == 0:

            us = list(chain.from_iterable(us))      
            s_inds = list(chain.from_iterable(s_inds)) 

            x = self.coarse_mesh.vertex_coordinates
            c_2_v = self.coarse_mesh.cell_vertex_conn

            n = [1,1]
            degree = self.geometry.basis_degree
            dim = 2
            levelset = self.geometry.levelset

            pl = pv.Plotter(shape=(1, 1))

            for s_id, u in zip(s_inds, us):

                vertices = c_2_v[s_id]

                bezier_element = self.geometry.get_bezier_element(s_id)

                comm = MPI.COMM_SELF

                impl_func = levelset(list(self.coarse_mesh.get_cell_parameters(s_id)), np.array([0.0,0.0]), np.array([1.0,1.0]))

                unf_mesh = create_unfitted_impl_Cartesian_mesh(
                    comm, impl_func, n, np.array([0.0,0.0]), np.array([1.0,1.0]), exclude_empty_cells=False
                )

                V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree, (dim,)))
                uh = dolfinx.fem.Function(V)
                uh.x.array[:] = u

                reparam_degree = 3
                reparam = qugar.reparam.create_reparam_mesh(unf_mesh, degree=reparam_degree, levelset=False)
                reparam_mesh = reparam.create_mesh()

                V_reparam = dolfinx.fem.functionspace(reparam_mesh, ("CG", reparam_degree, (dim,)))
                uh_reparam = dolfinx.fem.Function(V_reparam)

                cmap = reparam_mesh.topology.index_map(reparam_mesh.topology.dim)
                num_cells = cmap.size_local + cmap.num_ghosts
                cells = np.arange(num_cells, dtype=np.int32)

                interpolation_data = dolfinx.fem.create_interpolation_data(V_reparam, V, cells, padding=1.0e-14)
                uh_reparam.interpolate_nonmatching(uh, cells, interpolation_data=interpolation_data)

                reparam_pv = qugar.plot.reparam_mesh_to_PyVista(reparam)
                pv_mesh = reparam_pv.get("reparam")
                pv_mesh.points = bezier_element.evaluate(pv_mesh.points)

                u_plot = uh_reparam.x.array.reshape(-1, dim)
                pv_mesh.point_data["uh"] = np.hstack((u_plot, np.zeros((u_plot.shape[0], 1))))
                pv_mesh_warped = pv_mesh.warp_by_vector("uh", factor=1)

                pl.add_mesh(pv_mesh_warped, show_edges=False)


            pl.view_xy()
            pl.show_axes()
            pl.show()

    def plot_stress(self, us) -> None:

        s_inds = self.process_subdomains
        us = self.transform_to_fenicsx(us)
        us = self.communicators.global_comm.gather(us, root = 0)
        s_inds = self.communicators.global_comm.gather(s_inds, root = 0)

        if self.communicators.global_comm.Get_rank() == 0:

            us = list(chain.from_iterable(us))      
            s_inds = list(chain.from_iterable(s_inds)) 

            x = self.coarse_mesh.vertex_coordinates
            c_2_v = self.coarse_mesh.cell_vertex_conn

            n = [1,1]
            degree = self.geometry.basis_degree
            dim = 2
            levelset = self.geometry.levelset

            E = self.linear_pde.E
            nu = self.linear_pde.nu

            mu = E / (2.0 * (1.0 + nu))
            lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

            def sigma(v):
                return lmbda * tr(sym(grad(v))) * Identity(len(v)) + 2.0 * mu * sym(grad(v))

            def von_mises(v):
                s = sigma(v) - (1./3) * tr(sigma(v)) * Identity(len(v))
                return sqrt(3./2 * inner(s, s))

            pl = pv.Plotter(shape=(1, 1))

            for s_id, u in zip(s_inds, us):

                vertices = c_2_v[s_id]

                bezier_element = self.geometry.get_bezier_element(s_id)

                comm = MPI.COMM_SELF

                impl_func = levelset(list(self.coarse_mesh.get_cell_parameters(s_id)), np.array([0.0,0.0]), np.array([1.0,1.0]))

                unf_mesh = create_unfitted_impl_Cartesian_mesh(
                    comm, impl_func, n, np.array([0.0,0.0]), np.array([1.0,1.0]), exclude_empty_cells=False
                )

                V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree, (dim,)))
                uh = dolfinx.fem.Function(V)
                uh.x.array[:] = u

                W = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree, (1, )))
                von_mises_expr = dolfinx.fem.Expression(von_mises(uh), W.element.interpolation_points())
                sh = dolfinx.fem.Function(W)
                sh.interpolate(von_mises_expr)

                reparam_degree = 3
                reparam = qugar.reparam.create_reparam_mesh(unf_mesh, degree=reparam_degree, levelset=False)
                reparam_mesh = reparam.create_mesh()

                V_reparam = dolfinx.fem.functionspace(reparam_mesh, ("CG", reparam_degree, (dim,)))
                uh_reparam = dolfinx.fem.Function(V_reparam)

                W_reparam = dolfinx.fem.functionspace(reparam_mesh, ("CG", reparam_degree, (1, )))
                sh_reparam = dolfinx.fem.Function(W_reparam)

                cmap = reparam_mesh.topology.index_map(reparam_mesh.topology.dim)
                num_cells = cmap.size_local + cmap.num_ghosts
                cells = np.arange(num_cells, dtype=np.int32)

                interpolation_data_u = dolfinx.fem.create_interpolation_data(V_reparam, V, cells, padding=1.0e-14)
                uh_reparam.interpolate_nonmatching(uh, cells, interpolation_data=interpolation_data_u)

                interpolation_data_s = dolfinx.fem.create_interpolation_data(W_reparam, W, cells, padding=1.0e-14)
                sh_reparam.interpolate_nonmatching(sh, cells, interpolation_data=interpolation_data_s)

                # PyVista conversion
                reparam_pv = qugar.plot.reparam_mesh_to_PyVista(reparam)
                pv_mesh = reparam_pv.get("reparam")
                pv_mesh.points = bezier_element.evaluate(pv_mesh.points)

                # Add both data sets to the same mesh object
                u_plot = uh_reparam.x.array.reshape(-1, dim)
                pv_mesh.point_data["displ_mag"] = np.linalg.norm(u_plot, axis=1)
                pv_mesh.point_data["uh_vec"] = np.hstack((u_plot, np.zeros((u_plot.shape[0], 1))))
                pv_mesh.point_data["von_mises"] = sh_reparam.x.array

                pl.add_mesh(pv_mesh, scalars="von_mises", show_edges=False)

            pl.view_xy()
            pl.show_axes()
            pl.show()