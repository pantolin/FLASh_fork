import numpy as np
import numpy.typing as npt
import scipy.sparse
import matplotlib.pyplot as plt

from typing import Callable, TypeAlias
from functools import lru_cache

from FLASh.mesh.gauss_lobatto import Lagrange2D
from FLASh.mesh.geometry import SomeName

from mpi4py import MPI
from pathlib import Path
from functools import cached_property

import pyvista as pv
import ufl
import dolfinx
import dolfinx.fem

import qugar
from qugar import impl
from qugar.mesh import create_unfitted_impl_Cartesian_mesh
from qugar.dolfinx import form_custom
from qugar.mesh.unfitted_cart_mesh import UnfittedCartMesh

import line_profiler

UnfittedDomain: TypeAlias = qugar.cpp.UnfittedImplDomain_2D | qugar.cpp.UnfittedImplDomain_3D
type SparseMatrix = scipy.sparse._csr.csr_matrix
dtype = np.float64

def c_0(x, y):
    return (1+0*x[0])
def c_0_x(x,y):
    return (1+0*x[0],0*x[1])
def c_0_y(x,y):
    return (0*x[0],1+0*x[1])
def c_1(x,y):
    return (x[1]-y[1], -x[0]+y[0])
        
def zero_function(x):
    return (0*x[0],0*x[0])

class Subdomain:
    _cached_C = None
    
    def __init__(
        self,
        n: list[int],
        degree: int,
        dim: int,
        p0: list[float],
        p1: list[float],
        parameters: list[int],
        levelset: Callable,
        linear_pde,
        map,
        opts: dict = None
    ):
        """Initializes the subdomain.

        Args:
            n (list[int]): Number of elements per direction in the subdomain.
            degree (int): Discretization space degree.
            p0 (list[float], optional): Bottom-left corner of the subdomain.
                Defaults to [0.0, 0.0].
            p1 (list[float], optional): Top-right corner of the subdomain.
                Defaults to [1.0, 1.0].
            assemble (bool, optional): If True, the stiffness matrix and
                right-hand-side vector are assembled. Defaults to False.
        """
        self._n = n
        self._degree = degree
        self._dim = dim
        self._p0 = p0
        self._p1 = p1
        self._parameters = parameters

        self._levelset = levelset
        self._linear_pde = linear_pde
        self._map = map
        self._basis = Lagrange2D(degree, np.zeros(2), np.ones(2))

        self._total_size = self._basis.get_total_number_basis() * dim

        opts = opts or {}

        self._num_primals_per_edge = opts.get("primals_per_edge", 3)
        self._vertex_primals = opts.get("vertex_primals", True)
        self._edge_primals = opts.get("edge_primals", True)
        self._stab = opts.get("stabilize", False)
        self._stab_val = opts.get("stabilization", 1e-3)
        self._approx = opts.get("approximate_geometry", False)
        self._approx_degree = opts.get("approximate_geometry_degree", 2)
        self._parametric_bc = opts.get("parametric_bc", False)
        
        self._create_lagrange_extraction()
        self._set_coordinates()
        self._set_somename()
        self.create_edge_centers()
        self.create_primal_dofs()
        self.create_boundary_dofs()
        
        if opts.get("assemble", False) == True:
            self.assemble()

    @cached_property
    def all_dofs(self) -> npt.NDArray[np.int32]:
        """Gets an array with all the degrees-of-freedom in the subdomain's
        space.

        Returns:
            npt.NDArray[np.int32]: Sorted array with all the
                degrees-of-freedom.
        """

        number_of_basis = self._basis.get_total_number_basis()
        return np.arange(0, self._dim * number_of_basis)

    @cached_property
    def vertices_dofs(self) -> list[npt.NDArray[np.int32]]:
        """Gets the degrees-of-freedom associated to the four courners of the
        subdomain.

        The ordering of the four corners follows the basix convention for
        quadrilaterals.
        See https://docs.fenicsproject.org/basix/v0.8.0/index.html

        Returns:
            npt.NDArray[np.int32]: Sorted array with the corner
            degrees-of-freedom.
        """
        basis_vertices_dofs = self._basis.get_vertices_dofs()
        dim = self._dim
        vertices_dofs = []

        for dofs in basis_vertices_dofs:
            dofs = tuple(dim * dofs + i for i in range(dim))
            vertices_dofs.append(np.column_stack(dofs).flatten())

        return vertices_dofs

    @cached_property
    def edges_dofs(self) -> list[npt.NDArray[np.int32]]:
        """Gets the degrees-of-freedom for the 4 faces of the subdomain.

        The ordering of the four faces follows the basix convention for
        quadrilaterals.
        See https://docs.fenicsproject.org/basix/v0.8.0/index.html

        Within each face the nodes are ordered in growing X[0] (X[1])
        coordinates for horizontal (vertical) faces

        Returns:
            list[npt.NDArray[np.int32]]: Sorted arrays of degrees-of-freedom.
                One for every face.
        """

        basis_edges_dofs = self._basis.get_edges_dofs()
        dim = self._dim
        edges_dofs = []

        for dofs in basis_edges_dofs:
            dofs = tuple(dim * dofs + i for i in range(dim))
            edges_dofs.append(np.column_stack(dofs).flatten())

        return edges_dofs
    
    @cached_property
    def interior_edges_dofs(self) -> list[npt.NDArray[np.int32]]:

        vertices_dofs = np.unique(np.hstack(self.vertices_dofs))
        edges_dofs = self.edges_dofs
        int_edges_dofs = []

        for dofs in edges_dofs:
            int_edges_dofs.append(np.setdiff1d(dofs, vertices_dofs))

        return int_edges_dofs
    
    @cached_property
    def boundary_dofs(self) -> npt.NDArray[np.int32]:
        """Gets the degrees-of-freedom associated to the fitted boundaries 
        of the subdomain.

        Returns:
            npt.NDArray[np.int32]: Sorted array with all the boundary
            degrees-of-freedom.
        """
        vertices_dofs = np.hstack(self.vertices_dofs)
        edges_dofs = np.hstack(self.interior_edges_dofs)

        boundary_dofs = np.hstack([
            vertices_dofs,
            edges_dofs
        ])

        return boundary_dofs

    @cached_property
    def interior_dofs(self) -> npt.NDArray[np.int32]:
        boundary_dofs = self.boundary_dofs
        all_dofs = self.all_dofs

        mask = ~np.isin(all_dofs, boundary_dofs)  
        interior_dofs = all_dofs[mask] 

        return interior_dofs
    

    def get_primal_ranges(self) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:

        return self._vertices_primals, self._edges_primals
    
    def get_num_local_primals(self) -> int:
        return self._edges_primals[-1,-1]
    
    def get_num_local_vertex_primals(self) -> int:
        return self._vertices_primals[-1,-1]
    
    def get_boundary_ranges(self) -> tuple[list[npt.NDArray[np.int32]], list[npt.NDArray[np.int32]]]:

        return self._vertices_boundary_ordering, self._edges_boundary_ordering
    
    def get_num_boundary_dofs(self):

        return self._num_boundary_dofs

    def create_edge_centers(self) -> None:

        ## This needs to be changed for cells which are not cartesian rectangles in the 
        # parametric mesh, but is only used for neumann bc.

        v0 = self._p0
        v1 = np.array([self._p1[0], self._p0[1]])
        v2 = np.array([self._p0[0], self._p1[1]])
        v3 = self._p1

        self._parametric_edge_centers = [
            0.5*(v0+v1),
            0.5*(v0+v2),
            0.5*(v1+v3),
            0.5*(v2+v3)
        ]

        self._parametric_cell_center = 0.25*(v0+v1+v2+v3)
        
        centers = [
            np.array([0.5, 0.0]),
            np.array([0.0, 0.5]),
            np.array([1.0, 0.5]),
            np.array([0.5, 1.0])
        ]

        self._edge_centers = []

        for center in centers:

            x = self._map.evaluate(center[None,:])[0]
            self._edge_centers.append(x)

        ##### This is used for other things.

        self._e_2_v = [
            np.array([0, 1], dtype = np.int32),
            np.array([0, 2], dtype = np.int32),
            np.array([1, 3], dtype = np.int32),
            np.array([2, 3], dtype = np.int32)
        ]

    def create_primal_dofs(self) -> None:

        count = 0

        vertices_dofs = self.vertices_dofs
        self._vertices_primals = np.zeros(shape = (4, 2), dtype=np.int32)

        for i, dofs in enumerate(vertices_dofs):
            self._vertices_primals[i,0] = count
            count += self._dim if len(dofs)>0 and self._vertex_primals else 0
            self._vertices_primals[i,1] = count

        edges_dofs = self.interior_edges_dofs
        self._edges_primals = np.zeros(shape = (4, 2), dtype=np.int32)

        for i, dofs in enumerate(edges_dofs):
            self._edges_primals[i,0] = count
            count += self._num_primals_per_edge if len(dofs)>0 and self._edge_primals else 0
            self._edges_primals[i,1] = count

    def create_boundary_dofs(self) -> None:

        count = 0

        vertices_dofs = self.vertices_dofs
        self._vertices_boundary_ordering = []

        for dofs in vertices_dofs:
            start = count
            count += dofs.size
            self._vertices_boundary_ordering.append(np.arange(start, count))

        edge_dofs = self.interior_edges_dofs
        self._edges_boundary_ordering = []
        
        for dofs in edge_dofs:
            start = count
            count += dofs.size
            self._edges_boundary_ordering.append(np.arange(start, count))

        self._num_boundary_dofs = count 

    def _create_lagrange_extraction(self) -> None:

        if Subdomain._cached_C is not None:
            self.C = Subdomain._cached_C
            return

        unf_mesh = self.create_mesh()
        V = self._linear_pde.create_function_space(unf_mesh, self._degree)
        x = V.tabulate_dof_coordinates()

        basis_C = self._basis.get_lagrange_extraction(x)
        basis_C = basis_C.tocoo()

        mask = np.abs(basis_C.data) >= 1e-12

        dim = self._dim

        rows = basis_C.row[mask]
        cols = basis_C.col[mask]
        data = basis_C.data[mask]

        rows = np.concatenate([dim*rows+i for i in range(dim)])
        cols = np.concatenate([dim*cols+i for i in range(dim)])
        data = np.concatenate([data for i in range(dim)])

        shape = (dim*basis_C.shape[0], dim*basis_C.shape[1])
        C = scipy.sparse.coo_matrix((data, (rows, cols)), shape=shape).tocsr()

        Subdomain._cached_C = C  
        self.C = C
     
    def _set_coordinates(self) -> None:

        self._x = self._basis.get_nodes()
        
    def _set_somename(self, p = None) -> None:

        if p is None:
            p = self._approx_degree

        element = SomeName(p)
        element.assemble_mass()

        self.element = element

    def create_qugar_mesh(self) -> UnfittedDomain:

        n_cells = [1] * self._dim
        cell_breaks = [np.linspace(0.0, 1.0, n_cells[dir] + 1, dtype=dtype) for dir in range(self._dim)]

        xmin = np.array([0.0, 0.0])
        xmax = np.array([1.0, 1.0])

        impl_func = self._levelset(self._parameters, xmin, xmax).cpp_object

        grid = qugar.cpp.create_cart_grid(cell_breaks)
        unf_mesh = qugar.cpp.create_unfitted_impl_domain(impl_func, grid)

        return unf_mesh
    
    def create_qugar_negative_mesh(self) -> UnfittedDomain:

        n_cells = [1] * self._dim
        cell_breaks = [np.linspace(0.0, 1.0, n_cells[dir] + 1, dtype=dtype) for dir in range(self._dim)]

        xmin = np.array([0.0, 0.0])
        xmax = np.array([1.0, 1.0])

        impl_func = impl.create_negative(self._levelset(self._parameters, xmin, xmax)).cpp_object

        grid = qugar.cpp.create_cart_grid(cell_breaks)
        unf_mesh = qugar.cpp.create_unfitted_impl_domain(impl_func, grid)

        return unf_mesh


    def create_mesh(self) -> UnfittedCartMesh:

        n = self._n
        comm = MPI.COMM_SELF
        xmin = np.array([0.0, 0.0])
        xmax = np.array([1.0, 1.0])

        impl_func = self._levelset(self._parameters, xmin, xmax)

        unf_mesh = create_unfitted_impl_Cartesian_mesh(
            comm, impl_func, n, xmin, xmax, exclude_empty_cells=False
        )

        return unf_mesh
    
    def create_C(self) -> SparseMatrix:

        constraints = [c_0_x, c_0_y, c_1]        
        vertices_primals_ranges, edge_primals_ranges = self.get_primal_ranges()
        vertices_b_dofs, edges_b_dofs = self.get_boundary_ranges()
        edges_dofs = self.interior_edges_dofs
        num_primals = edge_primals_ranges[-1,-1]

        rows = []
        cols = []
        values = []

        for limit, dofs in zip(vertices_primals_ranges, vertices_b_dofs):
            if limit[1] > limit[0]:
                rows.append(np.arange(limit[0], limit[1]))
                cols.append(dofs)
                values.append(np.ones(limit[1]-limit[0]))

        for limit, b_dofs, dofs, center in zip(edge_primals_ranges, edges_b_dofs, edges_dofs, self._edge_centers):

            for i, f in zip(np.arange(limit[0],limit[1]), constraints):

                c = self.get_boundary_constraint(lambda x: f(x, center))
                
                rows.append(np.full(dofs.size, i))
                cols.append(b_dofs)
                values.append(c[dofs])

        rows = np.hstack(rows)
        cols = np.hstack(cols)
        values = np.hstack(values)

        C = scipy.sparse.csr_matrix((values, (rows, cols)), shape=(num_primals, self.boundary_dofs.size))

        return C
    

    def assemble(self) -> None:

        n_b = self._basis.get_total_number_basis()
        n_c = self.element.n

        if self._linear_pde.K_model:

            self.element.fit(
                lambda x: self._map.evaluate_A(
                    x, 
                    lambda_ = self._linear_pde.lambda_,
                    mu = self._linear_pde.mu,
                )
            )

            k_coefs = self.element.c.reshape((-1, 2, 2, 2, 2))

            K_core = self._linear_pde.K_model.evaluate(self._parameters.reshape(1,4))
            K_core = K_core.reshape(n_b, n_b, 2, 2, n_c)

            K = np.tensordot(K_core, k_coefs, axes=([2,3,4],[1,3,0]))
            K = K.transpose(0, 2, 1, 3)
            K = K.reshape(n_b * 2, n_b * 2)

            if self._stab:

                K_core_neg = self._linear_pde.K_full_core - K_core

                K_neg = np.tensordot(K_core_neg, k_coefs, axes=([2,3,4],[1,3,0]))
                K_neg = K_neg.transpose(0, 2, 1, 3)
                K_neg = K_neg.reshape(n_b * 2, n_b * 2)

                K += self._stab_val * K_neg
        else: 

            K = self.assemble_K()

        # The mass is only used to computing errors and it is not necesarry to solve the problem
        
        if self._linear_pde.M_model:

            self.element.fit(self._map.evaluate_jacobian_determinant)
            m_coefs = self.element.c.reshape((-1))

            M_core = self._linear_pde.M_model.evaluate(self._parameters.reshape(1,4))
            M_core = M_core.reshape(n_b, n_b, n_c)

            I_d = np.eye(2)

            M = np.tensordot(M_core, m_coefs, axes=([2],[0]))
            M = M[:, None, :, None] * I_d[None, :, None, :]
            M = M.reshape(n_b * 2, n_b * 2)

        else: 
            
            M = self.assemble_M()

        if self._linear_pde.bM_model:

            self.element.fit(self._map.evaluate_arclen)
            bm_coefs = self.element.c.reshape((-1, 2))

            bM_core = self._linear_pde.bM_model.evaluate(self._parameters.reshape(1,4))
            bM_core = bM_core.reshape(n_b, n_b, n_c, 2)

            I_d = np.eye(2)

            bM = np.tensordot(bM_core, bm_coefs, axes=([2, 3],[0, 1]))
            bM = bM[:, None, :, None] * I_d[None, :, None, :]
            bM = bM.reshape(n_b * 2, n_b * 2)

        else:

            bM = self.assemble_bM()

        b = (np.array(self._linear_pde.source(self._map.evaluate(self._x).T)).T).flatten() 
        f = M @ b

        edge_dofs = self.edges_dofs

        for (type_, fun, marker, ind) in self._linear_pde.exterior_bc:
            if type_ == 1:
                for dofs, center in zip(edge_dofs, self._parametric_edge_centers):
                    if marker(center):

                        b = np.zeros((n_b * 2))
                        if self._parametric_bc:
                            x = self._x.copy()
                            x[:,0] = self._p0[0] + (self._p1[0] - self._p0[0]) * x[:,0]
                            x[:,1] = self._p0[1] + (self._p1[1] - self._p0[0]) * x[:,0]
                            b[dofs] = (np.array(fun(x.T)).T).flatten()[dofs]
                        else:
                            b[dofs] = (np.array(fun(self._map.evaluate(self._x).T)).T).flatten()[dofs]
                        f += bM @ b

        self.K = K
        self.M = M
        self.bM = bM
        self.f = f


    def assemble_K(self, approx = None, stab = None) -> np.ndarray:
            
        if approx is None:
            approx = self._approx
        if stab is None:
            stab = self._stab
            
        unf_domain = self.create_qugar_mesh()

        if approx:

            self.element.fit(
                lambda x: self._map.evaluate_A(
                    x, 
                    lambda_ = self._linear_pde.lambda_,
                    mu = self._linear_pde.mu,
                )
            )
            coeff = self.element.evaluate
            
        else:

            coeff = lambda x: self._map.evaluate_A(
                    x, 
                    lambda_ = self._linear_pde.lambda_,
                    mu = self._linear_pde.mu,
                )
            
        K = self._linear_pde.assemble_stiffness(unf_domain, self._basis, coeff)

        if stab:

            K_neg = self._linear_pde.assemble_stiffness(unf_domain, self._basis, coeff, full_cell = True) - K
            K += self._stab_val * K_neg
        
        return K
    
    def assemble_M(self, approx = None) -> np.ndarray:

        if approx is None:
            approx = self._approx

        unf_domain = self.create_qugar_mesh()

        if approx:

            self.element.fit(self._map.evaluate_jacobian_determinant)
            coeff = self.element.evaluate
            
        else:

            coeff = self._map.evaluate_jacobian_determinant
            
        return self._linear_pde.assemble_mass(unf_domain, self._basis, coeff)
    
    def assemble_bM(self, approx = None) -> np.ndarray:

        if approx is None:
            approx = self._approx

        unf_domain = self.create_qugar_mesh()

        if approx:

            self.element.fit(self._map.evaluate_arclen)
            coeff = self.element.evaluate
            
        else:

            coeff = self._map.evaluate_arclen

        bM = self._linear_pde.assemble_boundary_mass(unf_domain, self._basis, coeff)
        
        return bM 

    def assemble_f(self, approx = None) -> np.ndarray:

        if approx is None:
            approx = self._approx

        unf_domain = self.create_qugar_mesh()

        if approx:

            self.element.fit(self._map.evaluate_jacobian_determinant)
            coeff = self.element.evaluate
            
        else:

            coeff = self._map.evaluate_jacobian_determinant
            
        return self._linear_pde.assemble_right_hand_side(unf_domain, self._basis, coeff, self._parametric_edge_centers)




    def assemble_K_core(self, p = None) -> np.ndarray:

        if p is None:
            element = self.element
        else:
            element = SomeName(p)
        
        unf_domain = self.create_qugar_mesh()
        
        return self._linear_pde.assemble_stiffness_core(unf_domain, self._basis, element)
    
    def assemble_M_core(self, p = None) -> np.ndarray:

        if p is None:
            element = self.element
        else:
            element = SomeName(p)
        
        unf_domain = self.create_qugar_mesh()
        
        return self._linear_pde.assemble_mass_core(unf_domain, self._basis, element)
    
    def assemble_bM_core(self, p = None) -> np.ndarray:

        if p is None:
            element = self.element
        else:
            element = SomeName(p)
        
        unf_domain = self.create_qugar_mesh()
        
        return self._linear_pde.assemble_boundary_mass_core(unf_domain, self._basis, element)
    



    def get_projected_function(self, f) -> npt.NDArray[np.float64]:

        u = (np.array(f(self._map.evaluate(self._x).T)).T).flatten() 

        return u
    
    def get_boundary_constraint(self, f) -> npt.NDArray[np.float64]:

        b = (np.array(f(self._map.evaluate(self._x).T)).T).flatten() 
        y = self.bM @ b 

        return y 


    def plot(self, map = "True") -> None: 

        unf_mesh = self.create_mesh()

        reparam = qugar.reparam.create_reparam_mesh(unf_mesh, degree=3, levelset=False)
        reparam_pv = qugar.plot.reparam_mesh_to_PyVista(reparam)

        pv_mesh = reparam_pv.get("reparam")
        pv_wirebasket = reparam_pv.get("wirebasket")

        if map == "True":
            pv_mesh.points = self._map.evaluate(pv_mesh.points)
            pv_wirebasket.points = self._map.evaluate(pv_wirebasket.points)

        pl = pv.Plotter(shape=(1, 1))
        pl.add_mesh(pv_mesh, color="white")
        pl.add_mesh(pv_wirebasket, color="blue", line_width=2)
        pl.view_xy()
        pl.show()

    def plot_mesh_nodes(self, map = "True") -> None:

        unf_mesh = self.create_mesh()
        V, _ = self._linear_pde.create_function_space(unf_mesh, self._degree)
        x = V.tabulate_dof_coordinates()

        if map:
            x = self._map(x)

        plt.figure(figsize=(6, 5))
        plt.plot(x[:,0], x[:,1], 'o') 
        plt.title(f'Mesh nodes')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xticks(self._basis._nodes[0])
        plt.yticks(self._basis._nodes[1])
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_function(self, u) -> None:

        pl = pv.Plotter(shape=(1, 1))

        dim = self._dim
        n = self._n
        degree = self._degree

        xmin = np.array([0.0, 0.0])
        xmax = np.array([1.0, 1.0])

        comm = MPI.COMM_SELF

        impl_func = self._levelset(self._parameters, xmin, xmax)

        unf_mesh = create_unfitted_impl_Cartesian_mesh(
            comm, impl_func, n, xmin, xmax, exclude_empty_cells=False
        )

        V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree, (dim,)))
        uh = dolfinx.fem.Function(V)
        uh.x.array[:] = self.get_fenicsx_function(u)

        reparam_degree = 10
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
        pv_mesh.points = self._map.evaluate(pv_mesh.points)

        uh_data = uh_reparam.x.array.reshape(-1, dim)

        if dim == 2:
            # Add a zero z-component to make it 3D
            zeros = np.zeros((uh_data.shape[0], 1), dtype=uh_data.dtype)
            uh_data = np.hstack((uh_data, zeros))

            pv_mesh.point_data["uh"] = uh_data

            # pl.add_mesh(pv_mesh, scalars="uh", show_edges=False)
            warped = pv_mesh.warp_by_vector("uh", factor=1.0)  
            pl.add_mesh(warped, scalars="uh", show_edges=False)

            pl.show_axes()
            pl.show()

        if dim == 1:

            pv_mesh.point_data["uh"] = uh_data
            pl.add_mesh(pv_mesh, scalars="uh", show_edges=False)

            pl.show_axes()
            pl.show()

        if dim == 2:

            pl = pv.Plotter(shape=(1, 2))

            ux = uh_data[:,0]
            uy = uh_data[:,1]

            pl.subplot(0, 0)
            mesh_ux = pv_mesh.copy()
            mesh_ux.point_data["ux"] = ux
            mesh_ux.set_active_scalars("ux")
            pl.add_mesh(mesh_ux, show_edges=False)
            pl.show_axes()

            pl.subplot(0, 1)
            mesh_uy = pv_mesh.copy()
            mesh_uy.point_data["uy"] = uy
            mesh_uy.set_active_scalars("uy")
            pl.add_mesh(mesh_uy, show_edges=False)
            pl.show_axes()

            pl.link_views()
            pl.show()

    def get_fenicsx_function(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        return self.C.T @ u

    def write_solution(self, u, name) -> None: 

        unf_mesh = self.create_mesh()
        V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", self._degree, (self._dim,)))
        
        uh = dolfinx.fem.Function(V)
        uh.x.array[:] = self.get_fenicsx_function(u)

        reparam_degree = 3
        reparam = qugar.reparam.create_reparam_mesh(unf_mesh, degree=reparam_degree, levelset=False)
        reparam_mesh = reparam.create_mesh()
        reparam_mesh_wb = reparam.create_mesh(wirebasket=True)

        V_reparam = dolfinx.fem.functionspace(reparam_mesh, ("CG", reparam_degree, (self._dim,)))
        uh_reparam = dolfinx.fem.Function(V_reparam, dtype=dtype)

        cmap = reparam_mesh.topology.index_map(reparam_mesh.topology.dim)
        num_cells = cmap.size_local + cmap.num_ghosts
        cells = np.arange(num_cells, dtype=np.int32)
        interpolation_data = dolfinx.fem.create_interpolation_data(V_reparam, V, cells, padding=1.0e-14)

        uh_reparam.interpolate_nonmatching(uh, cells, interpolation_data=interpolation_data)

        results_folder = Path("results")
        results_folder.mkdir(exist_ok=True, parents=True)
        filename = results_folder / name

        with dolfinx.io.VTKFile(reparam_mesh.comm, filename.with_suffix(".pvd"), "w") as vtk:
            vtk.write_function(uh_reparam)
            vtk.write_mesh(reparam_mesh_wb)