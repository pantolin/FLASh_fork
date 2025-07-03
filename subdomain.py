import numpy as np
import numpy.typing as npt
import scipy.sparse
import matplotlib.pyplot as plt

from typing import Callable
from functools import lru_cache

import scipy.sparse.linalg
from splines import BSpline2D, BSpline3D
from legendre import Legendre2D
from gauss_lobatto import Lagrange2D

from mpi4py import MPI
from pathlib import Path

import pyvista as pv
import ufl
import dolfinx
import dolfinx.fem

import qugar
from qugar.mesh import create_unfitted_impl_Cartesian_mesh
from qugar.dolfinx import form_custom
from qugar.mesh.unfitted_cart_mesh import UnfittedCartMesh

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

def intersect_with_active(method):
    def wrapper(self, *args, **kwargs):
        # get_active = kwargs.pop("get_active", True)

        dofs = method(self, *args, **kwargs)

        # if not get_active:
        return dofs

        # if isinstance(dofs, np.ndarray):
        #     return dofs[np.isin(dofs, self._active_dofs)]

        # return [d[np.isin(d, self._active_dofs)] for d in dofs]

    return wrapper

def intersect_with_boundary_active(method):
    @lru_cache
    def wrapper(self, *args, **kwargs):
        get_active = kwargs.pop("get_active", True)

        dofs = method(self, *args, **kwargs)

        # if not get_active:
        return dofs

        # if isinstance(dofs, np.ndarray):
        #     return dofs[np.isin(dofs, self._boundary_active_dofs)]

        # return [d[np.isin(d, self._boundary_active_dofs)] for d in dofs]

    return wrapper

class Subdomain:
    
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
        assemble = False
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
        # self._spline_basis = BSpline2D(n, degree, p0, p1)
        self._spline_basis = Lagrange2D(degree, p0, p1)

        self._total_size = self._spline_basis.get_total_number_basis() * dim

        self._num_primals_per_edge = 3
        self._vertex_primals = True
        self._edge_primals = True
        
        self._create_lagrange_extraction()
        self._set_coordinates()
        self.create_edge_centers()
        self.create_primal_dofs()
        self.create_boundary_dofs()
        
        if assemble == True:
            self.assemble_boundary_integrals()
            self.assemble_K_and_f()

    @intersect_with_active
    @lru_cache
    def get_all_dofs(self) -> npt.NDArray[np.int32]:
        """Gets an array with all the degrees-of-freedom in the subdomain's
        space.

        Returns:
            npt.NDArray[np.int32]: Sorted array with all the
                degrees-of-freedom.
        """

        number_of_basis = self._spline_basis.get_total_number_basis()
        return np.arange(0, self._dim * number_of_basis)

    @intersect_with_boundary_active
    @lru_cache
    def get_vertices_dofs(self) -> list[npt.NDArray[np.int32]]:
        """Gets the degrees-of-freedom associated to the four courners of the
        subdomain.

        The ordering of the four corners follows the basix convention for
        quadrilaterals.
        See https://docs.fenicsproject.org/basix/v0.8.0/index.html

        Returns:
            npt.NDArray[np.int32]: Sorted array with the corner
            degrees-of-freedom.
        """
        basis_vertices_dofs = self._spline_basis.get_vertices_dofs()
        dim = self._dim
        vertices_dofs = []

        for dofs in basis_vertices_dofs:
            dofs = tuple(dim * dofs + i for i in range(dim))
            vertices_dofs.append(np.column_stack(dofs).flatten())

        return vertices_dofs

    @intersect_with_boundary_active
    @lru_cache
    def get_edges_dofs(self) -> list[npt.NDArray[np.int32]]:
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

        basis_edges_dofs = self._spline_basis.get_edges_dofs()
        dim = self._dim
        edges_dofs = []

        for dofs in basis_edges_dofs:
            dofs = tuple(dim * dofs + i for i in range(dim))
            edges_dofs.append(np.column_stack(dofs).flatten())

        return edges_dofs
    
    @intersect_with_boundary_active
    @lru_cache
    def get_interior_edges_dofs(self) -> list[npt.NDArray[np.int32]]:

        vertices_dofs = np.unique(np.hstack(self.get_vertices_dofs()))
        edges_dofs = self.get_edges_dofs()
        int_edges_dofs = []

        for dofs in edges_dofs:
            int_edges_dofs.append(np.setdiff1d(dofs, vertices_dofs))

        return int_edges_dofs
    
    @lru_cache
    def get_boundary_dofs(self) -> npt.NDArray[np.int32]:
        """Gets the degrees-of-freedom associated to the fitted boundaries 
        of the subdomain.

        Returns:
            npt.NDArray[np.int32]: Sorted array with all the boundary
            degrees-of-freedom.
        """
        vertices_dofs = np.hstack(self.get_vertices_dofs())
        edges_dofs = np.hstack(self.get_interior_edges_dofs())

        boundary_dofs = np.hstack([
            vertices_dofs,
            edges_dofs
        ])

        return boundary_dofs

    @lru_cache
    def get_interior_dofs(self) -> npt.NDArray[np.int32]:
        boundary_dofs = self.get_boundary_dofs()
        all_dofs = self.get_all_dofs()

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

        v0 = self._p0
        v1 = np.array([self._p1[0], self._p0[1]])
        v2 = np.array([self._p0[0], self._p1[1]])
        v3 = self._p1

        self._edge_centers = [
            0.5*(v0+v1),
            0.5*(v0+v2),
            0.5*(v1+v3),
            0.5*(v2+v3)
        ]

        self._cell_center = 0.25*(v0+v1+v2+v3)

        self._e_2_v = [
            np.array([0, 1], dtype = np.int32),
            np.array([0, 2], dtype = np.int32),
            np.array([1, 3], dtype = np.int32),
            np.array([2, 3], dtype = np.int32)
        ]

    def create_primal_dofs(self) -> None:

        count = 0

        vertices_dofs = self.get_vertices_dofs()
        self._vertices_primals = np.zeros(shape = (4, 2), dtype=np.int32)

        for i, dofs in enumerate(vertices_dofs):
            self._vertices_primals[i,0] = count
            count += self._dim if len(dofs)>0 and self._vertex_primals else 0
            self._vertices_primals[i,1] = count

        edges_dofs = self.get_interior_edges_dofs()
        self._edges_primals = np.zeros(shape = (4, 2), dtype=np.int32)

        for i, dofs in enumerate(edges_dofs):
            self._edges_primals[i,0] = count
            count += self._num_primals_per_edge if len(dofs)>0 and self._edge_primals else 0
            self._edges_primals[i,1] = count

    def create_boundary_dofs(self) -> None:

        count = 0

        vertices_dofs = self.get_vertices_dofs()
        self._vertices_boundary_ordering = []

        for dofs in vertices_dofs:
            start = count
            count += dofs.size
            self._vertices_boundary_ordering.append(np.arange(start, count))

        edge_dofs = self.get_interior_edges_dofs()
        self._edges_boundary_ordering = []
        
        for dofs in edge_dofs:
            start = count
            count += dofs.size
            self._edges_boundary_ordering.append(np.arange(start, count))

        self._num_boundary_dofs = count 

    def _create_lagrange_extraction(self) -> None:

        unf_mesh = self.create_mesh()
        V, V2 = self._linear_pde.create_function_space(unf_mesh, self._degree)
        x = V.tabulate_dof_coordinates()
        
        basis_C = self._spline_basis.get_lagrange_extraction(x)
        basis_C = basis_C.tocoo()

        dim = self._dim

        rows = basis_C.row
        cols = basis_C.col
        data = basis_C.data

        rows = np.concatenate([dim*rows+i for i in range(dim)])
        cols = np.concatenate([dim*cols+i for i in range(dim)])
        data = np.concatenate([data for i in range(dim)])

        shape = (dim*basis_C.shape[0], dim*basis_C.shape[1])
        self.C = scipy.sparse.coo_matrix((data, (rows, cols)), shape = shape).tocsr()
     
    def _set_coordinates(self) -> None:

        self._x = self._spline_basis.get_nodes()
        
    def create_mesh(self) -> UnfittedCartMesh:

        n = self._n
        comm = MPI.COMM_SELF

        impl_func = self._levelset(self._parameters, self._p0, self._p1)

        unf_mesh = create_unfitted_impl_Cartesian_mesh(
            comm, impl_func, n, self._p0, self._p1, exclude_empty_cells=False
        )

        return unf_mesh
    
    def create_C(self) -> SparseMatrix:

        constraints = [c_0_x, c_0_y, c_1]        
        vertices_primals_ranges, edge_primals_ranges = self.get_primal_ranges()
        vertices_b_dofs, edges_b_dofs = self.get_boundary_ranges()
        edges_dofs = self.get_interior_edges_dofs()
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

        C = scipy.sparse.csr_matrix((values, (rows, cols)), shape=(num_primals, self.get_boundary_dofs().size))

        return C
    
    def assemble_boundary_integrals(self) -> None:

        # if self._linear_pde.bM_model:

        #     bM = self._linear_pde.bM_model.evaluate(self._parameters.reshape(1,4))
        #     bM = bM.reshape(self._total_size, self._total_size)

        # else:

        unf_mesh = self.create_mesh()
        V, V2 = self._linear_pde.create_function_space(unf_mesh, self._degree)
        bM = self._linear_pde.assemble_boundary_mass(unf_mesh, V, self.C)

        self.bM = bM

        self._boundary_mass = self.bM.sum(axis=1)
        self._boundary_active_dofs = np.nonzero(self._boundary_mass)[0]

    def assemble_K_and_f(self) -> None:

        unf_mesh = self.create_mesh()
        V, V2 = self._linear_pde.create_function_space(unf_mesh, self._degree)

        if self._linear_pde.K_model:

            K = self._linear_pde.K_model.evaluate(self._parameters.reshape(1,4))
            K = K.reshape(self._total_size, self._total_size)
            K = 0.5*(K + K.T)

        else: 

            K = self._linear_pde.assemble_stiffness(unf_mesh, V, self.C)

        M = self._linear_pde.assemble_mass(unf_mesh, V, self.C)
        f = self._linear_pde.assemble_right_hand_side(unf_mesh, V, V2, self.C)
        
        self.K = K
        self.f = f
        self.M = M

        self._active_dofs = np.nonzero(self.K.diagonal())[0]

    def assemble_K(self) -> np.ndarray:

        unf_mesh = self.create_mesh()
        V, V2 = self._linear_pde.create_function_space(unf_mesh, self._degree)

        return self._linear_pde.assemble_stiffness(unf_mesh, V, self.C)
    
    def assemble_M(self) -> np.ndarray:

        unf_mesh = self.create_mesh()
        V, V2 = self._linear_pde.create_function_space(unf_mesh, self._degree)

        return self._linear_pde.assemble_mass(unf_mesh, V, self.C)
    
    def assemble_bM(self) -> np.ndarray:

        unf_mesh = self.create_mesh()
        V, V2 = self._linear_pde.create_function_space(unf_mesh, self._degree)

        return self._linear_pde.assemble_boundary_mass(unf_mesh, V, self.C)

    def get_projected_function(self, f, boundary = False) -> npt.NDArray[np.float64]:

        b = (np.array(f(self._x.T)).T).flatten() 

        n_dofs = self._total_size
        u = np.zeros(shape = (n_dofs))

        if boundary:

            M = self.bM
            y = M @ b

            a_dofs = self._boundary_active_dofs

            M = M[a_dofs,:][:,a_dofs]
            y = y[a_dofs]

            u[a_dofs] = scipy.sparse.linalg.spsolve(M, y)

        else:

            M = self.M 
            y = M @ b

            a_dofs = self._active_dofs

            M = M[a_dofs,:][:,a_dofs]
            y = y[a_dofs]

            u[a_dofs] = scipy.sparse.linalg.spsolve(M, y)

        return u
    
    def get_boundary_constraint(self, f) -> npt.NDArray[np.float64]:

        b = (np.array(f(self._x.T)).T).flatten()
        y = self.bM @ b 

        return y 

    def plot(self) -> None: 

        unf_mesh = self.create_mesh()

        reparam = qugar.reparam.create_reparam_mesh(unf_mesh, degree=3, levelset=False)
        reparam_pv = qugar.plot.reparam_mesh_to_PyVista(reparam)

        pl = pv.Plotter(shape=(1, 1))
        pl.add_mesh(reparam_pv.get("reparam"), color="white")
        pl.add_mesh(reparam_pv.get("wirebasket"), color="blue", line_width=2)
        pl.view_xy()
        pl.show()

    def plot_mesh_nodes(self) -> None:

        unf_mesh = self.create_mesh()
        V, _ = self._linear_pde.create_function_space(unf_mesh, self._degree)
        x = V.tabulate_dof_coordinates()

        plt.figure(figsize=(6, 5))
        plt.plot(x[:,0], x[:,1], 'o') 
        plt.title(f'Mesh nodes')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xticks(self._spline_basis._nodes[0])
        plt.yticks(self._spline_basis._nodes[1])
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_function(self, u) -> None:

        pl = pv.Plotter(shape=(1, 1))

        dim = self._dim
        n = self._n
        degree = self._degree

        comm = MPI.COMM_SELF

        impl_func = self._levelset(self._parameters, self._p0, self._p1)

        unf_mesh = create_unfitted_impl_Cartesian_mesh(
            comm, impl_func, n, self._p0, self._p1, exclude_empty_cells=False
        )

        V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree, (dim,)))
        uh = dolfinx.fem.Function(V)
        uh.x.array[:] = self.get_lagrange_function(u)

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

    def get_lagrange_function(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        return self.C.T @ u
    
    def compute_error(self, u, u_ex) -> float: 

        dim = self._dim
        n = self._n
        degree = self._degree

        comm = MPI.COMM_SELF

        impl_func = self._levelset(self._parameters, self._p0, self._p1)

        unf_mesh = create_unfitted_impl_Cartesian_mesh(
            comm, impl_func, n, self._p0, self._p1, exclude_empty_cells=False
        )

        V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree, (dim,)))
        uh = dolfinx.fem.Function(V)
        uh.x.array[:] = self.get_lagrange_function(u)

        uh_ex = dolfinx.fem.Function(V)
        uh_ex.name = "uh_ex"
        uh_ex.interpolate(u_ex)

        # Compute L2 error
        error_form = form_custom((uh - uh_ex) ** 2 * ufl.dx, unf_mesh, dtype=dtype)
        error_L2 = np.sqrt(
            unf_mesh.comm.allreduce(
                dolfinx.fem.assemble_scalar(error_form, coeffs=error_form.pack_coefficients()),
                op=MPI.SUM, 
            )
        )

        return error_L2

    def write_solution(self, u, name) -> None: 

        unf_mesh = self.create_mesh()
        V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", self._degree, (self._dim,)))
        
        uh = dolfinx.fem.Function(V)
        uh.x.array[:] = self.get_lagrange_function(u)

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