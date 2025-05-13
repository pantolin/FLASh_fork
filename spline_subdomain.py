import numpy as np
import numpy.typing as npt
import scipy.sparse

from typing import Callable
from splines import BSpline2D, BSpline3D

from mpi4py import MPI
from typing import cast

import pyvista as pv
import ufl
import dolfinx
import dolfinx.fem
import dolfinx.fem.petsc

import qugar
import qugar.impl
from qugar.mesh import create_unfitted_impl_Cartesian_mesh
from qugar.dolfinx import CustomForm, ds_bdry_unf, form_custom, mapped_normal

type SparseMatrix = scipy.sparse._csr.csr_matrix
dtype = np.float64

def deafault_bc(X):
    return (0*X[0],0*X[0])

class Subdomain:
    
    def __init__(
        self,
        n: list[int],
        degree: int,
        dim: int,
        parameters: list[int],
        levelset: Callable 
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
        self._parameters = parameters

        self._levelset = levelset
        self._spline_basis = BSpline2D(n, degree)

    def get_all_dofs(self) -> npt.NDArray[np.int32]:
        """Gets an array with all the degrees-of-freedom in the subdomain's
        space.

        Returns:
            npt.NDArray[np.int32]: Sorted array with all the
                degrees-of-freedom.
        """

        number_of_basis = self._spline_basis.get_total_number_basis()
        return np.arange(0, self._dim * number_of_basis)
    
    def get_boundary_dofs(self) -> npt.NDArray[np.int32]:
        """Gets the degrees-of-freedom associated to the fitted boundaries 
        of the subdomain.

        Returns:
            npt.NDArray[np.int32]: Sorted array with all the boundary
            degrees-of-freedom.
        """

        boundary_dofs = self._spline_basis.get_boundary_dofs()
        dim = self._dim
        boundary_dofs = tuple(dim * boundary_dofs + i for i in range(dim))
        return np.column_stack(boundary_dofs).flatten()

    def get_vertices_dofs(self) -> npt.NDArray[np.int32]:
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

        basis_faces_dofs = self._spline_basis.get_edges_dofs()
        dim = self._dim
        faces_dofs = []

        for dofs in basis_faces_dofs:
            dofs = tuple(dim * dofs + i for i in range(dim))
            faces_dofs.append(np.column_stack(dofs).flatten())

        return faces_dofs
    
    def get_interior_dofs(self) -> npt.NDArray[np.int32]:
        """Gets the degrees-of-freedom not associated to the external 
        boundaries of the subdomain 

        Returns:
            npt.NDArray[np.int32]: Sorted array with all the interior
            degrees-of-freedom.
        """        
        boundary_dofs = self.get_boundary_dofs()
        all_dofs = self.get_all_dofs()

        mask = ~np.isin(all_dofs, boundary_dofs)  
        interior_dofs = all_dofs[mask] 

        return interior_dofs
    
    def get_lagrange_extraction(self, x:np.ndarray) -> SparseMatrix:

        dim = self._dim
        basis_C = self._spline_basis.get_lagrange_extraction(x)
        basis_C = basis_C.tocoo()

        rows = basis_C.row
        cols = basis_C.col
        data = basis_C.data

        rows = np.concatenate([dim*rows+i for i in range(dim)])
        cols = np.concatenate([dim*cols+i for i in range(dim)])
        data = np.concatenate([data for i in range(dim)])

        shape = (2*basis_C.shape[0], 2*basis_C.shape[1])
        C = scipy.sparse.coo_matrix((data, (rows, cols)), shape = shape).tocsr()

        return C
    
    def get_lagrange_extraction_connection(self, x:np.ndarray) -> SparseMatrix:

        dim = self._dim
        basis_B = self._spline_basis.get_lagrange_extraction_connection(x)
        basis_B = basis_B.tocoo()

        rows = basis_B.row
        cols = basis_B.col
        data = basis_B.data

        rows = np.concatenate([dim*rows+i for i in range(dim)])
        cols = np.concatenate([dim*cols+i for i in range(dim)])
        data = np.concatenate([data for i in range(dim)])

        shape = (2*basis_B.shape[0], 2*basis_B.shape[1])
        B = scipy.sparse.coo_matrix((data, (rows, cols)), shape = shape).tocsr()

        return B

    def assemble_K(self) -> None:

        dim = self._dim
        n = self._n
        degree = self._degree

        xmin = np.zeros(dim, dtype)
        xmax = np.ones(dim, dtype)
        comm = MPI.COMM_SELF

        impl_func = self._levelset(self._parameters)

        unf_mesh = create_unfitted_impl_Cartesian_mesh(
            comm, impl_func, n, xmin, xmax, exclude_empty_cells=False
        )

        V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree, (dim,)))
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

        def epsilon(u):
            return ufl.sym(ufl.grad(u)) 

        def sigma(u):
            return ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * epsilon(u)

        a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
        a_form = cast(CustomForm, form_custom(a, unf_mesh, dtype=dtype))
        A = dolfinx.fem.petsc.assemble_matrix(a_form, coeffs=a_form.pack_coefficients())
        A.assemble()

        ia, ja, a = A.getValuesCSR()
        K = scipy.sparse.csr_matrix((a, ja, ia), shape=A.getSize())
        C = self.get_lagrange_extraction(V.tabulate_dof_coordinates())
        self.K = C @ K @ C.T

    def pyvista_plot(self) -> None: 

        dim = self._dim
        n = self._n
        degree = self._degree

        xmin = np.zeros(dim, dtype)
        xmax = np.ones(dim, dtype)
        comm = MPI.COMM_SELF

        impl_func = self._levelset(self._parameters)

        unf_mesh = create_unfitted_impl_Cartesian_mesh(
            comm, impl_func, n, xmin, xmax, exclude_empty_cells=False
        )

        reparam = qugar.reparam.create_reparam_mesh(unf_mesh, degree=3, levelset=False)
        reparam_pv = qugar.plot.reparam_mesh_to_PyVista(reparam)

        pl = pv.Plotter(shape=(1, 1))
        pl.add_mesh(reparam_pv.get("reparam"), color="white")
        pl.add_mesh(reparam_pv.get("wirebasket"), color="blue", line_width=2)

        pl.show()