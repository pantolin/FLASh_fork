import splipy as sp
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from functools import reduce
from matplotlib.patches import Circle

class BSpline2D:

    def __init__(self, n: list[int], degree: int, p0 = np.array([0.0,0.0]), p1 = np.array([1.0,1.0])) -> None:

        self._n = n
        self._degree = degree

        knots_x = [p0[0]]*degree + list(np.linspace(p0[0],p1[0],n[0]+1)) + [p1[0]]*degree
        knots_y = [p0[1]]*degree + list(np.linspace(p0[1],p1[1],n[1]+1)) + [p1[1]]*degree

        self._knots = [knots_x, knots_y]

        basis_x = sp.BSplineBasis(degree+1, knots_x) 
        basis_y = sp.BSplineBasis(degree+1, knots_y)

        self._BSpline_basis = [basis_x, basis_y]
        self._number_basis_per_direction = [n[0]+degree, n[1]+degree]
        self._total_number_basis = (n[0]+degree) * (n[1]+degree)

        self._compute_greville_points()

    def _compute_greville_points(self) -> None:

        knots_x = np.array(self._knots[0])
        knots_y = np.array(self._knots[1])

        p = self._degree
        n = self._number_basis_per_direction

        greville_x = np.array([np.sum(knots_x[i+1:i+p+1]) / p for i in range(n[0])])
        greville_y = np.array([np.sum(knots_y[i+1:i+p+1]) / p for i in range(n[1])])

        self._greville_points = [greville_x, greville_y]

    def evaluate(self, x: np.ndarray) -> np.ndarray:

        fx = self._BSpline_basis[0].evaluate(x[:,0])
        fy = self._BSpline_basis[1].evaluate(x[:,1])

        f = (fy[:,:,None]*fx[:,None,:]).reshape(x.shape[0], self._total_number_basis)
        
        return f
    
    def get_lagrange_extraction(self, x: np.ndarray) -> np.ndarray:

        return sc.sparse.csr_matrix(self.evaluate(x).T)

    def get_lagrange_extraction_connection(self, x: np.ndarray) -> np.ndarray:

        lagrange_extraction = self.get_lagrange_extraction(x)
        
        lagrange_extraction_connection = lagrange_extraction.copy()
        lagrange_extraction_connection.data[:] = 1

        return lagrange_extraction_connection
    
    def get_greville_points(self) -> list[np.ndarray]:
        return self._greville_points
    
    def get_boundary_dofs(self) -> np.ndarray:

        nx = self._number_basis_per_direction[0]
        ny = self._number_basis_per_direction[1]
        
        ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        indices = (iy * nx + ix).flatten()
        boundary_mask = (
            (ix == 0) | (ix == nx - 1) |
            (iy == 0) | (iy == ny - 1) 
        ).flatten()

        boundary_indices = indices[boundary_mask]

        return np.sort(boundary_indices)
    
    def get_edges_dofs(self) -> np.ndarray:

        nx = self._number_basis_per_direction[0]
        ny = self._number_basis_per_direction[1]

        ix, iy = np.meshgrid(
            np.arange(nx), np.arange(ny), indexing='ij'
        )

        ix, iy = ix.flatten(), iy.flatten()
        node_ids = ix + iy * nx

        edges_dofs = [
            np.sort(node_ids[iy == 0]),
            np.sort(node_ids[ix == 0]),
            np.sort(node_ids[ix == nx - 1]),
            np.sort(node_ids[iy == ny - 1]),
        ]

        return edges_dofs
    
    def get_vertices_dofs(self) -> np.ndarray:

        nx = self._number_basis_per_direction[0]
        ny = self._number_basis_per_direction[1]

        ix, iy = np.meshgrid(
            np.arange(nx), np.arange(ny), indexing='ij'
        )

        ix, iy = ix.flatten(), iy.flatten()
        node_ids = ix + iy * nx

        def mask(*conds):
            return np.logical_and.reduce(list(conds))

        vertices_dofs = [
            node_ids[mask(ix==0, iy==0)],
            node_ids[mask(ix==nx-1, iy==0)],
            node_ids[mask(ix==0, iy==ny-1)],
            node_ids[mask(ix==nx-1, iy==ny-1)],
        ]

        return vertices_dofs

    def plot_basis(self, n: list[int], basis_index: list[int]) -> None:

        x = np.linspace(0, 1, n[0])
        y = np.linspace(0, 1, n[1])

        X, Y = np.meshgrid(x, y)

        b1 = self._BSpline_basis[0].evaluate(x)
        b2 = self._BSpline_basis[0].evaluate(x)

        A = b1[None, :, None, :]
        B = b2[:, None, :, None]
        C = (B * A).reshape(n[0], n[1], self._total_number_basis) 

        for i in basis_index:

            plt.figure(figsize=(6, 5))
            contour = plt.contourf(X, Y, C[:,:,i//2], levels=20, cmap='viridis')

            plt.colorbar(contour)
            plt.title(f'Contour Plot of $B{i}$')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xticks(self._knots[0])
            plt.yticks(self._knots[1])
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def plot_function(self, coefs: np.ndarray, n: list[int]) ->  None:

        x = np.linspace(self._knots[0][0], self._knots[0][-1], n[0])
        y = np.linspace(self._knots[1][0], self._knots[1][-1], n[1])

        X, Y = np.meshgrid(x, y)

        b1 = self._BSpline_basis[0].evaluate(x)
        b2 = self._BSpline_basis[0].evaluate(x)

        A = b1[None, :, None, :]
        B = b2[:, None, :, None]
        C = (B * A).reshape(n[0], n[1], self._total_number_basis) 

        D = (C * coefs[None, None, :]).sum(axis=2)
        # D_clipped = np.clip(D, 0, 0.1)

        plt.figure(figsize=(6, 5))
        contour = plt.contourf(X, Y, D, levels=20, cmap='viridis')#, vmin=0, vmax=1)
        # contour = plt.contourf(X, Y, D_clipped, levels=20, cmap='viridis')

        plt.colorbar(contour)
        plt.title(f'Contour Plot of $F$')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xticks(self._knots[0])
        plt.yticks(self._knots[1])
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def get_total_number_basis(self) -> None:

        return self._total_number_basis