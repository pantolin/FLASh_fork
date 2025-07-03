import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BarycentricInterpolator

def evaluate_lagrange_basis(n, x, a=-1.0, b=1.0, lobatto=True):
    """
    Evaluate Lagrange basis functions at Gauss–Legendre or Gauss–Lobatto–Legendre nodes.

    Parameters:
        n       : int
            Degree of the polynomial basis (n+1 basis functions).
        x       : scalar or 1D array
            Evaluation points (in interval [a, b]).
        a, b    : floats
            Interval endpoints (default: [-1, 1]).
        lobatto : bool
            If True, use Gauss–Lobatto–Legendre nodes; otherwise, Gauss–Legendre.

    Returns:
        ndarray
            Matrix of shape (len(x), n+1) with basis values: A[i, j] = φ_j(x[i])
    """
    x = np.atleast_1d(x)
    x_hat = 2 * (x - a) / (b - a) - 1 

    if lobatto:
        from numpy.polynomial.legendre import legder, legval
        def compute_gll_nodes(n):
            if n == 1:
                return np.array([-1.0, 1.0])
            Pn = np.zeros(n+1)
            Pn[-1] = 1  # P_n(x)
            dPn = legder(Pn)
            roots = np.polynomial.legendre.legroots(dPn) 
            return np.concatenate(([-1.0], roots, [1.0]))

        nodes = compute_gll_nodes(n)
    else:
        nodes, _ = leggauss(n + 1)

    basis = np.zeros((len(x_hat), n + 1))
    for j in range(n + 1):
        values = np.zeros(n + 1)
        values[j] = 1.0
        interpolator = BarycentricInterpolator(nodes, values)
        basis[:, j] = interpolator(x_hat)

    return basis

def get_nodes(n, a=-1.0, b=1.0, lobatto=True):

    if lobatto:
        from numpy.polynomial.legendre import legder, legval
        def compute_gll_nodes(n):
            if n == 1:
                return np.array([-1.0, 1.0])
            Pn = np.zeros(n+1)
            Pn[-1] = 1  # P_n(x)
            dPn = legder(Pn)
            roots = np.polynomial.legendre.legroots(dPn) 
            return np.concatenate(([-1.0], roots, [1.0]))

        nodes = compute_gll_nodes(n)
    else:
        nodes, _ = leggauss(n + 1)

    t_nodes = 0.5 * (a + b + (b - a) * nodes)
    return t_nodes

class Lagrange2D:

    def __init__(self, degree: int, p0 = np.array([0.0,0.0]), p1 = np.array([1.0,1.0])) -> None:

        self._degree = degree

        self._p0 = p0
        self._p1 = p1

        nodes_x = get_nodes(degree, p0[0], p1[0])
        nodes_y = get_nodes(degree, p0[1], p1[1])

        self._nodes = [nodes_x, nodes_y]

        self._number_basis_per_direction = [1+degree, 1+degree]
        self._total_number_basis = (1+degree) ** 2

    def evaluate(self, x: np.ndarray) -> np.ndarray:

        fx = evaluate_lagrange_basis(self._degree, x[:,0], self._p0[0], self._p1[0])
        fy = evaluate_lagrange_basis(self._degree, x[:,1], self._p0[1], self._p1[1])

        f = (fy[:,:,None]*fx[:,None,:]).reshape(x.shape[0], self._total_number_basis)
        
        return f
    
    def get_lagrange_extraction(self, x: np.ndarray) -> np.ndarray:

        return sc.sparse.csr_matrix(self.evaluate(x).T)

    def get_lagrange_extraction_connection(self, x: np.ndarray) -> np.ndarray:

        lagrange_extraction = self.get_lagrange_extraction(x)
        
        lagrange_extraction_connection = lagrange_extraction.copy()
        lagrange_extraction_connection.data[:] = 1

        return lagrange_extraction_connection
    
    def get_interpolation_points(self) -> list[np.ndarray]:
        return self._nodes
    
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

        x = np.linspace(self._p0[0], self._p1[0], n[0])
        y = np.linspace(self._p0[1], self._p1[1], n[1])

        X, Y = np.meshgrid(x, y)

        b1 = evaluate_lagrange_basis(self._degree, x, self._p0[0], self._p1[0])
        b2 = evaluate_lagrange_basis(self._degree, y, self._p0[1], self._p1[1])

        A = b1[None, :, None, :]
        B = b2[:, None, :, None]
        C = (B * A).reshape(n[0], n[1], self._total_number_basis) 

        for i in basis_index:

            plt.figure(figsize=(6, 5))
            contour = plt.contourf(X, Y, C[:,:,i], levels=20, cmap='viridis')

            plt.colorbar(contour)
            plt.title(f'Contour Plot of $B{i}$')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xticks(self._nodes[0])
            plt.yticks(self._nodes[1])
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def plot_function(self, coefs: np.ndarray, n: list[int]) ->  None:

        x = np.linspace(self._p0[0], self._p1[0], n[0])
        y = np.linspace(self._p0[1], self._p1[1], n[1])

        X, Y = np.meshgrid(x, y)

        b1 = evaluate_lagrange_basis(self._degree, x, self._p0[0], self._p1[0])
        b2 = evaluate_lagrange_basis(self._degree, y, self._p0[1], self._p1[1])

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
        plt.xticks(self._nodes[0])
        plt.yticks(self._nodes[1])
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def get_total_number_basis(self) -> None:

        return self._total_number_basis
