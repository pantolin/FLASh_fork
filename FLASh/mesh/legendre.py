import splipy as sp
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from scipy.special import eval_legendre

def evaluate_legendre_basis(n, x, a=-1.0, b=1.0, normalize=True):
    """
    Evaluate Legendre polynomials P_0 to P_n at points x (mapped to [-1, 1]),
    optionally L^2-normalized over [-1, 1].

    Parameters:
        n         : int
            Maximum degree of the Legendre polynomials.
        x         : scalar or 1D array
            Points in the interval [a, b] where the polynomials are evaluated.
        a, b      : floats
            Interval endpoints (default [-1, 1]).
        normalize : bool
            Whether to return L^2-normalized basis (default: True).

    Returns:
        ndarray
            Array of shape (len(x), n+1) where A[i, j] = normalized P_j(x[i]).
    """
    x = np.atleast_1d(x)
    x_hat = 2 * (x - a) / (b - a) - 1  # map x to [-1, 1]
    basis = np.array([eval_legendre(i, x_hat) for i in range(n + 1)])  # shape (n+1, len(x))
    basis *= np.sqrt(2/(b-a))

    if normalize:
        weights = np.sqrt((2 * np.arange(n + 1) + 1) / 2).reshape(-1, 1)  # shape (n+1, 1)
        basis *= weights

    return basis.T  # shape (len(x), n+1)

class Legendre2D:

    def __init__(self, degree: int, p0 = np.array([0.0,0.0]), p1 = np.array([1.0,1.0])) -> None:

        self._p0 = p0
        self._p1 = p1

        self._degree = degree
        self._number_basis_per_direction = [degree+1, degree+1]
        self._total_number_basis = (1+degree) ** 2


    def evaluate(self, x: np.ndarray) -> np.ndarray:

        fx = evaluate_legendre_basis(self._degree, x[:,0], self._p0[0], self._p1[0])
        fy = evaluate_legendre_basis(self._degree, x[:,1], self._p0[1], self._p1[1])

        f = (fy[:,:,None]*fx[:,None,:]).reshape(x.shape[0], self._total_number_basis)
        
        return f
    
    def get_lagrange_extraction(self, x: np.ndarray) -> np.ndarray:

        return sc.sparse.csr_matrix(self.evaluate(x).T)

    def get_lagrange_extraction_connection(self, x: np.ndarray) -> np.ndarray:

        lagrange_extraction = self.get_lagrange_extraction(x)
        
        lagrange_extraction_connection = lagrange_extraction.copy()
        lagrange_extraction_connection.data[:] = 1

        return lagrange_extraction_connection
    

    def plot_basis(self, n: list[int], basis_index: list[int]) -> None:

        x = np.linspace(self._p0[0], self._p1[0], n[0])
        y = np.linspace(self._p0[1], self._p1[1], n[1])

        X, Y = np.meshgrid(x, y)

        b1 = evaluate_legendre_basis(self._degree, x, self._p0[0], self._p1[0])
        b2 = evaluate_legendre_basis(self._degree, x, self._p0[1], self._p1[1])

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
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def plot_function(self, coefs: np.ndarray, n: list[int]) ->  None:

        x = np.linspace(self._p0[0], self._p1[0], n[0])
        y = np.linspace(self._p0[1], self._p1[1], n[1])

        X, Y = np.meshgrid(x, y)

        b1 = evaluate_legendre_basis(self._degree, x, self._p0[0], self._p1[0])
        b2 = evaluate_legendre_basis(self._degree, x, self._p0[1], self._p1[1])

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
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def get_total_number_basis(self) -> None:

        return self._total_number_basis
    