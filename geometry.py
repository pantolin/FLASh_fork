import numpy as np
import scipy as sc
import splipy as sp
import matplotlib.pyplot as plt

from typing import Self, Callable
from splipy import surface_factory

from numpy.polynomial.legendre import leggauss

def compute_extraction_operators(U, p):
    """
    Compute the local extraction operators for a 1D B-spline.

    Parameters
    ----------
    U : array_like
        Knot vector (1D array of length m)
    p : int
        Degree of the B-spline basis

    Returns
    -------
    C_list : list of (p+1, p+1) np.ndarray
        Element extraction operators
    """
    U = np.asarray(U, dtype=float)
    m = len(U)
    
    a = p   
    b = a + 1     
    nb = 0        

    C_list = [np.eye(p + 1)]  

    while b < m - 1:
        C_next = np.eye(p + 1)
        i = b

        while b < m -1 and U[b] == U[b + 1]:
            b += 1
        mult = b - i + 1

        if mult < p:
            r = p - mult
            alphas = np.zeros(r)

            numer = U[b] - U[a]  
            for j in range(p, mult, -1):
                alphas[j - mult - 1] = numer / (U[a + j] - U[a])

            C_current = C_list[-1]

            for j in range(1, r + 1):
                save = r - j
                s = mult + j

                for k in range(p, s - 1, -1): 
                    alpha = alphas[k - s]
                    C_current[:, k] = alpha * C_current[:, k] + (1.0 - alpha) * C_current[:, k - 1]

                if b < m - 1:
                    C_next[save:save+j+1, save] = C_current[p-j:p+1, p]

        if b < m - 1:
            a = b
            b += 1

            C_list.append(C_next)
            nb += 1

    return C_list

def get_basis_support(U, p):
    """
    Given a knot vector U and degree p, return a list where each element
    contains the indices of basis functions that have support on that element.

    Parameters
    ----------
    U : array_like
        Knot vector (non-decreasing, possibly with repeated knots)
    p : int
        Degree of B-spline basis

    Returns
    -------
    element_basis : list of lists
        element_basis[i] contains indices of basis functions that are nonzero
        on the i-th element (knot span with U[i] != U[i+1])
    elements : list of (start, end)
        The actual knot intervals forming the elements
    """
    U = list(U)
    m = len(U)
    n = m - p - 1  # number of basis functions (usually)
    
    element_basis = []
    elements = []

    for i in range(m - 1):
        if U[i] != U[i + 1]:  # Skip zero-length knot spans (repeated knots)
            elements.append((U[i], U[i + 1]))
            supported = []
            for j in range(n):
                if U[j] < U[i + 1] and U[j + p + 1] > U[i]:
                    supported.append(j)
            element_basis.append(supported)

    return element_basis, elements

def find_element_containing_point(a, elements):
    """
    Given a point a and a list of knot intervals (elements),
    return the index of the element that contains a.

    Parameters
    ----------
    a : float
        Point in the parametric domain
    elements : list of (start, end)
        Knot spans from get_basis_support

    Returns
    -------
    e : int
        Index of the element containing a, or None if not found
    """
    for e, (u_start, u_end) in enumerate(elements):

        if (u_start <= a < u_end) or (e == len(elements) - 1 and a == u_end):
            return e
    return None 


class SomeName:

    def __init__(
        self,
        degree: float
    ) -> None:

        self.degree = degree
        self.basis = sp.BSplineBasis(degree+1, [0] * (degree + 1) + [1] * (degree + 1)) 

    def assemble_mass(
        self
    ) -> None:
        
        n_quad = self.degree + 1
        xi, w = leggauss(n_quad)

        x = 0.5 * (xi + 1)
        w = 0.5 * w

        f = self.basis.evaluate(x)

        self.m = (w[:,None,None] * f[:,:,None] * f[:,None,:]).sum(axis=0)
        self.l = np.linalg.cholesky(self.m)

    def fit(
        self,
        fun: Callable
    ) -> None:
        
        n_quad = self.degree + 1
        xi, w = leggauss(n_quad)

        x = 0.5 * (xi + 1)
        w = 0.5 * w

        b = self.basis.evaluate(x)

        X, Y = np.meshgrid(x, x, indexing='ij')
        xy = np.stack((X, Y), axis=-1)
        xy = xy.reshape(-1, 2)

        f = fun(xy)
        f = f.reshape(x.size, x.size, *f.shape[1:])

        rhs = np.einsum("ij...,ik,jl,i,j->kl...", f, b, b, w, w)
        inv_m = np.linalg.inv(self.m)
        self.c = np.einsum("ki,lj,kl...->ij...", inv_m, inv_m, rhs)

    def evaluate(
        self,
        x
    ) -> np.ndarray:

        bx = self.basis.evaluate(x[:,0])
        by = self.basis.evaluate(x[:,1])

        f = np.einsum('ik,ij,kj...->i...', bx, by, self.c)

        return f

    def evaluate_basis(
        self,
        x
    ) -> np.ndarray:

        bx = self.basis.evaluate(x[:,0])
        by = self.basis.evaluate(x[:,1])

        f = (bx[:,:,None]*by[:,None,:]).reshape((x.shape[0], -1))

        return f


class BezierElement:

    def __init__(
        self,
        degree: int,
        control_points: np.ndarray,
    ) -> None:
        
        self.degree = degree
        self.basis = sp.BSplineBasis(degree+1, [0] * (degree + 1) + [1] * (degree + 1)) 
        self.control_points = control_points

    def evaluate(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        
        fx = self.basis.evaluate(x[:,0])
        fy = self.basis.evaluate(x[:,1])

        f = np.einsum('ik,ij,kjd->id', fx, fy, self.control_points)

        return f
    
    def evaluate_jacobian(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        
        fx = self.basis.evaluate(x[:,0])
        fy = self.basis.evaluate(x[:,1])
        dfx = self.basis.evaluate(x[:,0], d = 1)
        dfy = self.basis.evaluate(x[:,1], d = 1)

        jfx = np.einsum('ik,ij,kjd->id', dfx, fy, self.control_points[:,:,:2])
        jfy = np.einsum('ik,ij,kjd->id', fx, dfy, self.control_points[:,:,:2])

        jf = np.stack((jfx, jfy), axis=-1)

        return jf
    
    def evaluate_jacobian_determinant(
        self,
        x: np.ndarray
    ) -> np.ndarray:

        jf = self.evaluate_jacobian(x)
        det = jf[:,0,0]*jf[:,1,1] - jf[:,0,1]*jf[:,1,0]

        return det

    def evaluate_jacobian_inverse(
        self,
        x: np.ndarray
    ) -> np.ndarray:

        jf = self.evaluate_jacobian(x)
        det = jf[:,0,0]*jf[:,1,1] - jf[:,0,1]*jf[:,1,0]

        ijf = np.empty_like(jf)

        ijf[:,0,0] =  jf[:,1,1] / det
        ijf[:,0,1] = -jf[:,0,1] / det
        ijf[:,1,0] = -jf[:,1,0] / det
        ijf[:,1,1] =  jf[:,0,0] / det

        return ijf

    def evaluate_A(
        self,
        x: np.ndarray,
        lambda_: float = 1,
        mu: float = 1
    ) -> np.ndarray:

        jf = self.evaluate_jacobian(x)
        det = jf[:,0,0]*jf[:,1,1] - jf[:,0,1]*jf[:,1,0]

        ijf = np.empty_like(jf)

        ijf[:,0,0] =  jf[:,1,1] / det
        ijf[:,0,1] = -jf[:,0,1] / det
        ijf[:,1,0] = -jf[:,1,0] / det
        ijf[:,1,1] =  jf[:,0,0] / det

        kron = np.zeros_like(jf)

        kron[:,0,0] =  1
        kron[:,1,1] =  1

        a1 = np.einsum('npk,nql->npkql', ijf, ijf)
        a2 = np.einsum('nlk,nps,nqs->npkql', kron, ijf, ijf)
        a3 = np.einsum('npl,nqk->npkql', ijf, ijf)

        a = lambda_ * a1 + mu * (a2 + a3)
        a = det[:,None,None,None,None] * a

        return a

    def evaluate_arclen(
        self,
        x: np.ndarray
    ) -> np.ndarray:

        tol = 1e-6
        jf = self.evaluate_jacobian(x)

        is_vert = (np.abs(x[:,0] - 0) < tol) | (np.abs(x[:,0] - 1) < tol)
        is_horiz = (np.abs(x[:,1] - 0) < tol) | (np.abs(x[:,1] - 1) < tol)

        tangents = np.empty((x.shape[0],2))
        tangents[is_vert] = jf[is_vert][:, :, 1]
        tangents[is_horiz] = jf[is_horiz][:, :, 0]

        arclen = np.linalg.norm(tangents, axis=1)

        return arclen


    def plot(self) -> None:

        fig, ax = plt.subplots()

        fx = self.basis.evaluate(np.linspace(0,1,10))
        fy = self.basis.evaluate(np.linspace(0,1,10))

        ft = np.einsum('ik,jl,kld->ijd', fx, fy, self.control_points)

        X = ft[:, :, 0] 
        Y = ft[:, :, 1]

        for i in range(1, ft.shape[0] - 1):
            ax.plot(X[i, :], Y[i, :], color='black', linewidth=0.25, alpha = 0.5)
        for j in range(1, ft.shape[1] - 1):
            ax.plot(X[:, j], Y[:, j], color='black', linewidth=0.25, alpha = 0.5)

        ax.plot(X[:, 0], Y[:, 0], color='red', linewidth=1.5, label='Boundary')
        ax.plot(X[:, -1], Y[:, -1], color='red', linewidth=1.5)
        ax.plot(X[0, :], Y[0, :], color='red', linewidth=1.5)
        ax.plot(X[-1, :], Y[-1, :], color='red', linewidth=1.5)

        ax.set_title("Mapped Domain (Parametric Grid Image)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()
    

class SplineGeometry:

    def __init__(
        self,
        knots: list[int],
        degree: int,
        basis: list,
        control_points: np.ndarray,
        levelset
    ) -> None:
        
        self.knots = knots
        self.degree = degree
        self.basis = basis
        self.control_points = control_points
        self.levelset = levelset

        self.p0 = [knots[0][0], knots[1][0]]
        self.p1 = [knots[0][-1], knots[1][-1]]

        element_basis_x, elements_x = get_basis_support(knots[0], degree)
        element_basis_y, elements_y = get_basis_support(knots[1], degree)

        n = [len(elements_x), len(elements_y)]

        self.elements = [elements_x, elements_y]
        self.elements_basis = [element_basis_x, element_basis_y]

        self.n = n

        self.create_bezier_extraction()

    @staticmethod
    def interpolate_map(
        knots: list[int],
        degree: int,
        map: Callable,
        levelset
    ) -> Self:

        basis_x = sp.BSplineBasis(degree+1, knots[0]) 
        basis_y = sp.BSplineBasis(degree+1, knots[1])

        basis = [basis_x, basis_y]

        greeville_x = basis_x.greville()
        greeville_y = basis_y.greville()

        X, Y = np.meshgrid(greeville_x, greeville_y, indexing='ij')
        out = map(X, Y) 

        surface = surface_factory.interpolate(out, basis)
        control_points = surface.controlpoints

        return SplineGeometry(knots, degree, basis, control_points, levelset)
    
    def create_bezier_extraction(self) -> None:

        C_x_list = compute_extraction_operators(self.knots[0], self.degree)
        C_y_list = compute_extraction_operators(self.knots[1], self.degree)

        self.C = [C_x_list, C_y_list]    

    def get_bezier_element(self, x) -> BezierElement:

        i = find_element_containing_point(x[0], self.elements[0])
        j = find_element_containing_point(x[1], self.elements[1])

        Cx = self.C[0][i]
        Cy = self.C[1][j]

        basis_x = self.elements_basis[0][i]
        basis_y = self.elements_basis[1][j]

        element_control_points = self.control_points[np.ix_(basis_x, basis_y, np.arange(self.control_points.shape[2]))]
        control_points = np.einsum('ki,lj,kld->ijd', Cx, Cy, element_control_points)

        return BezierElement(self.degree, control_points)

    def plot(self) -> None:

        knots_x = self.knots[0]
        knots_y = self.knots[1]

        basis_x = self.basis[0]
        basis_y = self.basis[1]

        control_points = self.control_points

        fig, ax = plt.subplots()

        for x0, x1 in zip(knots_x[:-1], knots_x[1:]):
            for y0, y1 in zip(knots_y[:-1], knots_y[1:]):

                fx = basis_x.evaluate(np.linspace(x0,x1,10))
                fy = basis_y.evaluate(np.linspace(y0,y1,10))

                ft = np.einsum('ik,jl,kld->ijd', fx, fy, control_points)

                X = ft[:, :, 0] 
                Y = ft[:, :, 1]

                for i in range(1, ft.shape[0] - 1):
                    ax.plot(X[i, :], Y[i, :], color='black', linewidth=0.25, alpha = 0.5)
                for j in range(1, ft.shape[1] - 1):
                    ax.plot(X[:, j], Y[:, j], color='black', linewidth=0.25, alpha = 0.5)


                ax.plot(X[:, 0], Y[:, 0], color='red', linewidth=1.5, label='Boundary')
                ax.plot(X[:, -1], Y[:, -1], color='red', linewidth=1.5)
                ax.plot(X[0, :], Y[0, :], color='red', linewidth=1.5)
                ax.plot(X[-1, :], Y[-1, :], color='red', linewidth=1.5)

        # ax.set_aspect('equal')
        ax.set_title("Mapped Domain (Parametric Grid Image)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()

    def countour(
        self
    ) -> None:

        knots_x = self.knots[0]
        knots_y = self.knots[1]

        p = 2
        element = SomeName(p)
        element.assemble_mass()

        levels = np.linspace(0, 0.01, 20)

        fig, ax = plt.subplots()

        for x0, x1 in zip(knots_x[self.degree:-self.degree-1], knots_x[self.degree+1:-self.degree]):
            for y0, y1 in zip(knots_y[self.degree:-self.degree-1], knots_y[self.degree+1:-self.degree]):

                center = 0.5*np.array([x0+x1,y0+y1])
                bezier_element = self.get_bezier_element(center)

                x = np.linspace(0, 1, 10)
                X, Y = np.meshgrid(x, x, indexing="ij")
                X = np.stack((X, Y), axis = -1).reshape(-1,2)

                f = bezier_element.evaluate(X)
                f = f.reshape(10, 10, 3)

                element.fit(bezier_element.evaluate_A)
                f_exact = bezier_element.evaluate_A(X)
                f_spline = element.evaluate(X)

                error = f_exact - f_spline

                f_norm = np.sqrt(np.sum(f_exact**2, axis=tuple(range(1, f_exact.ndim))))
                f_error = np.sqrt(np.sum(error**2, axis=tuple(range(1, error.ndim))))

                r = f_error/f_norm
                r = r.reshape(10, 10)

                X = f[:, :, 0]
                Y = f[:, :, 1]

                contour = ax.contourf(X, Y, r, levels=levels, cmap='viridis')

                ax.plot(X[:, 0], Y[:, 0], color='red', linewidth=1.5)
                ax.plot(X[:, -1], Y[:, -1], color='red', linewidth=1.5)
                ax.plot(X[0, :], Y[0, :], color='red', linewidth=1.5)
                ax.plot(X[-1, :], Y[-1, :], color='red', linewidth=1.5)

        fig.colorbar(contour, ax=ax, label="Scalar value")
        ax.set_title("Function")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()

    def plot_det(
        self
    ) -> None:

        knots_x = self.knots[0]
        knots_y = self.knots[1]

        levels = np.linspace(0.99, 1.01, 20)

        fig, ax = plt.subplots()

        for x0, x1 in zip(knots_x[self.degree:-self.degree-1], knots_x[self.degree+1:-self.degree]):
            for y0, y1 in zip(knots_y[self.degree:-self.degree-1], knots_y[self.degree+1:-self.degree]):

                center = 0.5*np.array([x0+x1,y0+y1])
                bezier_element = self.get_bezier_element(center)

                x = np.linspace(0, 1, 10)
                X, Y = np.meshgrid(x, x, indexing="ij")
                X = np.stack((X, Y), axis = -1).reshape(-1,2)

                f = bezier_element.evaluate(X)
                f = f.reshape(10, 10, 3)

                det = bezier_element.evaluate_jacobian_determinant(X)
                det = det.reshape(10, 10)

                X = f[:, :, 0]
                Y = f[:, :, 1]

                contour = ax.contourf(X, Y, det, levels=levels, cmap='viridis')

                ax.plot(X[:, 0], Y[:, 0], color='red', linewidth=1.5)
                ax.plot(X[:, -1], Y[:, -1], color='red', linewidth=1.5)
                ax.plot(X[0, :], Y[0, :], color='red', linewidth=1.5)
                ax.plot(X[-1, :], Y[-1, :], color='red', linewidth=1.5)

        fig.colorbar(contour, ax=ax, label="Scalar value")
        ax.set_title("Function")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()

    
