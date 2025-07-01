import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

import math
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BarycentricInterpolator
from collections import defaultdict

def get_nodes(p, a = -1, b = 1):

    n = p+1
    k = np.arange(n, 0, -1)
    nodes = np.cos(((2*k-1)*np.pi)/(2*n))
    tnodes = 0.5 * (nodes*(b-a) + a + b)

    return tnodes

def get_basis(x):

    y = np.eye(x.size)
    basis = sc.interpolate.BarycentricInterpolator(x, y)

    return basis

def evaluate_basis(basis, x, a = -1, b = 1):

    tx = (2*x-a-b)/(b-a)
    y = basis(tx)

    return y


class Interpolator:

    def __init__(self, d: int, p: int, p0: np.ndarray, p1: np.ndarray) -> None:

        self._d = d
        self._p = p

        self._p0 = p0
        self._p1 = p1

        self._create_1d_basis()
        self._create_nodes()

        self._number_basis_per_direction = p+1
        self._number_basis = (p+1)**d

    def _create_1d_basis(self) -> None: 

        nodes = get_nodes(self._p)
        self._basis = get_basis(nodes)

    def _create_nodes(self) -> None:

        self._nodes = []

        for x0, x1 in zip(self._p0, self._p1):
            self._nodes.append(get_nodes(self._p, x0, x1))

    def evaluate_basis(self, x):

        basis_1d = []
        for ai, bi, xi in zip(self._p0, self._p1, x.T):
            f = evaluate_basis(self._basis, xi, ai, bi)  
            basis_1d.append(f)

        f = basis_1d[0]

        for i in range(1, self._d):
            shape = (x.shape[0], f.shape[1] * basis_1d[i].shape[1])
            f = (f[:, None, :] * basis_1d[i][:, :, None]).reshape(shape)

        return f
    
    def get_nodes(self):

        y = (self._nodes[0])[:,None]
        for nodes in self._nodes[1:]:
            ny = y.shape[0]
            nx = nodes.shape[0]
            d = y.shape[1]
            y = np.broadcast_to(y[None, :, :], (nx, ny, d)).reshape((nx*ny, d))
            x = np.broadcast_to(nodes[:, None, None], (nx, ny, 1)).reshape((nx*ny, 1))
            y = np.column_stack((y, x))

        return y

    def set_weights(self, f):

        nodes = self.get_nodes()
        self._weights = f(nodes)

    def evaluate(self, x):

        return self.evaluate_basis(x) @ self._weights
