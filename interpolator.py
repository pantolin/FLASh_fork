import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

import math
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BarycentricInterpolator
from collections import defaultdict


def evaluate_lagrange_basis(p, nodes, x):

    basis = np.zeros((len(x), p + 1))
    for j in range(p + 1):
        values = np.zeros(p + 1)
        values[j] = 1.0
        interpolator = BarycentricInterpolator(nodes, values)
        basis[:, j] = interpolator(x)

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

class LagrangeInterpolator:

    def __init__(self, d: int, m: int, n: list[int], p: list[int], p0: np.ndarray, p1: np.ndarray) -> None:

        self._d = d
        self._m = m
        self._n = n
        self._p = p

        self._p0 = p0
        self._p1 = p1

        self._create_nodes()

        self._number_basis_per_direction = []

        for ni, pi in zip (n, p):
            self._number_basis_per_direction.append(ni*pi+1)

        self._number_basis = math.prod(self._number_basis_per_direction)

    def _create_nodes(self) -> None:

        self._points = []

        for x0, x1, ni in zip(self._p0, self._p1, self._n):
            self._points.append(np.linspace(x0, x1, ni+1))

        self._nodes = []

        for p, points in zip(self._p, self._points):
            nodes = []
            for x0, x1 in zip(points[:-1], points[1:]):
                nodes.append(get_nodes(p, x0, x1))
            self._nodes.append(nodes)

    def locate_points(self, x):

        x = np.atleast_2d(x)
        n = x.shape[0]
        indices = np.empty((n, self._d), dtype=int)

        for i, grid in enumerate(self._points):
            idx = np.searchsorted(grid, x[:, i], side='right') - 1
            idx = np.clip(idx, 0, len(grid) - 2) 
            indices[:, i] = idx

        return indices

    def group_points_by_element(self, element_indices):

        grouped = defaultdict(list)
        for i, idx in enumerate(element_indices):
            key = tuple(idx)
            grouped[key].append(i)
        return grouped
    
    def evaluate_basis_on_element(self, x, element_index):

        basis_1d = []
        for i in range(self._d):
            nodes_i = self._nodes[i][element_index[i]]  
            basis_vals = evaluate_lagrange_basis(self._p[i], nodes_i, x[:, i])  
            basis_1d.append(basis_vals)

        f = basis_1d[0]

        for i in range(1, self._d):
            shape = (x.shape[0], f.shape[1] * basis_1d[i].shape[1])
            f = (f[:, None, :] * basis_1d[i][:, :, None]).reshape(shape)

        return f
    
    def get_element_basis_indices(self, element_index):

        local_indx = []

        for i, p in zip(element_index, self._p):

            local_indx.append(np.arange(i*p,(i+1)*p+1))

        b_xs = [local_indx[0]]
        xs = local_indx[1:]

        for x in xs:
            for i, b_x in enumerate(b_xs):
                b_xs[i] = np.broadcast_to(b_x[None, :], (x.shape[0], b_x.shape[0])).flatten()

            b_xs.append(np.broadcast_to(x[:, None], (x.shape[0], b_x.shape[0])).flatten())

        indx = b_xs[0]
        count = 1

        for indices, n, p in zip(b_xs[1:], self._n[1:], self._p[1:]):
            count *= (n*p + 1)
            indx += count * indices

        return indx
    
    def evaluate_basis(self, x):

        values = np.zeros((x.shape[0], self._number_basis))
        element_indices = self.locate_points(x)
        grouped_points = self.group_points_by_element(element_indices)

        for elem_idx, point_ids in grouped_points.items():
            x_elem = x[point_ids] 
            basis_vals = self.evaluate_basis_on_element(x_elem, elem_idx)
            basis_index = self.get_element_basis_indices(elem_idx)

            point_ids = np.array(point_ids)
            idx_1 = np.broadcast_to(point_ids[:, None], (point_ids.shape[0], basis_index.shape[0]))
            idx_2 = np.broadcast_to(basis_index[None, :], (point_ids.shape[0], basis_index.shape[0]))

            values[idx_1, idx_2] = basis_vals

        return values
    
    def get_nodes(self):

        all_nodes = []

        for dir_nodes in self._nodes:
            nodes = []
            for x in dir_nodes:
                nodes.append(x[:-1])
            nodes.append(x[-1])
            all_nodes.append(np.hstack(nodes))

        y = (all_nodes[0])[:,None]
        for nodes in all_nodes[1:]:
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

        y = self.evaluate_basis(x)
        return self.evaluate_basis(x) @ self._weights
    
