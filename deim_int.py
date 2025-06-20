import numpy as np
import scipy as sp

from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd

class DataTrimmer:

    def __init__(
        self, 
        snapshots,
    ) -> None:

        self.shape = snapshots.shape
        self.size = snapshots.size

        t_snapshots = snapshots.reshape((self.shape[0], self.shape[1]*self.shape[2])).T
        self.rows = np.where(np.sum(np.abs(t_snapshots), axis = 1) > 0)[0]
        self.row_idx, self.col_idx = np.divmod(self.rows, self.shape[2])
        data = np.ones(shape=self.row_idx.shape)
        csr = sp.sparse.csr_matrix((data, (self.row_idx, self.col_idx)), shape=(self.shape[1], self.shape[2]))
        self.indices = csr.indices
        self.indptr = csr.indptr

    
    def trim_snapshots(
        self,
        snapshots
    ) -> np.ndarray:

        t_snapshots = snapshots.reshape((snapshots.shape[0], self.shape[1]*self.shape[2])).T
        t_snapshots = t_snapshots[self.rows,:]

        return t_snapshots.T
    
    def reconstruct_snapshots(
        self,
        snapshots
    ) -> np.ndarray:
        
        n_snapshots = snapshots.shape[0]

        if n_snapshots > 1:
            r_snapshots = np.zeros((snapshots.shape[0], self.shape[1]*self.shape[2])).T
            r_snapshots[self.rows] = snapshots.T
            r_snapshots = r_snapshots.T.reshape((snapshots.shape[0], self.shape[1], self.shape[2]))

        else:
            
            r_snapshots = sp.sparse.csr_matrix((snapshots[0], self.indices, self.indptr), shape=(self.shape[1], self.shape[2]))

        return r_snapshots

def compute_rSVD_basis(A, tol = 1e-5, k = 50, set_n = False, n = 6):

    tk = np.minimum(k, A.shape[1])
    tn = np.minimum(n, tk)
    
    U, s, Vh = randomized_svd(A, n_components=tk, random_state=0)
    total_variance = np.sum(np.square(s))

    if not set_n: 

        for tn in range(1, s.size):

            n_variance = np.sum(np.square(s[:tn]))

            if (1 - n_variance/total_variance) < tol:
                break

    return U[:,:tn], s, tn

def compute_magic_points(U):

    i = np.argmax(np.abs(U[:,0]))
    I = [i]

    for k in range(1, U.shape[1]):

        r = U[:,k] - U[:,:k] @ np.linalg.solve(U[I,:k], U[I,k])
        i = np.argmax(np.abs(r))
        I.append(i)

    return I

def assemble_snapshot_matrix(snapshots):

    return np.array(snapshots).T

def compute_deim_coefficients(U, I, ks):

    coefficients = np.linalg.solve(U[I,:], ks[:,I].T)

    return coefficients

def compute_aproximations(U, coefficients):

    ks = np.dot(U, coefficients)

    return ks

def create_RBF_interpolator(parameters, coefficients):

    points = parameters
    values = coefficients.T

    if points.ndim == 1:
        points = points.reshape(-1,1)

    if values.ndim == 1:
        values = values.reshape(-1,1)

    interpolator = sp.interpolate.RBFInterpolator(points, values, kernel = "cubic")

    return interpolator

def interpolate_coefficients(interpolator, parameters):

    points = np.array(parameters)

    if points.ndim == 1:
        points = points.reshape(-1,1)
    
    coefficients = interpolator(points)

    return coefficients.T


    size = len(xs)
    max_size = 0
    ind = 0
        
    for i, x in enumerate(xs):
        if x.size > max_size:
            max_size = x.size
            ind = i

    ext_xs = []
    for x in xs:
        ext_x = np.zeros(max_size)
        ext_x[:x.size] = x
        ext_xs.append(ext_x)

    if method == "avg":

        y = np.average(np.array(ext_xs), axis = 0)

    elif method == "max":

        y = np.max(np.array(ext_xs), axis = 0)

    elif method == "lar":

        y = ext_xs[ind]

    return y