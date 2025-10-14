import numpy as np
import scipy as sp

from sklearn.utils.extmath import randomized_svd

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