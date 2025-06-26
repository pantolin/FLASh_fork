import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import h5py
import os
import pickle

from mpi4py import MPI
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans

from deim_int import compute_rSVD_basis, compute_magic_points, compute_deim_coefficients
from deim_int import DataTrimmer, create_RBF_interpolator, interpolate_coefficients, compute_aproximations
from plotter import Plotter
from subdomain import Subdomain
from linear_pde import Elasticity

from qugar import impl
from qugar.cpp import create_affine_transformation

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List

def _make_levelset_function(
    surface_name: str,
    surface_func_constructor,
    periods=(1, 1),
    z=0.0,
    negative=True,
):

    def levelset_func(parameters: list[int], p0, p1):
        origin = np.array(p0)
        axis_x = np.array([1.0, 0.0])
        scale_x = np.array(p1[0] - p0[0])
        scale_y = np.array(p1[1] - p0[1])  

        affine = create_affine_transformation(origin, axis_x, scale_x, scale_y)

        base_func = impl.create_affinely_transformed_functions(
            surface_func_constructor(periods=periods, z=z), affine
        )

        if negative:
            impl_func = impl.create_functions_subtraction(
                impl.create_dim_linear(parameters, affine_trans=affine),
                base_func
            )
        else:
            impl_func = impl.create_functions_subtraction(
                base_func,
                impl.create_dim_linear(parameters, affine_trans=affine)
            )

        return impl_func

    levelset_func.__name__ = f"levelset_{surface_name}"
    return levelset_func

@dataclass
class LevelsetConfig:
    name: str
    surface_func_constructor: Callable
    periods: List[int] = field(default_factory=lambda: [1, 1])
    z: float = 0.0
    negative: bool = False

configs = [
    LevelsetConfig(name="schwarz_diamond", surface_func_constructor=impl.create_Schwarz_Diamond),
    LevelsetConfig(name="fischer_koch_s", surface_func_constructor=impl.create_Fischer_Koch_S),
    LevelsetConfig(name="schoen", surface_func_constructor=impl.create_Schoen),
    LevelsetConfig(name="schwarz_primitive_1", surface_func_constructor=impl.create_Schwarz_Primitive, negative=True),
    LevelsetConfig(name="schwarz_primitive_2", surface_func_constructor=impl.create_Schwarz_Primitive, z=0.5),
    LevelsetConfig(name="schoen_FRD", surface_func_constructor=impl.create_Schoen_FRD, negative=True),
    LevelsetConfig(name="schoen_IWP", surface_func_constructor=impl.create_Schoen_IWP),
]

for config in configs:
    func = _make_levelset_function(
        surface_name=config.name,
        surface_func_constructor=config.surface_func_constructor,
        periods=config.periods,
        z=config.z,
        negative=config.negative,
    )
    globals()[f"{config.name}"] = func

class Plots(Plotter):

    @classmethod
    def plot_1(
        cls,
        size,
        data,
        path: str,
        dir: str,
        title: str,
        xlabel: str = "$x$",
        ylabel: str = "$y$",
        legend_loc = (0.5, -0.15),
        fa = [0.05, 0.95, 0.9, 0.05, 0.25, 0.5],
        nc = 3
    ) -> None:

        cls._folder = os.path.join(os.path.join(os.getcwd(), "figs"), dir)
        cls.__clear__()
        cls.__setup_config__()

        fig, ax = plt.subplots(figsize=size)

        fig.suptitle(title, fontsize=16)

        x = data["x"]
        y = data["y"]
        label = data["label"]

        ax.loglog(x, y, "-", markersize=4, label=label)


        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.grid()

        ax.legend(
            loc="upper center",
            bbox_to_anchor=legend_loc,
            ncol=nc,
            fontsize=12,
            edgecolor="black",  # Legend border color
            fancybox=False      # Rounded box edges
        )

        plt.savefig(cls.add_folder(path), bbox_inches="tight")

    @classmethod
    def plot_2(
        cls,
        size,
        data,
        path: str,
        dir: str,
        title: str,
        xlabel: str = "$x$",
        ylabel: str = "$y$",
        legend_loc = (0.5, -0.15),
        fa = [0.05, 0.95, 0.9, 0.05, 0.25, 0.5],
        nc = 3
    ) -> None:

        cls._folder = os.path.join(os.path.join(os.getcwd(), "figs"), dir)
        cls.__clear__()
        cls.__setup_config__()

        fig, ax = plt.subplots(figsize=size)

        fig.suptitle(title, fontsize=16)

        xs = data["x"]
        ys = data["y"]
        labels = data["label"]

        for x, y, label in zip(xs, ys, labels):
            ax.loglog(x, y, "-", markersize=4, label=label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.grid()

        ax.legend(
            loc="upper center",
            bbox_to_anchor=legend_loc,
            ncol=nc,
            fontsize=12,
            edgecolor="black",  # Legend border color
            fancybox=False      # Rounded box edges
        )

        plt.savefig(cls.add_folder(path), bbox_inches="tight")

def rbf_interpolator_error(coefficients, test_coefficients, parameters, test_parameters, basis, test_snapshots):

    interpolator = create_RBF_interpolator(parameters, coefficients.T)
    coefs_i = interpolate_coefficients(interpolator, parameters)

    test_coefs_i = interpolate_coefficients(interpolator, test_parameters)

    error = np.linalg.norm(coefficients.T-coefs_i, np.inf, axis = 0)
    error = np.average(error/np.linalg.norm(coefficients.T, np.inf, axis = 0))

    test_error = np.linalg.norm(test_coefficients.T-test_coefs_i, np.inf, axis = 0)
    test_error = np.average(test_error/np.linalg.norm(test_coefficients.T, np.inf, axis = 0))

    snapshots_c = compute_aproximations(basis, test_coefficients.T)
    snapshots_i = compute_aproximations(basis, test_coefs_i)

    snapshots_error = np.linalg.norm(test_snapshots-snapshots_c, np.inf, axis = 0)
    snapshots_error = np.average(snapshots_error/np.linalg.norm(test_snapshots, np.inf, axis = 0))

    snapshots_rbf_error = np.linalg.norm(test_snapshots-snapshots_i, np.inf, axis = 0)
    snapshots_rbf_error = np.average(snapshots_rbf_error/np.linalg.norm(test_snapshots, np.inf, axis = 0))

    return error, test_error, snapshots_error, snapshots_rbf_error


def bcast_array(array, comm):
    shape = comm.bcast(array.shape if rank == 0 else None, root=0)
    dtype = comm.bcast(str(array.dtype) if rank == 0 else None, root=0)
    if rank == 0:
        array = np.ascontiguousarray(array)
    else:
        array = np.empty(shape, dtype=dtype)
    comm.Bcast(array, root=0)
    return array

def generate_snapshots(parameters):

    n = [1, 1]
    degree = 8
    dim = 2

    xmin = np.array([0.0, 0.0])
    xmax = np.array([1.0, 1.0])

    elasticity_pde = Elasticity()

    snapshots = []

    for parameters_i in parameters:

        my_subdomain = Subdomain(
            n,
            degree,
            dim,
            xmin,
            xmax,
            parameters_i,
            schwarz_diamond,
            elasticity_pde,
            assemble=True
        )

        K = my_subdomain.K.toarray()
        snapshots.append(K)

    return snapshots

    
if __name__ == "__main__":      

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    epsilon_min = 0.1
    epsilon_max = 0.9

    total_points = 40
    batch_size = 10
    number_of_clusters = 1

    if rank == 0:
        sampler = sp.stats.qmc.LatinHypercube(d=4)
        parameters = epsilon_min + (epsilon_max - epsilon_min) * sampler.random(n=total_points)
    else:
        parameters = None

    parameters = comm.bcast(parameters, root=0)
    local_snapshots = np.array_split(np.arange(total_points), size)[rank]

    local_parameters = parameters[local_snapshots]

    snapshots = generate_snapshots(local_parameters)
    snapshots = comm.gather(snapshots, root=0)

    U = [None] * number_of_clusters
    I = [None] * number_of_clusters
    
    clusterer = KMeans(number_of_clusters)
    clusterer.fit(parameters)
    
    if rank == 0:

        snapshots = [K for sublist in snapshots for K in sublist]
        snapshots = np.array(snapshots) 
        snapshots = snapshots.reshape((snapshots.shape[0], snapshots.shape[1]*snapshots.shape[2])).T

        labels = clusterer.predict(parameters)
        clusters_deim = []

        for cluster in range(number_of_clusters):

            members = np.where(labels == cluster)[0].astype(int)

            cluster_snapshots = snapshots.T[members].T
            cluster_parameters = parameters[members]

            U[cluster], _, _ = compute_rSVD_basis(cluster_snapshots, k = 8, set_n = True, n = 4)
            I[cluster] = np.array(compute_magic_points(U[cluster]), dtype=int)


    for cluster in range(number_of_clusters):

        comm.barrier()
        U[cluster] = bcast_array(U[cluster], comm)
        I[cluster] = bcast_array(I[cluster], comm)
        I[cluster] = I[cluster].astype(int)

    ### TEST SAMPLES ###

    if rank == 0:

        test_points = 10

        test_parameters = epsilon_min + (epsilon_max - epsilon_min) * np.random.rand(test_points, 4)

        cluster_test_coefficients = [[] for _ in range(number_of_clusters)]
        cluster_test_parameters = [[] for _ in range(number_of_clusters)]
        cluster_test_snapshots = [[] for _ in range(number_of_clusters)]

        for batch_i in range(0, len(test_parameters), batch_size):

            batch_params = test_parameters[batch_i:batch_i + batch_size]

            snapshots = generate_snapshots(batch_params)
            snapshots = np.array(snapshots)

            S = snapshots.reshape((snapshots.shape[0], snapshots.shape[1]*snapshots.shape[2])).T

            labels = clusterer.predict(batch_params)

            for cluster in range(number_of_clusters):

                members = np.where(labels == cluster)[0].astype(int)
                cluster_S = S.T[members].T

                cluster_test_parameters[cluster].append(batch_params[members])
                cluster_test_coefficients[cluster].append(compute_deim_coefficients(U[cluster], I[cluster], cluster_S.T).T)
                cluster_test_snapshots[cluster].append(cluster_S)

        cluster_test_parameters = [np.vstack(p) if p else [] for p in cluster_test_parameters]
        cluster_test_coefficients = [np.vstack(c) if c else [] for c in cluster_test_coefficients]
        cluster_test_snapshots = [np.vstack(s) if s else [] for s in cluster_test_snapshots]

    ### RBF INTERPOLATOR ###

    rbf_n = [40, 80]

    rbf_errors = []
    rbf_test_errors = []

    rbf_snapshots_errors = []
    snapshots_errors = []

    for total_points in rbf_n:

        if rank == 0:
            sampler = sp.stats.qmc.LatinHypercube(d=4)
            parameters = epsilon_min + (epsilon_max - epsilon_min) * sampler.random(n=total_points)
        else:
            parameters = None

        parameters = comm.bcast(parameters, root=0)
        local_snapshots = np.array_split(np.arange(total_points), size)[rank]
        local_parameters = parameters[local_snapshots]

        cluster_coefficients = [[] for _ in range(number_of_clusters)]
        cluster_parameters = [[] for _ in range(number_of_clusters)]

        for batch_i in range(0, len(local_parameters), batch_size):

            batch_params = local_parameters[batch_i:batch_i + batch_size]

            snapshots = generate_snapshots(batch_params)
            snapshots = np.array(snapshots)

            S = snapshots.reshape((snapshots.shape[0], snapshots.shape[1]*snapshots.shape[2])).T

            labels = clusterer.predict(batch_params)

            for cluster in range(number_of_clusters):

                members = np.where(labels == cluster)[0].astype(int)
                cluster_S = S.T[members].T

                cluster_parameters[cluster].append(batch_params[members])
                cluster_coefficients[cluster].append(compute_deim_coefficients(U[cluster], I[cluster], cluster_S.T).T)

        cluster_parameters = [np.vstack(p) if p else [] for p in cluster_parameters]
        cluster_coefficients = [np.vstack(c) if c else [] for c in cluster_coefficients]

        all_parameters = comm.gather(cluster_parameters, root=0)
        all_coefficients = comm.gather(cluster_coefficients, root=0)

        if rank == 0:

            cluster_parameters = []
            cluster_coefficients = []

            cluster_error = []
            cluster_test_error = []

            cluster_snapshot_error = []
            cluster_rbf_snapshot_error = []

            for i in range(number_of_clusters):

                parameters = [p[i] for p in all_parameters]
                coefficients = [c[i] for c in all_coefficients]

                cluster_parameters.append(np.vstack(parameters))
                cluster_coefficients.append(np.vstack(coefficients))

                error, test_error, s_error, rbf_s_errors= rbf_interpolator_error(
                    cluster_coefficients[i], 
                    cluster_test_coefficients[i], 
                    cluster_parameters[i], 
                    cluster_test_parameters[i],
                    U[i],
                    cluster_test_snapshots[i]
                )

                cluster_error.append(error)
                cluster_test_error.append(test_error)

                cluster_snapshot_error.append(s_error)
                cluster_rbf_snapshot_error.append(rbf_s_errors)

            print("Cocolo colocao: ", total_points, "\n")

            rbf_errors.append(cluster_error)
            rbf_test_errors.append(cluster_test_error)

            snapshots_errors.append(cluster_snapshot_error)
            rbf_snapshots_errors.append(cluster_rbf_snapshot_error)

    
    if rank == 0:

        print(rbf_errors)
        print(rbf_test_errors)

        print(snapshots_errors)
        print(rbf_snapshots_errors)

        errors = [sum(e) / len(e) for e in rbf_test_errors]
        labels = f"$N_c = {number_of_clusters}$"

        data = {
            "x": rbf_n,
            "y": errors,
            "label": labels
        }

        Plots.plot_1(
            (5, 4),
            data,
            "and_what_do_i_do.pdf",
            "K",
            f"Interpolating coefficient errors for $K$ ($4$ parameters)",
            f"$N_s$",
            f"$\\frac{{\|\\theta-\hat{{\\theta}}\|}}{{\|\\theta\|}}$"
        )

        x = [rbf_n, rbf_n]
        y = [
            [sum(e) / len(e) for e in snapshots_errors],
            [sum(e) / len(e) for e in rbf_snapshots_errors]
        ]

        data = {
            "x": x,
            "y": y,
            "label": ["Computing", "Interpolating"]
        }

        Plots.plot_2(
            (5, 4),
            data,
            "so_be_it.pdf",
            "K",
            f"IErrors for $K$ ($4$ parameters)",
            f"$N_s$",
            f"$\\frac{{\|\\theta-\hat{{\\theta}}\|}}{{\|\\theta\|}}$"
        )

