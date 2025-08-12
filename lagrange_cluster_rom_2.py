import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import h5py
import os
import pickle

from mpi4py import MPI
from sklearn.cluster import KMeans

from deim_int import compute_rSVD_basis, compute_magic_points, compute_deim_coefficients
from deim_int import DataTrimmer, create_RBF_interpolator, interpolate_coefficients, compute_aproximations
from interpolator_2 import Interpolator
from plotter import Plotter

from mapped_linear_pde import MappedElasticity
from geometry import SomeName

from qugar import impl
from qugar.cpp import create_affine_transformation
from qugar.mesh import create_unfitted_impl_Cartesian_mesh

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List

from itertools import product

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

def error(coefficients, test_coefficients, parameters, test_parameters, basis, test_snapshots, interpolator):

    coefs_i = interpolator.evaluate(parameters)
    test_coefs_i = interpolator.evaluate(test_parameters)

    error = np.linalg.norm(coefficients.T-coefs_i.T, np.inf, axis = 0)
    error = np.average(error/np.linalg.norm(coefficients.T, np.inf, axis = 0))

    test_error = np.linalg.norm(test_coefficients.T-test_coefs_i.T, np.inf, axis = 0)
    test_error = np.average(test_error/np.linalg.norm(test_coefficients.T, np.inf, axis = 0))

    snapshots_c = compute_aproximations(basis, test_coefficients.T)
    snapshots_i = compute_aproximations(basis, test_coefs_i.T)

    snapshots_comp_error = np.linalg.norm(test_snapshots-snapshots_c, np.inf, axis = 0)
    snapshots_comp_error = np.average(snapshots_comp_error/np.linalg.norm(test_snapshots, np.inf, axis = 0))

    snapshots_int_error = np.linalg.norm(test_snapshots-snapshots_i, np.inf, axis = 0)
    snapshots_int_error = np.average(snapshots_int_error/np.linalg.norm(test_snapshots, np.inf, axis = 0))

    return error, test_error, snapshots_comp_error, snapshots_int_error

def bcast_array(array, comm):
    shape = comm.bcast(array.shape if rank == 0 else None, root=0)
    dtype = comm.bcast(str(array.dtype) if rank == 0 else None, root=0)
    if rank == 0:
        array = np.ascontiguousarray(array)
    else:
        array = np.empty(shape, dtype=dtype)
    comm.Bcast(array, root=0)
    return array

def map(x, y):
    return np.stack([x, y + x, 0*x], axis=-1)

def generate_snapshots(parameters):

    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 1.0])

    comm = MPI.COMM_SELF

    n = [1, 1]
    degree = 8
    p = 2
    dim = 2

    mapped_elasticity_pde = MappedElasticity()
    basis = SomeName(p)

    snapshots = []

    for parameters_i in parameters:

        impl_func = schwarz_diamond(parameters_i, p0, p1)

        unf_mesh = create_unfitted_impl_Cartesian_mesh(
            comm, impl_func, n, p0, p1, exclude_empty_cells=False
        )

        K = mapped_elasticity_pde.assemble_stiffness_core(unf_mesh, basis, degree)
        snapshots.append(K)

    return snapshots


if __name__ == "__main__":      

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    epsilon_min = 0.1
    epsilon_max = 0.9

    total_points = 20

    n = 2

    epsilon = np.linspace(epsilon_min, epsilon_max, n+1)

    for i, j, k, l in product(range(2), repeat=4):
        
        index = np.array([i, j, k, l])

        epsilon_0 = epsilon[index]
        epsilon_1 = epsilon[index+1]

        aa = 1

        if rank == 0:
            sampler = sp.stats.qmc.LatinHypercube(d=4)
            parameters = epsilon_0 + (epsilon_1 - epsilon_0) * sampler.random(n=total_points)
        else:
            parameters = None

        parameters = comm.bcast(parameters, root=0)
        local_snapshots = np.array_split(np.arange(total_points), size)[rank]

        local_parameters = parameters[local_snapshots]

        snapshots = generate_snapshots(local_parameters)
        snapshots = comm.gather(snapshots, root=0)
        U = None
        I = None
        
        if rank == 0:
            snapshots = [K for sublist in snapshots for K in sublist]
            snapshots = np.array(snapshots) 
            snapshots = snapshots.reshape((snapshots.shape[0], -1)).T

            U, _, _ = compute_rSVD_basis(snapshots, k = 10, set_n = True, n = 5)
            I = np.array(compute_magic_points(U), dtype=int)

        comm.barrier()
        U = bcast_array(U, comm)
        I = bcast_array(I, comm)
        I = I.astype(int)

        ### TEST COEFFICIENTS ###

        if rank == 0:

            test_points = 5

            sampler = sp.stats.qmc.LatinHypercube(d=4)
            test_parameters = epsilon_0 + (epsilon_1 - epsilon_0) * sampler.random(n=test_points)

            snapshots = generate_snapshots(test_parameters)
            snapshots = np.array(snapshots)

            S = snapshots.reshape((snapshots.shape[0], -1)).T
            test_coefficients = compute_deim_coefficients(U, I, S.T).T
            test_snapshots = S

        ### LAGRANGE INTERPOLATOR ###

        p_list = [1, 2, 3]
        batch_size = 10

        coef_error = []
        coef_test_error = []

        snap_comp_error = []
        snap_int_error = []

        for p in p_list:

            interpolator = Interpolator(4, p, epsilon_0, epsilon_1)

            parameters = interpolator.get_nodes()
            total_points = parameters.shape[0]
            local_snapshots = np.array_split(np.arange(total_points), size)[rank]
            local_parameters = parameters[local_snapshots]

            coefficients = []
            for batch_i in range(0, len(local_parameters), batch_size):
                batch_params = local_parameters[batch_i:batch_i + batch_size]

                snapshots = generate_snapshots(batch_params)
                snapshots = np.array(snapshots)

                S = snapshots.reshape((snapshots.shape[0], -1)).T
                coefficients.append(compute_deim_coefficients(U, I, S.T).T)

            coefficients = np.vstack(coefficients)
            coefficients = comm.gather(coefficients, root=0)

            if rank == 0:

                coefficients = np.vstack(coefficients)
                interpolator._weights = coefficients
                c_e, c_t_e, s_c_e, s_i_e = error(coefficients, test_coefficients, parameters, test_parameters, U, test_snapshots, interpolator)

                coef_error.append(c_e)
                coef_test_error.append(c_t_e)

                snap_comp_error.append(s_c_e)
                snap_int_error.append(s_i_e)

        if rank == 0:

            print(coef_error)
            print(coef_test_error)

            print(snap_comp_error)
            print(snap_int_error)

            ind = i + j*n + k*n*n + l*n*n*n

            labels = [""]
            data = {
                "x": p_list,
                "y": coef_test_error,
                "label": labels
            }

            Plots.plot_1(
                (5, 4),
                data,
                f"mapped_lagrange_intepolation_error_{ind}.pdf",
                "K_test",
                f"Interpolating coefficient errors for $K$ ($4$ parameters)",
                f"$p$",
                f"$\\frac{{\|K-\hat{{K}}\|}}{{\|K\|}}$"
            )

            x = [p_list, p_list]
            y = [snap_comp_error, snap_int_error]

            data = {
                "x": x,
                "y": y,
                "label": ["Computing", "Interpolating"]
            }

            Plots.plot_2(
                (5, 4),
                data,
                f"mapped_lagrange_snapshot_error_{ind}.pdf",
                "K_test",
                f"Errors for $K$ ($4$ parameters)",
                f"$p$",
                f"$\\frac{{\|K-\hat{{K}}\|}}{{\|K\|}}$"
            )


