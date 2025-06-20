import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import h5py
import os
import pickle

from mpi4py import MPI
from sklearn.cluster import KMeans

from deim_int import compute_rSVD_basis, compute_magic_points, compute_deim_coefficients
from deim_int import DataTrimmer

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

def bcast_array(array, comm):
    shape = comm.bcast(array.shape if rank == 0 else None, root=0)
    dtype = comm.bcast(str(array.dtype) if rank == 0 else None, root=0)
    if rank == 0:
        array = np.ascontiguousarray(array)
    else:
        array = np.empty(shape, dtype=dtype)
    comm.Bcast(array, root=0)
    return array

if __name__ == "__main__":      

    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parameter_name = "parameters"
    operator_name = "K"
    operator_plot_name = "K"

    U = None
    I = None

    if rank == 0: 

        n_files = 10

        folder = os.path.join(os.getcwd(), "snapshot_data")
        
        parameters = []
        snapshots = []

        for data_rank in range(n_files):

            name = f"data_rank_{data_rank}"

            h5f = h5py.File(os.path.join(folder, f"{name}.h5"),'r')
            parameters.append(h5f[parameter_name][:])
            snapshots.append(h5f[operator_name][:])
            h5f.close()

        parameters = np.concatenate(parameters, axis = 0)
        snapshots = np.concatenate(snapshots, axis = 0)
        snapshots = snapshots.reshape((snapshots.shape[0], snapshots.shape[1]*snapshots.shape[2])).T

        U, _, _ = compute_rSVD_basis(snapshots, k = 25, set_n = True, n = 10)
        I = np.array(compute_magic_points(U), dtype=int)

    comm.barrier()
    U = bcast_array(U, comm)
    I = bcast_array(I, comm)
    I = I.astype(int)

    folder = os.path.join(os.getcwd(), "coefficient_data")
    
    if rank == 0:
        if not os.path.exists(folder):
            os.mkdir(folder)

    comm.Barrier()

    n_snapshots = 10
    n = [1, 1]
    degree = 8
    dim = 2

    xmin = np.array([0.0, 0.0])
    xmax = np.array([1.0, 1.0])

    epsilon_min = 0.1
    epsilon_max = 0.9

    elasticity_pde = Elasticity()

    total_points = 200
    batch_size = 10

    sampler = sp.stats.qmc.LatinHypercube(d=4)
    parameters = epsilon_min + (epsilon_max - epsilon_min) * sampler.random(n=total_points)
    local_snapshots = np.array_split(np.arange(total_points), size)[rank]
    parameters = parameters[local_snapshots]

    rank_folder = os.path.join(folder, f"rank_{rank}")
    os.makedirs(rank_folder, exist_ok=True)

    # Batch loop
    for batch_i in range(0, len(parameters), batch_size):
        batch_params = parameters[batch_i:batch_i + batch_size]

        snapshots_K = []

        for parameters_i in batch_params:

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
            snapshots_K.append(K)

        snapshots_K = np.array(snapshots_K)
        S = snapshots_K.reshape((snapshots_K.shape[0], snapshots_K.shape[1]*snapshots_K.shape[2])).T
        coefficients = compute_deim_coefficients(U, I, S.T)

        batch_file = os.path.join(rank_folder, f"batch_{batch_i//batch_size}.h5")
        with h5py.File(batch_file, 'w') as h5f:
            h5f.create_dataset("coefficients", data=coefficients)
            h5f.create_dataset("parameters", data=batch_params)


    folder = os.path.join(os.getcwd(), "test_coefficient_data")

    total_points = 20
    batch_size = 10

    sampler = sp.stats.qmc.LatinHypercube(d=4)
    parameters = epsilon_min + (epsilon_max - epsilon_min) * np.random.rand(total_points, 4)
    local_snapshots = np.array_split(np.arange(total_points), size)[rank]
    parameters = parameters[local_snapshots]

    rank_folder = os.path.join(folder, f"rank_{rank}")
    os.makedirs(rank_folder, exist_ok=True)

    # Batch loop
    for batch_i in range(0, len(parameters), batch_size):
        batch_params = parameters[batch_i:batch_i + batch_size]

        snapshots_K = []

        for parameters_i in batch_params:

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
            snapshots_K.append(K)

        snapshots_K = np.array(snapshots_K)
        S = snapshots_K.reshape((snapshots_K.shape[0], snapshots_K.shape[1]*snapshots_K.shape[2])).T
        coefficients = compute_deim_coefficients(U, I, S.T)

        batch_file = os.path.join(rank_folder, f"batch_{batch_i//batch_size}.h5")
        with h5py.File(batch_file, 'w') as h5f:
            h5f.create_dataset("coefficients", data=coefficients)
            h5f.create_dataset("parameters", data=batch_params)



    
