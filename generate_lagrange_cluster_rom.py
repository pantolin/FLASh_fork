import numpy as np
import scipy as sp

import h5py
import os

from mpi4py import MPI

from deim_int import compute_rSVD_basis, compute_magic_points, compute_deim_coefficients
from interpolator_2 import Interpolator
from subdomain import Subdomain
from linear_pde import Elasticity

from qugar import impl
from qugar.cpp import create_affine_transformation

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

    operator_name = "K"
    geometry_name = "schwarz_diamond"

    folder = os.path.join("rom_data", geometry_name, operator_name)
    os.makedirs(folder, exist_ok=True)

    epsilon_min = 0.1
    epsilon_max = 0.9

    total_points = 20
    batch_size = 100

    n = 2
    p = 2
    d = 4

    epsilon = np.linspace(epsilon_min, epsilon_max, n+1)

    for idx in product(range(n), repeat=d):

        index = np.array((idx[::-1]))

        id = 0
        count = 1
        for i in idx:
            id += i*count
            count *= n
            
        epsilon_0 = epsilon[index]
        epsilon_1 = epsilon[index+1]

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
            snapshots = snapshots.reshape((snapshots.shape[0], snapshots.shape[1]*snapshots.shape[2])).T

            U, _, _ = compute_rSVD_basis(snapshots, k = 10, set_n = True, n = 5)
            I = np.array(compute_magic_points(U), dtype=int)

        comm.barrier()
        U = bcast_array(U, comm)
        I = bcast_array(I, comm)
        I = I.astype(int)

        ### LAGRANGE INTERPOLATOR ###

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

            S = snapshots.reshape((snapshots.shape[0], snapshots.shape[1]*snapshots.shape[2])).T
            coefficients.append(compute_deim_coefficients(U, I, S.T).T)

        coefficients = np.vstack(coefficients)
        coefficients = comm.gather(coefficients, root=0)

        if rank == 0:

            coefficients = np.vstack(coefficients)

            file_path = os.path.join(folder, f"data_{id}.h5")

            with h5py.File(file_path, "w") as f:
                f.create_dataset("basis", data=U)
                f.create_dataset("weights", data=coefficients)

            print(f"Saved to {file_path}")


