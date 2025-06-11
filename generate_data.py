import numpy as np
import scipy as sp

from mpi4py import MPI

import h5py
import os

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



if __name__ == "__main__":      

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    folder = os.path.join(os.getcwd(), "snapshot_data")
    
    if rank == 0:
        if not os.path.exists(folder):
            os.mkdir(folder)

    comm.Barrier()

    n_snapshots = 2500
    n = [1, 1]
    degree = 8
    dim = 2

    xmin = np.array([0,0])
    xmax = np.array([1,1])

    epsilon_min = 0.1
    epsilon_max = 0.9

    sampler = sp.stats.qmc.LatinHypercube(d=4)
    parameters = epsilon_min + (epsilon_max - epsilon_min) * sampler.random(n=n_snapshots)
    local_snapshots = np.array_split(np.arange(n_snapshots), size)[rank]
    parameters = parameters[local_snapshots]
    name = f"data_rank_{rank}"

    elasticity_pde = Elasticity()

    h5f = h5py.File(os.path.join(folder, f"{name}.h5"), 'w')
    snapshots_K = []
    snapshots_L = []
    snapshots_M = []
    snapshots_bM = []

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

        snapshots_K.append(my_subdomain.K.toarray())
        snapshots_L.append(scipy.linalg.cholesky(my_subdomain.K.toarray()))
        snapshots_M.append(my_subdomain.M.toarray())
        snapshots_bM.append(my_subdomain.bM.toarray())

    snapshots_K = np.array(snapshots_K)
    snapshots_L = np.array(snapshots_L)
    snapshots_M = np.array(snapshots_M)
    snapshots_bM = np.array(snapshots_bM)

    h5f.create_dataset("K", data = snapshots_K)
    h5f.create_dataset("L", data = snapshots_L)
    h5f.create_dataset("M", data = snapshots_M)
    h5f.create_dataset("cM", data = snapshots_bM)
    h5f.create_dataset("parameters", data = parameters)
    h5f.close()
