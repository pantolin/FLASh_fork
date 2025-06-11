import numpy as np
import scipy as sp

from mpi4py import MPI

import h5py
import os

from subdomain import Subdomain
from linear_pde import Elasticity
from qugar import impl

def levelset(parameters: list[int]):

    impl_func = impl.create_functions_subtraction(
        impl.create_Schwarz_Diamond(periods=[1, 1], z=0.0),
        impl.create_dim_linear(parameters)
    )
    return impl_func


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

    xmin = np.array([0,1])
    xmax = np.array([0.1])

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
    snapshots_M = []
    snapshots_bM = []

    for parameters_i in parameters:

        my_subdomain = Subdomain(
            n,
            degree,
            dim,
            xmin,
            xmax,
            parameters,
            levelset,
            elasticity_pde,
            assemble=True
        )

        snapshots_K.append(my_subdomain.K.toarray())
        snapshots_M.append(my_subdomain.M.toarray())
        snapshots_bM.append(my_subdomain.bM.toarray())

    snapshots_K = np.array(snapshots_K)
    snapshots_M = np.array(snapshots_M)
    snapshots_bM = np.array(snapshots_bM)

    h5f.create_dataset("K", data = snapshots_K)
    h5f.create_dataset("M", data = snapshots_M)
    h5f.create_dataset("cM", data = snapshots_bM)
    h5f.create_dataset("parameters", data = parameters)
    h5f.close()
