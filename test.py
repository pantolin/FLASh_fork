import numpy as np
import scipy as sp

from mpi4py import MPI

import h5py
import os

from splines import BSpline2D
from spline_subdomain import Subdomain

from qugar import impl

def levelset(parameters: list[int]):

    impl_func = impl.create_functions_subtraction(
        impl.create_Schoen_IWP(periods=[1, 1], z=0.0),
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

    n_snapshots = 100
    n = [20, 20]
    degree = 3
    dim = 2

    epsilon_min = -1.0
    epsilon_max = 1.0

    sampler = sp.stats.qmc.LatinHypercube(d=4)
    parameters = epsilon_min + (epsilon_max - epsilon_min) * sampler.random(n=n_snapshots)
    local_snapshots = np.array_split(np.arange(n_snapshots), size)[rank]
    parameters = parameters[local_snapshots]
    name = f"data_rank_{rank}"

    h5f = h5py.File(os.path.join(folder, f"{name}.h5"), 'w')
    snapshots = []

    for parameters_i in parameters:

        my_subdomain = Subdomain(n, degree, dim, parameters_i, levelset)
        my_subdomain.assemble_K()

        snapshots.append(my_subdomain.K.toarray())

    snapshots = np.array(snapshots)
    h5f.create_dataset("K", data = snapshots)
    h5f.create_dataset("parameters", data = parameters)
    h5f.close()




