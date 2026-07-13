import numpy as np
import scipy as sp
from pathlib import Path

import h5py

from mpi4py import MPI

from FLASh.rom import (
    compute_rSVD_basis, 
    compute_magic_points, 
    compute_deim_coefficients,
    compute_aproximations,
    generate_snapshots
)

from FLASh.mesh import (
    gyroid,
)

from itertools import product
dtype = np.float64

# Paths
from _paths import ROM_DATA_DIR

def bcast_array(array, comm):
    shape = comm.bcast(array.shape if comm.Get_rank() == 0 else None, root=0)
    dtype = comm.bcast(str(array.dtype) if comm.Get_rank() == 0 else None, root=0)
    if comm.Get_rank() == 0:
        array = np.ascontiguousarray(array)
    else:
        array = np.empty(shape, dtype=dtype)
    comm.Bcast(array, root=0)
    return array

def compute_error(U, S, I):

    coefficients = compute_deim_coefficients(U, I, S.T).T
    approximations = compute_aproximations(U, coefficients.T)

    error = np.linalg.norm(S - approximations, axis=0, ord=np.inf) / np.linalg.norm(S, axis=0, ord=np.inf)
    return np.mean(error)

# Geometry cases: (geometry_name, levelset, epsilon_min, epsilon_max).
# schwarz_diamond_1 is the original case; schwarz_diamond / schoen_iwp are the
# two new geometries ported from rom_basis_test_bis. Their _3 / _4 suffixes encode
# the two calibrated parameter ranges of each geometry.
cases = [
    ("schwarz_diamond_3",   gyroid.SchwarzDiamond().make_function(),    0.1, 0.9),
    ("schwarz_diamond_4",   gyroid.SchwarzDiamond().make_function(),    0.1, 1.0),
    ("schoen_iwp_3",        gyroid.SchoenIWP().make_function(),        -2.5, 2.5),
    ("schoen_iwp_4",        gyroid.SchoenIWP().make_function(),        -2.5, 3.0),
    ("schoen_frd_3",        gyroid.SchoenFRD().make_function(),        -6.5, 0.5),
    ("schoen_frd_4",        gyroid.SchoenFRD().make_function(),        -7.0, 0.5),
    ("schwarz_primitive_3", gyroid.SchwarzPrimitive().make_function(), -0.5, 0.8),
    ("schwarz_primitive_4", gyroid.SchwarzPrimitive().make_function(), -0.5, 1.0),
]

if __name__ == "__main__":

    operator_name = "K_core"

    ns = [1, 2]
    d = 4

    samples_per_basis = 20
    batch_size = 100
    basis_size = 10
    basis_oversample = 5
    directory = str(ROM_DATA_DIR)

    test_samples = 10

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # The case loop must be identical on every rank so that all collective
    # calls (bcast, gather, barrier, allgather) stay in the same order.
    for geometry_name, levelset, epsilon_min, epsilon_max in cases:

        folder = Path(directory) / geometry_name / operator_name
        if rank == 0:
            folder.mkdir(parents=True, exist_ok=True)
        comm.barrier()

        total_points = samples_per_basis

        all_errors = [np.zeros(basis_size) for _ in range(len(ns))]

        for n, errors in zip(ns, all_errors):

            epsilon = np.linspace(epsilon_min, epsilon_max, n+1)

            for idx in product(range(n), repeat=d):

                index = np.array((idx[::-1]))

                id = 0
                count = 1
                for i in index:
                    id += i*count
                    count *= n

                epsilon_0 = epsilon[index]
                epsilon_1 = epsilon[index+1]

                print(f"[{geometry_name}] Interpolator id: {id}, limits: {epsilon_0}, {epsilon_1}.\n")

                #### GENERATE BASIS ####

                if rank == 0:
                    sampler = sp.stats.qmc.LatinHypercube(d=4)
                    parameters = epsilon_0 + (epsilon_1 - epsilon_0) * sampler.random(n=total_points)
                else:
                    parameters = None

                parameters = comm.bcast(parameters, root=0)
                local_snapshots = np.array_split(np.arange(total_points), size)[rank]

                local_parameters = parameters[local_snapshots]

                snapshots = generate_snapshots(
                    local_parameters,
                    levelset=levelset,
                    operator_name=operator_name
                )

                snapshots = comm.gather(snapshots, root=0)
                U = None
                I = None

                if rank == 0:
                    snapshots = [K for sublist in snapshots for K in sublist]
                    snapshots = np.array(snapshots)
                    snapshots = snapshots.reshape((snapshots.shape[0], -1)).T

                    U, _, _ = compute_rSVD_basis(snapshots, k = basis_size + basis_oversample, set_n = True, n = basis_size)
                    I = np.array(compute_magic_points(U), dtype=int)

                comm.barrier()
                U = bcast_array(U, comm)
                I = bcast_array(I, comm)
                I = I.astype(int)

                ### GENERATE TEST DATA ####

                if rank == 0:
                    sampler = sp.stats.qmc.LatinHypercube(d=4)
                    parameters = epsilon_0 + (epsilon_1 - epsilon_0) * sampler.random(n=test_samples)
                else:
                    parameters = None

                parameters = comm.bcast(parameters, root=0)
                local_snapshots = np.array_split(np.arange(test_samples), size)[rank]

                local_parameters = parameters[local_snapshots]

                snapshots = generate_snapshots(
                    local_parameters,
                    levelset=levelset,
                    operator_name=operator_name
                )

                snapshots = np.array(snapshots)

                S = snapshots.reshape((snapshots.shape[0], -1)).T

                for count, n_basis in enumerate(range(1, basis_size + 1)):

                    U_n = U[:, :n_basis]
                    I_n = I[:n_basis]

                    error = compute_error(U_n, S, I_n)
                    error = comm.allgather(error)
                    errors[count] += sum(error) / ((n**d)*size)


        if rank == 0:

            x = np.array([np.arange(1, basis_size + 1)] * len(ns))
            y = np.array(all_errors)

            file_path = folder / "error_data.h5"

            with h5py.File(file_path, "w") as f:
                f.create_dataset("basis_number", data=x)
                f.create_dataset("errors", data=y)

            print(f"Saved to {file_path}")

        comm.barrier()
