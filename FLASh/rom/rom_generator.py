import numpy as np
import scipy as sp

import h5py
import os

from mpi4py import MPI

from FLASh.rom import (
    compute_rSVD_basis, 
    compute_magic_points, 
    compute_deim_coefficients,
    Interpolator
)

from FLASh.mesh import (
    SomeName,
    gyroid,
    Lagrange2D
)

from FLASh.pde import Elasticity

import qugar


from itertools import product
dtype = np.float64

def bcast_array(array, comm):
    shape = comm.bcast(array.shape if comm.Get_rank() == 0 else None, root=0)
    dtype = comm.bcast(str(array.dtype) if comm.Get_rank() == 0 else None, root=0)
    if comm.Get_rank() == 0:
        array = np.ascontiguousarray(array)
    else:
        array = np.empty(shape, dtype=dtype)
    comm.Bcast(array, root=0)
    return array

schwarz_diamond = gyroid.SchwarzDiamond().make_function()

def generate_snapshots(parameters, levelset = schwarz_diamond, degree = 8, p = 2, operator_name = "K_core", get_full_K_core = False):

    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 1.0])

    comm = MPI.COMM_SELF

    n = [1, 1]

    elasticity_pde = Elasticity()
    basis = Lagrange2D(degree, np.zeros(2), np.ones(2)) 
    element = SomeName(p)

    snapshots = []

    if get_full_K_core:

        impl_func = levelset(parameters, p0, p1).cpp_object

        cell_breaks = [np.linspace(0.0, 1.0, n[dir] + 1, dtype=dtype) for dir in range(2)]

        grid = qugar.cpp.create_cart_grid(cell_breaks)
        unf_mesh = qugar.cpp.create_unfitted_impl_domain(impl_func, grid)

        return elasticity_pde.assemble_stiffness_core(unf_mesh, basis, element, full_cell=True)

    for parameters_i in parameters:

        impl_func = levelset(parameters_i, p0, p1).cpp_object

        cell_breaks = [np.linspace(0.0, 1.0, n[dir] + 1, dtype=dtype) for dir in range(2)]

        grid = qugar.cpp.create_cart_grid(cell_breaks)
        unf_mesh = qugar.cpp.create_unfitted_impl_domain(impl_func, grid)

        if operator_name == "K_core":
            A = elasticity_pde.assemble_stiffness_core(unf_mesh, basis, element)
        elif operator_name == "M_core":
            A = elasticity_pde.assemble_mass_core(unf_mesh, basis, element)
        elif operator_name == "bM_core":
            A = elasticity_pde.assemble_boundary_mass_core(unf_mesh, basis, element)

        snapshots.append(A)

    return snapshots

def generate_rom_model(
        operator_name, 
        geometry_name, 
        levelset = schwarz_diamond, 
        epsilon_0 = 0.1, 
        epsilon_1 = 0.9, 
        n = 2, 
        p = 2, 
        d = 4, 
        samples_per_basis = 20,
        batch_size = 100,
        basis_size = 10,
        basis_oversample = 5,
        directory = "rom_data"
    ):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    folder = os.path.join(directory, geometry_name, operator_name)
    os.makedirs(folder, exist_ok=True)

    total_points = samples_per_basis

    epsilon = np.linspace(epsilon_0, epsilon_1, n+1)

    for idx in product(range(n), repeat=d):

        index = np.array((idx[::-1]))

        id = 0
        count = 1
        for i in index:
            id += i*count
            count *= n

        epsilon_0 = epsilon[index]
        epsilon_1 = epsilon[index+1]

        print(f"Interpolator id: {id}, limits: {epsilon_0}, {epsilon_1}.\n")

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

        ### LAGRANGE INTERPOLATOR ###

        interpolator = Interpolator(4, p, epsilon_0, epsilon_1)

        parameters = interpolator.get_nodes()
        total_points = parameters.shape[0]
        local_snapshots = np.array_split(np.arange(total_points), size)[rank]
        local_parameters = parameters[local_snapshots]

        coefficients = []
        for batch_i in range(0, len(local_parameters), batch_size):
            batch_params = local_parameters[batch_i:batch_i + batch_size]

            snapshots = generate_snapshots(
                local_parameters,
                levelset=levelset,
                operator_name=operator_name
            )

            snapshots = np.array(snapshots)

            S = snapshots.reshape((snapshots.shape[0], -1)).T
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
