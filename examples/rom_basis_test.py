import numpy as np
import scipy as sp

import os

from mpi4py import MPI
from matplotlib import pyplot as plt

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

from FLASh.utils import Plotter

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

def compute_error(U, S, I):

    coefficients = compute_deim_coefficients(U, I, S.T).T
    approximations = compute_aproximations(U, coefficients.T)

    error = np.linalg.norm(S - approximations, axis=0, ord=np.inf) / np.linalg.norm(S, axis=0, ord=np.inf)
    return np.mean(error)

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
        ylabel_left: str = "$y_1$",
        legend_loc=(0.5, -0.15),
        nc=3
    ) -> None:

        cls._folder = os.path.join(os.path.join(os.getcwd(), "figs"), dir)
        cls.__clear__()
        cls.__setup_config__()

        fig, ax = plt.subplots(figsize=size)

        fig.suptitle(title, fontsize=16)


        xs = data.get("x", [])
        ys = data.get("y", [])
        labels = data.get("labels", [])

        colors = ["blue", "red", "orange", "green"]

        for x, y, label, color in zip(xs, ys, labels, colors):
            ax.plot(x, y, "-^", markersize=4, color = color, label=label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel_left)

        ax.grid()

        ax.legend(
            loc="upper center",
            bbox_to_anchor=legend_loc,
            ncol=nc,
            fontsize=12,
            edgecolor="black",
            fancybox=False
        )

        plt.savefig(cls.add_folder(path), bbox_inches="tight")


schwarz_diamond = gyroid.SchwarzDiamond().make_function()

if __name__ == "__main__":
    
    operator_name = "K_core"
    geometry_name = "schwarz_diamond_1"
    levelset = schwarz_diamond
    epsilon_min = 0.1
    epsilon_max = 0.9

    ns = [1, 2]
    d = 4

    samples_per_basis = 20
    batch_size = 100
    basis_size = 10
    basis_oversample = 5
    directory = "rom_data"

    test_samples = 10

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    folder = os.path.join(directory, geometry_name, operator_name)
    os.makedirs(folder, exist_ok=True)

    total_points = samples_per_basis

    all_errors = [np.zeros(basis_size)] * len(ns)

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

            print(f"Interpolator id: {id}, limits: {epsilon_0}, {epsilon_1}.\n")

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

        x = [np.arange(1, basis_size + 1)] * len(ns)
        y = all_errors
        
        labels = ["$N_c=1$", "$N_c=16$"]

        data = {
            "x": x,
            "y": y,
            "label": labels
        }

        Plots.plot_1(
            (5, 4),
            data,
            "basis_error.pdf",
            "rom",
            "Snapshot Error vs. Basis Size",
            f"$N_b$",
            f"$\|A-\hat{{A}}\|$"
        )
        