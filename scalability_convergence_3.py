import numpy as np
import matplotlib.pyplot as plt


import os
from utils import Communicators

from qugar import impl

from qugar.cpp import create_affine_transformation

from bddc import bddc_solver

dtype = np.float64

import os
from plotter import Plotter

from dataclasses import dataclass, field
from typing import Callable, List

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

        xs = data["x"]
        ys = data["y"]
        labels = data["label"]

        for x, y, label in zip(xs, ys, labels):
            ax.plot(x, y, "-^", markersize=4, label=label)

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

class Tracker:

    """Class for tracking the iterative solver residuals and interations.
    """

    def __init__(self):
        self.iter = []
        self.time = []

    def __call__(self, iter = None, time = None):
        self.iter.append(iter)
        self.time.append(time)

def source(X):
    return (0+0.0*X[0], 0+0*X[0])

def bc_1(X):
    return (0.0*X[0], 0*X[0])

def bc_2(X):
    return (0.2+0.0*X[0], 0+0*X[0])

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

    dim = 2

    P0 = [0.0, 0.0]
    P1 = [1.0, 0.5]

    N = [4, 2]

    epsilon_min = 0.1
    epsilon_max = 1
    levelset = schwarz_diamond

    
    communicators = Communicators()

    exterior_bc = [(0, bc_1, lambda x: np.isclose(x[0], P0[0]), 0),
                   (0, bc_2, lambda x: np.isclose(x[0], P1[0]), 1)]


    def parameter_function(X, eps_min=epsilon_min, eps_max=epsilon_max):
        num_points = X.shape[1]
        return eps_min + (eps_max - eps_min) * np.random.rand(num_points)
    
    stats_1 = []
    stats_2 = []
    stats_3 = []

    n_max = 7
    for i in range(2, n_max):

        n = [2*i, 2*i]

        degree = 1
        geometry = {
            "n": n,
            "N": N,
            "degree": degree,
            "dim": dim,
            "P0": P0,
            "P1": P1,
            "levelset": levelset,
        }

        stats_1.append(bddc_solver(geometry, parameter_function, exterior_bc, source, communicators))

        degree = 2
        geometry = {
            "n": n,
            "N": N,
            "degree": degree,
            "dim": dim,
            "P0": P0,
            "P1": P1,
            "levelset": levelset,
        }

        stats_2.append(bddc_solver(geometry, parameter_function, exterior_bc, source, communicators))

        degree = 3
        geometry = {
            "n": n,
            "N": N,
            "degree": degree,
            "dim": dim,
            "P0": P0,
            "P1": P1,
            "levelset": levelset,
        }

        stats_3.append(bddc_solver(geometry, parameter_function, exterior_bc, source, communicators))

    if communicators.global_comm.Get_rank() == 0:

        x = [np.arange(2,n_max)] * 3

        iterations = [
            [stat["iterations"] for stat in stats_1],
            [stat["iterations"] for stat in stats_2],
            [stat["iterations"] for stat in stats_3],
            [(1+np.log(2*np.arange(2,n_max))**2)],
        ]

        condition_numbers = [
            [stat["condition_number"] for stat in stats_1],
            [stat["condition_number"] for stat in stats_2],
            [stat["condition_number"] for stat in stats_3],
            [(1+np.log(2*np.arange(2,n_max))**2)],
        ]

        labels = [
            "p=1",
            "p=2",
            "p=3",
            "c*(1+log(H/h)^2)"
        ]

        data_i = {
            "x": x,
            "y": iterations,
            "label": labels
        }

        data_k = {
            "x": x,
            "y": condition_numbers,
            "label": labels
        }

        Plots.plot_1(
            (5, 4),
            data_i,
            "local_iterations_3.pdf",
            "scalability",
            "Local Iterations",
            f"$H/h$",
            f"I",
        )

        Plots.plot_1(
            (5, 4),
            data_k,
            "condition_number_3.pdf",
            "scalability",
            "Condition number",
            f"$H/h$",
            f"k",
        )






