import numpy as np
import scipy as sp
import sympy as sy
import matplotlib.pyplot as plt

import os 

from utils import Communicators
from linear_pde import Elasticity

from qugar import impl
from qugar.cpp import create_affine_transformation

from bddc import bddc_solver
from plotter import Plotter

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List

dtype = np.float64

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
        nc = 2
    ) -> None:

        cls._folder = os.path.join(os.path.join(os.getcwd(), "figs"), dir)
        cls.__clear__()
        cls.__setup_config__()

        fig, ax = plt.subplots(figsize=size)
        fig.suptitle(title, fontsize=16)

        colors1 = ["blue", "red", "orange"]
        colors2 = ["green", "yellow", "purple"]

        x = data["x"]

        ys1 = data["y1"]
        labels1 = data["labels1"]


        for color, y, label in zip(colors1, ys1, labels1):
            ax.loglog(x, y, '-^', color=color, markersize=4, label=label)

        ys2 = data["y2"]
        labels2 = data["labels2"]

        for color, y, label in zip(colors2, ys2, labels2):
            ax.loglog(x, y, '-', color=color, markersize=4, label=label)

        ax.loglog(x, (x**2), '--x', color='black', linewidth=1, markersize=4, label="c*h^2")
        ax.loglog(x, (x**3), '--', color='black', linewidth=1, markersize=4, label="c*h^3")
        ax.loglog(x, (x**4), '--^', color='black', linewidth=1, markersize=4, label="c*h^4")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
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
        nc = 2
    ) -> None:

        cls._folder = os.path.join(os.path.join(os.getcwd(), "figs"), dir)
        cls.__clear__()
        cls.__setup_config__()

        fig, ax = plt.subplots(figsize=size)
        fig.suptitle(title, fontsize=16)

        colors1 = ["blue", "red", "orange"]
        colors2 = ["green", "yellow", "purple"]

        x = data["x"]

        tts1 = data["tt1"]
        sts1 = data["st1"]
        labels1 = data["labels1"]


        for color, tt, st, label in zip(colors1, tts1, sts1, labels1):
            ax.loglog(x, st, '-^', color=color, markersize=4, label=f"Solve time, {label}")
            ax.loglog(x, tt, '--^', color=color, markersize=4, label=f"Total time, {label}")

        tts2 = data["tt2"]
        sts2 = data["st2"]
        labels2 = data["labels2"]

        for color, tt, st, label in zip(colors2, tts2, sts2, labels2):
            ax.loglog(x, st, '-', color=color, markersize=4, label=f"Solve time, {label}")
            ax.loglog(x, tt, '--', color=color, markersize=4, label=f"Total time, {label}")


        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
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

    @classmethod
    def plot_3(
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
        nc = 2
    ) -> None:

        cls._folder = os.path.join(os.path.join(os.getcwd(), "figs"), dir)
        cls.__clear__()
        cls.__setup_config__()

        fig, ax = plt.subplots(figsize=size)
        fig.suptitle(title, fontsize=16)

        colors1 = ["blue", "red", "orange"]
        colors2 = ["green", "yellow", "purple"]

        es1 = data["e1"]
        tts1 = data["tt1"]
        sts1 = data["st1"]
        labels1 = data["labels1"]


        for color, e, tt, st, label in zip(colors1, es1, tts1, sts1, labels1):
            ax.loglog(e, st, '-^', color=color, markersize=4, label=f"Solve time, {label}")
            ax.loglog(e, tt, '--^', color=color, markersize=4, label=f"Total time, {label}")

        es2 = data["e2"]
        tts2 = data["tt2"]
        sts2 = data["st2"]
        labels2 = data["labels2"]

        for color, e, tt, st, label in zip(colors2, es2, tts2, sts2, labels2):
            ax.loglog(e, st, 'o', color=color, markersize=4, label=f"Solve time, {label}")
            ax.loglog(e, tt, '*', color=color, markersize=4, label=f"Total time, {label}")


        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
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

    N = [2, 2]
    H = 1
    dim = 2

    P0 = [0.0, 0.0]
    P1 = [1.0, 1.0]

    epsilon_min = 0
    epsilon_max = 1
    
    communicators = Communicators()

    x, y = sy.symbols('x y')

    u1 = 0.1 * sy.sin(sy.pi * (x))**2 * sy.sin(sy.pi * (y))**2 * (-1+2*y)
    u2 = 0.1 * sy.sin(sy.pi * (x))**2 * sy.sin(sy.pi * (y))**2 * (1-2*x)

    def bc(x):
        return (0*x[0], 0*x[0])
    
    exterior_bc = [(0, bc, lambda x: np.isclose(x[0], P0[0]), 0),
                   (0, bc, lambda x: np.isclose(x[0], P1[0]), 0),
                   (0, bc, lambda x: np.isclose(x[1], P0[1]), 0),
                   (0, bc, lambda x: np.isclose(x[1], P1[1]), 0)]
    
    elasticity_pde = Elasticity(
        exterior_bc = exterior_bc,
        u = [u1, u2]
    )

    def parameter_function(X):
        num_points = X.shape[1]
        random_vals = epsilon_min + (epsilon_max - epsilon_min) * np.random.rand(num_points)
        return random_vals

    opts = {
        "return_stats" : True,
        "compute_error" : True
    }

    stats_1 = []
    stats_2 = []
    stats_3 = []

    n_max = 12

    for i in range(2,n_max):

        n = [2*i, 2*i]
        degree = 1

        geometry = {
            "n": n,
            "N": N,
            "degree": degree,
            "dim": dim,
            "P0": P0,
            'P1': P1,
            "levelset": schwarz_diamond
        }

        stats_1.append(bddc_solver(geometry, parameter_function, elasticity_pde, communicators, opts = opts))

        degree = 2

        geometry = {
            "n": n,
            "N": N,
            "degree": degree,
            "dim": dim,
            "P0": P0,
            'P1': P1,
            "levelset": schwarz_diamond
        }

        stats_2.append(bddc_solver(geometry, parameter_function, elasticity_pde, communicators, opts = opts))

        degree = 3

        geometry = {
            "n": n,
            "N": N,
            "degree": degree,
            "dim": dim,
            "P0": P0,
            'P1': P1,
            "levelset": schwarz_diamond
        }

        stats_3.append(bddc_solver(geometry, parameter_function, elasticity_pde, communicators, opts = opts))


    n = [1, 1]
    degree = 6

    geometry = {
        "n": n,
        "N": N,
        "degree": degree,
        "dim": dim,
        "P0": P0,
        'P1': P1,
        "levelset": schwarz_diamond
    }

    stats_6 = bddc_solver(geometry, parameter_function, elasticity_pde, communicators, opts = opts)

    degree = 8

    geometry = {
        "n": n,
        "N": N,
        "degree": degree,
        "dim": dim,
        "P0": P0,
        'P1': P1,
        "levelset": schwarz_diamond
    }

    stats_8 = bddc_solver(geometry, parameter_function, elasticity_pde, communicators, opts = opts)

    degree = 10

    geometry = {
        "n": n,
        "N": N,
        "degree": degree,
        "dim": dim,
        "P0": P0,
        'P1': P1,
        "levelset": schwarz_diamond
    }

    stats_10 = bddc_solver(geometry, parameter_function, elasticity_pde, communicators, opts = opts)

    if communicators.global_comm.Get_rank() == 0:

        x = 1/(2*np.arange(2,n_max))

        e1 = [
            [stats["error"] for stats in stats_1],
            [stats["error"] for stats in stats_2],
            [stats["error"] for stats in stats_3]
        ]
        e2 = [
            [stats_6["error"]] * x.size,
            [stats_8["error"]] * x.size,
            [stats_10["error"]] * x.size
        ]

        tt1 = [
            [stats["total time"] for stats in stats_1],
            [stats["total time"] for stats in stats_2],
            [stats["total time"] for stats in stats_3]
        ]
        tt2 = [
            [stats_6["total time"]] * x.size,
            [stats_8["total time"]] * x.size,
            [stats_10["total time"]] * x.size
        ]

        st1 = [
            [stats["solve time"] for stats in stats_1],
            [stats["solve time"] for stats in stats_2],
            [stats["solve time"] for stats in stats_3]
        ]
        st2 = [
            [stats_6["solve time"]] * x.size,
            [stats_8["solve time"]] * x.size,
            [stats_10["solve time"]] * x.size
        ]


        labels1 = [
            "p=2",
            "p=3",
            "p=4"
        ]
        labels2 = [
            "p=6 (1 element)",
            "p=8 (1 element)",
            "p=10 (1 element)"
        ]

        data = {
            "x": x,
            "y1": e1,
            "y2": e2,
            "labels1": labels1,
            "labels2": labels2,
        }

        Plots.plot_1(
            (10, 4),
            data,
            "error.pdf",
            "convergence_results",
            f"Error",
            f"$h$",
            f"$\|u-u_e\|$"
        )    

        data = {
            "x": x,
            "tt1": tt1,
            "st1": st1,
            "tt2": tt2,
            "st2": st2,
            "labels1": labels1,
            "labels2": labels2,
        }   
        
        Plots.plot_2(
            (10, 4),
            data,
            "solve_times.pdf",
            "convergence_results",
            f"Solve time",
            f"$h$",
            f"$t(s)$"
        )    

        data = {
            "e1": e1,
            "tt1": tt1,
            "st1": st1,
            "e2": e2,
            "tt2": tt2,
            "st2": st2,
            "labels1": labels1,
            "labels2": labels2,
        }   
        
        Plots.plot_3(
            (10, 4),
            data,
            "time_error.pdf",
            "convergence_results",
            f"Solve time vs error",
            f"$t(s)$",
            f"$\|u-u_e\|$"
        )    







