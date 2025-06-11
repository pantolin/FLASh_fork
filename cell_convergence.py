import numpy as np
import sympy as sy
import matplotlib.pyplot as plt

from mpi4py import MPI

import os

from subdomain import Subdomain
from plotter import Plotter

from qugar import impl
from linear_pde import Elasticity
import scipy.sparse

from typing import Callable
from mpi4py import MPI

from qugar.cpp import create_affine_transformation

from dataclasses import dataclass, field
from typing import Callable, List

type SparseMatrix = scipy.sparse._csr.csr_matrix

dtype = np.float64

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

        colors = ["blue", "red", "orange", "green", "yellow", "purple"]

        x = data["x"]
        ys = data["y"]
        labels = data["labels"]

        for color, y, label in zip(colors, ys, labels):
            ax.loglog(x, y, '--+', color=color, markersize=4, label=label)

        ax.loglog(x, (x**3), '--', color='black', linewidth=1, markersize=4, label="c*h^3")
        ax.loglog(x, (x**4), '--^', color='black', linewidth=1, markersize=4, label="c*h^4")
        ax.loglog(x, (x**5), '--x', color='black', linewidth=1, markersize=4, label="c*h^5")

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

def solve_problem(n = [10, 10], degree = 2):

    x, y = sy.symbols('x y')    

    u1 = 0.1 * sy.sin(sy.pi * (x))**2 * sy.sin(sy.pi * (y))**2 * (-1+2*y)
    u2 = 0.1 * sy.sin(sy.pi * (x))**2 * sy.sin(sy.pi * (y))**2 * (1-2*x)

    xmin = np.array([0.0, 0.0])
    xmax = np.array([1.0, 1.0])

    dim = 2
    parameters = [0.5] * 4

    comm = MPI.COMM_SELF

    elasticity_pde = Elasticity(u=[u1,u2])

    my_subdomain = Subdomain(
        n,
        degree,
        dim,
        xmin,
        xmax,
        parameters,
        schwarz_diamond,
        elasticity_pde,
        assemble=True
    )

    i_dofs = my_subdomain.get_interior_dofs()

    K = my_subdomain.pK
    f = my_subdomain.pf

    total_dofs = my_subdomain.get_all_dofs(get_active = False).size

    u = np.zeros((total_dofs,))
    u[i_dofs] = scipy.sparse.linalg.spsolve(K[i_dofs][:,i_dofs], f[i_dofs])

    return my_subdomain.compute_error(u, elasticity_problem.u_callable)/my_subdomain.compute_error(np.zeros((total_dofs,)), elasticity_problem.u_callable)

if __name__ == "__main__":   

    error_2 = []
    error_3 = []
    error_4 = []

    n_max = 10

    for i in range(2,n_max):

        # Number of elements per direction in mesh
        n = [2*i,2*i]

        error_2.append(solve_problem(n, 2))
        error_3.append(solve_problem(n, 3))
        error_4.append(solve_problem(n, 4))

    error_10 = [solve_problem([1, 1], 10)] * len(error_2)
    error_12 = [solve_problem([1, 1], 12)] * len(error_2)
    error_15 = [solve_problem([1, 1], 15)] * len(error_2)

    x = 1/(2*np.arange(2,n_max))
    y = [
        error_2,
        error_3,
        error_4,
        error_10,
        error_12,
        error_15
    ]

    labels = [
        "p=2",
        "p=3",
        "p=4",
        "p=10",
        "p=12",
        "p=15"
    ]

    data = {
        "x": x,
        "y": y,
        "labels": labels
    }

    Plots.plot_1(
        (10, 4),
        data,
        "error.pdf",
        "cell_convergence_results",
        f"Error",
        f"$h$",
        f"$\|u-u_e\|$"
    )         


