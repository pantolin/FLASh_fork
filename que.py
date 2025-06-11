import numpy as np
import sympy as sy

from mpi4py import MPI
from subdomain import Subdomain

from qugar import impl

import numpy as np
import scipy.sparse

from typing import Callable

from mpi4py import MPI
from qugar.cpp import create_affine_transformation

from linear_pde import Elasticity

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List

type SparseMatrix = scipy.sparse._csr.csr_matrix

dtype = np.float64

def source(X):
    return (0+0*X[0], 0*X[0])

def bc_1(X):
    return (0+0*X[0], 0 + 0*X[0])

def bc_2(X):
    return (0+0*X[0], -0.01+0*X[0])

def fun(x):
    return (np.cos(x[0]), x[1])

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


x, y = sy.symbols('x y')

u1 = 0.1 * sy.sin(sy.pi * (x))**2 * sy.sin(sy.pi * (y))**2 * (-1+2*y)
u2 = 0.1 * sy.sin(sy.pi * (x))**2 * sy.sin(sy.pi * (y))**2 * (1-2*x)

xmin = np.array([0.0, 0.0])
xmax = np.array([1.0, 1.0])

n = [10, 10]
degree = 3
dim = 2
parameters = [0.5] * 4

comm = MPI.COMM_SELF

exterior_bc = [(1, bc_2, lambda x: np.isclose(x[0], 1), 0)]
interior_bc = [(1, bc_1, lambda x: True, 0)]
u = []#[u1, u2]

elasticity_pde = Elasticity(
    interior_bc = interior_bc,
    exterior_bc = exterior_bc,
    source = source,
    u = u
)

my_subdomain = Subdomain(
    n,
    degree,
    dim,
    xmin,
    xmax,
    parameters,
    schoen_IWP,
    elasticity_pde,
    assemble=True
)

# my_subdomain.plot_solution(my_subdomain.greville_interpolation(fun))

# my_subdomain.pyvista_plot()


# a_dofs = my_subdomain.get_all_dofs()
# b_dofs = my_subdomain.get_edges_dofs()[1]
# i_dofs = np.setdiff1d(a_dofs, b_dofs)

# K = my_subdomain.pK
# f = my_subdomain.pf

# total_dofs = my_subdomain.get_all_dofs(get_active = False).size

# u = np.zeros((total_dofs,))
# u[i_dofs] = scipy.sparse.linalg.spsolve(K[i_dofs][:,i_dofs], f[i_dofs])

# my_subdomain.plot_solution(u)
# print(my_subdomain.compute_error(u, elasticity_problem.u_callable)/my_subdomain.compute_error(np.zeros((total_dofs,)), elasticity_problem.u_callable))




