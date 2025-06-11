import numpy as np
from utils import Communicators

from qugar import impl

from qugar.cpp import create_affine_transformation

from bddc import bddc_solver

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List

dtype = np.float64

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


    N = [4, 2]
    H = 1
    dim = 2

    P0 = [0.0, 0.0]
    P1 = [1.0, 0.5]

    epsilon_min = 0.1
    epsilon_max = 0.9

    n = [1, 1]
    degree = 11

    geometry = {
        "n": n,
        "N": N,
        "degree": degree,
        "dim": dim,
        "P0": P0,
        'P1': P1,
        "levelset": schwarz_diamond
    }
    
    communicators = Communicators()

    # def parameter_function(X):
    #     return 0.5*(epsilon_min*(P1[0]-X[0]) + epsilon_max*(X[0]-P0[0])) / (P1[0]-P0[0]) + 0.5*(epsilon_max*(P1[1]-X[1]) + epsilon_min*(X[1]-P0[1])) / (P1[1]-P0[1])
    
    # def parameter_function(X):
    #     return (epsilon_min*(P1[0]-X[0]) + epsilon_max*(X[0]-P0[0])) / (P1[0]-P0[0])

    def parameter_function(X):
        num_points = X.shape[1]
        random_vals = epsilon_min + (epsilon_max - epsilon_min) * np.random.rand(num_points)
        return random_vals

    exterior_bc = [(0, bc_1, lambda x: np.isclose(x[0], P0[0]), 0),
                   (0, bc_2, lambda x: np.isclose(x[0], P1[0]), 1)]

    bddc_solver(geometry, parameter_function, exterior_bc, exterior_bc, source, communicators)
    # gbl_dofs_mngr = GlobalDofsManager.plot_domain(geometry, parameter_function, source, communicators, exterior_bc)






