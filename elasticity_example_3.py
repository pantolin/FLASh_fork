import numpy as np
import sympy as sy
 
from utils import Communicators

from qugar import impl

from qugar.cpp import create_affine_transformation

from bddc import bddc_solver
from linear_pde import Elasticity

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List

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


if __name__ == "__main__":            


    N = [5, 5]
    dim = 2

    P0 = [0.0, 0.0]
    P1 = [1.0, 1.0]

    epsilon_min = 0
    epsilon_max = 1

    n = [2, 2]
    degree = 7

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
        "compute_error" : True
    }

    bddc_solver(geometry, parameter_function, elasticity_pde, communicators, opts = opts)






