from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

from qugar import impl
from qugar.cpp import create_affine_transformation

def _make_levelset_function(
    surface_name: str,
    surface_func_constructor,
    periods=np.array([1, 1]),
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

class Levelset(ABC):

    def __init__(self, periods=np.array([1, 1]), z=0.0, negative=False):
        self.periods = periods
        self.z = z
        self.negative = negative

    @abstractmethod
    def surface_func_constructor(self) -> Callable:
        pass

    def make_function(self):
        return _make_levelset_function(
            surface_name=self.__class__.__name__.lower(),
            surface_func_constructor=self.surface_func_constructor(),
            periods=self.periods,
            z=self.z,
            negative=self.negative,
        )
    
class SchwarzDiamond(Levelset):

    def surface_func_constructor(self):
        return impl.create_Schwarz_Diamond

class FischerKochS(Levelset):

    def surface_func_constructor(self):
        return impl.create_Fischer_Koch_S

class Schoen(Levelset):

    def surface_func_constructor(self):
        return impl.create_Schoen

class SchwarzPrimitive(Levelset):

    def __init__(self, **kwards):
        super().__init__(**kwards)

    def surface_func_constructor(self):
        return impl.create_Schwarz_Primitive

class SchoenFRD(Levelset):

    def __init__(self):
        super().__init__(negative=True)

    def surface_func_constructor(self):
        return impl.create_Schoen_FRD

class SchoenIWP(Levelset):

    def surface_func_constructor(self):
        return impl.create_Schoen_IWP
