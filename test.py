from splines import BSpline2D
from spline_subdomain import Subdomain

from qugar import impl

def levelset(parameters: list[int]):

    impl_func = impl.create_functions_subtraction(
        impl.create_Schoen_IWP(periods=[1, 1]),
        impl.create_dim_linear(parameters)
    )
    return impl_func

n = [3, 3]
degree = 2
dim = 2

my_spline = BSpline2D(n, degree)
my_subdomain = Subdomain(n, degree, dim, [-0.9, -0.9, 0.9, 0.9], levelset)
my_subdomain.assemble_K()
my_subdomain.pyvista_plot()
