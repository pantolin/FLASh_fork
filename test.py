import numpy as np

from mpi4py import MPI

from qugar import impl

import numpy as np
import scipy.sparse

from subdomain import Subdomain
import scipy.sparse.linalg

from mpi4py import MPI
from qugar.cpp import create_affine_transformation


type SparseMatrix = scipy.sparse._csr.csr_matrix

dtype = np.float64

def source(X):
    return (0*X[0], 0*X[0])

def bc(X):
    return (0.2+0*X[0], 0.0+0*X[0])

def levelset(parameters: list[int], p0, p1):

    origin = np.array(p0)
    axis_x = np.array([1.0, 0.0])
    scale_x = np.array(p1[0]-p0[0])
    scale_y = np.array(p1[0]-p0[0])

    affine = create_affine_transformation(origin, axis_x, scale_x, scale_y)

    impl_func = impl.create_functions_subtraction(
        impl.create_Schwarz_Diamond(periods=[1, 1], z=0.0),
        impl.create_dim_linear(parameters, affine_trans=affine)
    )

    return impl_func


xmin = np.array([0.0, 0.0])
xmax = np.array([1.0, 1.0])

n = [5, 5]
degree = 3
dim = 2

comm = MPI.COMM_SELF

my_subdomain_2 = Subdomain(
    n,
    degree,
    dim,
    xmin,
    xmax,
    [0.1, 0.1, 0.1, 0.1],
    levelset,
    source,
    assemble=True
)

my_subdomain_2.pyvista_plot()


v_dofs = my_subdomain_2.get_vertices_dofs()
a_v_dofs = my_subdomain_2.get_active_vertices_dofs()

e_dofs = my_subdomain_2.get_edge_average_dofs()
a_e_dofs = my_subdomain_2.get_active_edge_average_dofs()

d_dofs = my_subdomain_2.get_dual_dofs()
a_d_dofs = my_subdomain_2.get_active_dual_dofs()

i_dofs = my_subdomain_2.get_interior_dofs()
a_i_dofs = my_subdomain_2.get_active_interior_dofs()

b_dofs = my_subdomain_2.get_boundary_dofs()
a_b_dofs = my_subdomain_2.get_active_boundary_dofs()

bc_dofs = np.hstack([
    a_v_dofs[1],
    a_v_dofs[3],
    a_e_dofs[2],
    a_d_dofs[2]
])

i_dofs = np.hstack([
    a_i_dofs
])

K, f = my_subdomain_2.pK, my_subdomain_2.pf

# for dof in a_e_dofs[0]:
#     ud = np.zeros(my_subdomain_2.get_all_dofs().size)
#     ud[dof] = 1
#     my_subdomain_2.plot_solution(ud)

# for dof in a_e_dofs[3]:
#     ud = np.zeros(my_subdomain_2.get_all_dofs().size)
#     ud[dof] = 1
#     my_subdomain_2.plot_solution(ud)

# # ud = np.zeros(my_subdomain_2.get_all_dofs().size)
# # ud[:] = my_subdomain_2.get_projected_function(bc)
# # my_subdomain_2.plot_solution(ud)

ud = np.zeros(my_subdomain_2.get_all_dofs().size)
ud[bc_dofs] = my_subdomain_2.get_projected_function(bc)[bc_dofs]
my_subdomain_2.plot_solution(ud)

f -= K @ ud
Ki = K[i_dofs]         
Kii = Ki[:, i_dofs]   
fi = f[i_dofs]

u = np.zeros(my_subdomain_2.get_all_dofs().size)
u[i_dofs] = scipy.sparse.linalg.spsolve(Kii, fi)
my_subdomain_2.plot_solution(u+ud)




