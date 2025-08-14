import numpy as np
import sympy as sy
import scipy as sp

from typing import Callable, cast
from sympy import Expr

import ufl 

import dolfinx
from dolfinx.mesh import locate_entities
from dolfinx.fem import FunctionSpace

import qugar
from qugar.dolfinx import CustomForm, ds_bdry_unf, form_custom, mapped_normal
from qugar.mesh.unfitted_cart_mesh import UnfittedCartMesh

from geometry import BezierElement, SomeName

type SparseMatrix = sp.sparse._csr.csr_matrix
dtype = np.float64

def make_callable(
    components
) -> Callable:
    def callable_fn(x):
        return np.vstack([f(x[0], x[1]) + 0*x[0] for f in components])
    return callable_fn

def zero_function(X):
    return (0*X[0], 0*X[1])

class MappedElasticity:

    def __init__(
        self,
        E: float = 2.5,
        nu: float = 0.25,
        dim: int = 2,
        interior_bc: list[tuple] = [],
        exterior_bc: list[tuple] = [],
        source: Callable = zero_function,
        u: list[Expr] | None = None,
        K_model = None,
        M_model = None,
        bM_model = None
    ) -> None:
        
        self.E = E
        self.nu = nu
        self.dim = dim

        self.interior_bc = interior_bc
        self.exterior_bc = exterior_bc
        self.source = source
        self.u = u

        self.mu = E/(2*(1+nu))
        self.lambda_ = (E*nu)/((1+nu)*(1-2*nu))

        self.K_model = K_model
        self.M_model = M_model
        self.bM_model = bM_model

        if u:

            self.u_callable, self.f_callable, self.stress_callable = self._create_manufactured_solution()

    def _create_manufactured_solution(self) -> tuple[Callable[[np.ndarray], np.ndarray]]:

        x, y = sy.symbols('x y')
        u = sy.Matrix(self.u)

        grad_u = sy.Matrix([
            [sy.diff(u[0], x), sy.diff(u[0], y)],
            [sy.diff(u[1], x), sy.diff(u[1], y)]
        ])

        strain = (grad_u + grad_u.T) / 2

        mu, lmbda = sy.symbols('mu lambda')
        stress = lmbda * sy.trace(strain) * sy.eye(2) + 2 * mu * strain
        div_stress = sy.Matrix([
            sy.diff(stress[0, 0], x) + sy.diff(stress[0, 1], y),
            sy.diff(stress[1, 0], x) + sy.diff(stress[1, 1], y)
        ])

        f = - div_stress
        mu_val = self.mu
        lmbda_val = self.lambda_

        u_num = [sy.lambdify((x, y), u[i], "numpy") for i in range(self.dim)]
        f_num = [sy.lambdify((x, y), f[i].subs({"mu": mu_val, "lambda": lmbda_val}), "numpy") for i in range(self.dim)]
        stress_num = [sy.lambdify((x, y), stress[i, j].subs({"mu": mu_val, "lambda": lmbda_val}), "numpy") for i in range(self.dim) for j in range(self.dim)]

        u_callable = make_callable(u_num)
        f_callable = make_callable(f_num)
        stress_callable = make_callable(stress_num)

        return u_callable, f_callable, stress_callable

    def create_function_space(self, unf_mesh: UnfittedCartMesh, degree: int) -> tuple[FunctionSpace, FunctionSpace]:

        V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree, (self.dim,)))
        V2 = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree, (self.dim, self.dim)))

        return V, V2
    


    def assemble_boundary_mass(self, unf_mesh: UnfittedCartMesh, C: SparseMatrix, aux: Callable, degree: float) -> SparseMatrix:
        
        facet_tags = unf_mesh.create_facet_tags(cut_tag=0, full_tag=0, ext_integral=True)
        ds = ufl.ds(subdomain_id=0, domain=unf_mesh, subdomain_data=facet_tags)

        V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree))
        V2 = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree, (self.dim, self.dim)))

        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

        Ch = dolfinx.fem.Function(V2)
        Ch.interpolate(aux)

        k, l = ufl.indices(2)

        a = ufl.as_tensor(
            Ch[k, l] * u * v,
            (k, l)
        )

        forms = [[cast(CustomForm, form_custom(a[k, l] * ds, unf_mesh, dtype=dtype))
                   for l in range(self.dim)] for k in range(self.dim)]

        nb = V.dofmap.index_map.size_local
        A = np.zeros((nb, nb, self.dim, self.dim))

        for k in range(self.dim):
            for l in range(self.dim):
                mat = dolfinx.fem.petsc.assemble_matrix(forms[k][l], coeffs=forms[k][l].pack_coefficients())
                mat.assemble()
                mat_dense = mat.convert("dense")
                A[:, :, k, l] = mat_dense.getDenseArray()

        M = A.transpose(0, 2, 1, 3)
        M = M.reshape(nb * 2, nb * 2)
        M = C @ M @ C.T

        return M

    def assemble_stiffness(self, unf_mesh: UnfittedCartMesh, C: SparseMatrix, aux: Callable, degree: float) -> SparseMatrix:

        V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree))
        V4 = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree, (self.dim, self.dim, self.dim, self.dim)))

        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

        Ch = dolfinx.fem.Function(V4)
        Ch.interpolate(aux)

        grad_u = ufl.grad(u)
        grad_v = ufl.grad(v)

        k, l = ufl.indices(2)

        a = ufl.as_tensor(
            sum(Ch[p, k, q, l] * grad_u[p] * grad_v[q] for p in range(self.dim) for q in range(self.dim)),
            (k, l)
        )

        forms = [[cast(CustomForm, form_custom(a[k, l] * ufl.dx, unf_mesh, dtype=dtype))
                   for l in range(self.dim)] for k in range(self.dim)]

        nb = V.dofmap.index_map.size_local
        A = np.zeros((nb, nb, self.dim, self.dim))

        for k in range(self.dim):
            for l in range(self.dim):
                mat = dolfinx.fem.petsc.assemble_matrix(forms[k][l], coeffs=forms[k][l].pack_coefficients())
                mat.assemble()
                mat_dense = mat.convert("dense")
                A[:, :, l, k] = mat_dense.getDenseArray()

        K = A.transpose(0, 2, 1, 3)
        K = K.reshape(nb * 2, nb * 2)
        K = C @ K @ C.T

        return K
    
    def assemble_mass(self, unf_mesh: UnfittedCartMesh, C: SparseMatrix, aux: Callable, degree: float) -> SparseMatrix:

        V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree))
        V2 = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree, (self.dim, self.dim)))

        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

        Ch = dolfinx.fem.Function(V2)
        Ch.interpolate(aux)

        k, l = ufl.indices(2)

        a = ufl.as_tensor(
            Ch[k, l] * u * v,
            (k, l)
        )

        forms = [[cast(CustomForm, form_custom(a[k, l] * ufl.dx, unf_mesh, dtype=dtype))
                   for l in range(self.dim)] for k in range(self.dim)]

        nb = V.dofmap.index_map.size_local
        A = np.zeros((nb, nb, self.dim, self.dim))

        for k in range(self.dim):
            for l in range(self.dim):
                mat = dolfinx.fem.petsc.assemble_matrix(forms[k][l], coeffs=forms[k][l].pack_coefficients())
                mat.assemble()
                mat_dense = mat.convert("dense")
                A[:, :, k, l] = mat_dense.getDenseArray()

        M = A.transpose(0, 2, 1, 3)
        M = M.reshape(nb * 2, nb * 2)
        M = C @ M @ C.T

        return M
    
    
    def assemble_stiffness_core(self, unf_mesh: UnfittedCartMesh, basis: SomeName, degree: float) -> SparseMatrix:

        nc = (basis.degree+1)**2
        V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree))
        Vb = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree, (nc,)))

        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        
        def aux(x):

            return basis.evaluate_basis(x.T).T

        bh = dolfinx.fem.Function(Vb)
        bh.interpolate(aux)

        grad_u = ufl.grad(u)
        grad_v = ufl.grad(v)

        k, l, c = ufl.indices(3)

        a = ufl.as_tensor(
            grad_u[k] * grad_v[l] * bh[c], (k, l, c)
        )

        forms = [[[
            cast(CustomForm, form_custom(a[k, l, c] * ufl.dx, unf_mesh, dtype=dtype))
                   for l in range(self.dim)] 
                   for k in range(self.dim)]
                   for c in range(nc)]

        nb = V.dofmap.index_map.size_local
        A = np.zeros((nb, nb, self.dim, self.dim, nc))

        coeffs = forms[0][0][0].pack_coefficients()

        for k in range(self.dim):
            for l in range(self.dim):
                for c in range(nc):
                    mat = dolfinx.fem.petsc.assemble_matrix(
                        forms[c][k][l], 
                        coeffs = coeffs
                    )
                    mat.assemble()
                    mat_dense = mat.convert("dense")
                    A[:, :, k, l, c] = mat_dense.getDenseArray()

        return A
    
    def assemble_mass_core(self, unf_mesh: UnfittedCartMesh, basis: SomeName, degree: float) -> SparseMatrix:

        nc = (basis.degree+1)**2
        V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree))
        Vb = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree, (nc,)))

        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        
        def aux(x):

            return basis.evaluate_basis(x.T).T

        bh = dolfinx.fem.Function(Vb)
        bh.interpolate(aux)

        c = ufl.indices(1)

        a = ufl.as_tensor(
            u * v * bh[c], (c)
        )

        forms = [
            cast(CustomForm, form_custom(a[c] * ufl.dx, unf_mesh, dtype=dtype))
                   for c in range(nc)]

        nb = V.dofmap.index_map.size_local
        A = np.zeros((nb, nb, nc))

        for c in range(nc):
            mat = dolfinx.fem.petsc.assemble_matrix(
                forms[c], 
                coeffs=forms[c].pack_coefficients()
            )
            mat.assemble()
            mat_dense = mat.convert("dense")
            A[:, :, c] = mat_dense.getDenseArray()

        return A
    
    def assemble_boundary_mass_core(self, unf_mesh: UnfittedCartMesh, basis: SomeName, degree: float) -> SparseMatrix:

        facet_tags = unf_mesh.create_facet_tags(cut_tag=0, full_tag=0, ext_integral=True)
        ds = []

        hor_facet_tags = [(0, np.array([0, 0, 0, 3]))]
        vert_facet_tags = [(1, np.arrayy([0, 1, 0, 2]))]

        ds.append(ufl.ds(subdomain_id=0, domain=unf_mesh, subdomain_data=hor_facet_tags))
        ds.append(ufl.ds(subdomain_id=1, domain=unf_mesh, subdomain_data=vert_facet_tags))

        nc = (basis.degree+1)**2
        V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree))
        Vb = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree, (nc,)))

        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        
        def aux(x):

            return basis.evaluate_basis(x.T).T

        bh = dolfinx.fem.Function(Vb)
        bh.interpolate(aux)

        c = ufl.indices(1)

        a = ufl.as_tensor(
            u * v * bh[c], (c)
        )

        forms = [[
            cast(CustomForm, form_custom(a[c] * ds[k], unf_mesh, dtype=dtype))
                   for c in range(nc)]
                   for k in range(2)]

        nb = V.dofmap.index_map.size_local
        A = np.zeros((nb, nb, nc, 2))

        for c in range(nc):
            for k in range(2):

                mat = dolfinx.fem.petsc.assemble_matrix(
                    forms[k][c], 
                    coeffs=forms[k][c].pack_coefficients()
                )
                mat.assemble()
                mat_dense = mat.convert("dense")
                A[:, :, c, k] = mat_dense.getDenseArray()

        return A
    
    

    def assemble_right_hand_side(self, unf_mesh: UnfittedCartMesh, V: FunctionSpace, V2: FunctionSpace, C: SparseMatrix) -> np.ndarray: 

        if not self.u:
            return C @ self._assemble_rhs_general(unf_mesh, V)
                
        else: 
            return C @ self._assemble_rhs_manufatured_solution(unf_mesh, V, V2)

    def _assemble_rhs_manufatured_solution(self, unf_mesh: UnfittedCartMesh, V: FunctionSpace, V2: FunctionSpace) -> np.ndarray:

        v = ufl.TestFunction(V)

        fh = dolfinx.fem.Function(V)
        fh.interpolate(self.f_callable)
        
        cell_tags = unf_mesh.create_cell_meshtags(cut_tag=0, full_tag=1)

        dx = ufl.dx(
            subdomain_id=(0, 1),
            domain=unf_mesh,
            subdomain_data=cell_tags,
        )

        bound_normal = mapped_normal(unf_mesh)

        ds_unf = ds_bdry_unf(subdomain_id=0, domain=unf_mesh, subdomain_data=cell_tags)
        
        sh = dolfinx.fem.Function(V2)
        sh.name = "sh"
        sh.interpolate(self.stress_callable)

        # There is a bug when both integrals are computed toghether.

        L = ufl.dot(fh, v) * dx 
        L_form = cast(CustomForm, form_custom(L, unf_mesh, dtype=dtype))
        b = dolfinx.fem.petsc.assemble_vector(L_form, coeffs=L_form.pack_coefficients())
        f = b.array

        L = ufl.dot(ufl.dot(sh, bound_normal) ,v) * ds_unf
        L_form = cast(CustomForm, form_custom(L, unf_mesh, dtype=dtype))
        b = dolfinx.fem.petsc.assemble_vector(L_form, coeffs=L_form.pack_coefficients())
        f += b.array

        return f
    
    def _assemble_rhs_general(self, unf_mesh: UnfittedCartMesh, V: FunctionSpace) -> np.ndarray:

        v = ufl.TestFunction(V)

        cut_facets = unf_mesh.get_cut_facets(ext_integral=True)
        full_facets = unf_mesh.get_full_facets(ext_integral=True)
        exterior_facets = full_facets.concatenate(cut_facets)

        top = unf_mesh.topology

        top.create_connectivity(2, 1)
        top.create_connectivity(1, 2)

        c_2_e = top.connectivity(2, 1)
        e_2_c = top.connectivity(1, 2)
        
        fh = dolfinx.fem.Function(V)
        fh.interpolate(self.source)
        
        L = ufl.dot(fh, v) * ufl.dx 
        L_form = cast(CustomForm, form_custom(L, unf_mesh, dtype=dtype))
        b = dolfinx.fem.petsc.assemble_vector(L_form, coeffs=L_form.pack_coefficients())
        f = b.array
        
        facet_data = []
        funs = []

        for (type_, fun, marker, ind) in self.exterior_bc:

            if type_ == 1:

                facets = locate_entities(unf_mesh, 1, marker)
                if facets.size > 0:

                    cell_ids = np.array([e_2_c.links(facet)[0] for facet in facets])
                    local_facet_ids = np.array([np.where(c_2_e.links(c) == e)[0][0] for e, c in zip(facets, cell_ids)])

                    all_facets = qugar.mesh.mesh_facets.MeshFacets(cell_ids, local_facet_ids)
                    bc_facets = all_facets.intersect(exterior_facets)
                    facet_data.append(bc_facets.as_array())
                    funs.append(dolfinx.fem.Function(V))
                    funs[-1].interpolate(fun)

        if facet_data:

            facet_tags = list(enumerate(facet_data))

            for ind, fun in enumerate(funs):

                if ind == 0:
                    L = ufl.dot(fun, v) * ufl.ds(subdomain_id=ind, domain=unf_mesh, subdomain_data=facet_tags)
                else:
                    L += ufl.dot(fun, v) * ufl.ds(subdomain_id=ind, domain=unf_mesh, subdomain_data=facet_tags)

            L_form = cast(CustomForm, form_custom(L, unf_mesh, dtype=dtype))
            b = dolfinx.fem.petsc.assemble_vector(L_form, coeffs=L_form.pack_coefficients())
            f += b.array

        for (type_, fun, marker, ind) in self.interior_bc:

            assert type_ == 1

            cell_tags = unf_mesh.create_cell_meshtags(cut_tag=0, full_tag=1)
            ds_unf = ds_bdry_unf(subdomain_id=0, domain=unf_mesh, subdomain_data=cell_tags)

            fh = dolfinx.fem.Function(V)
            fh.interpolate(fun)
            L = ufl.dot(fh, v) * ds_unf

            L_form = cast(CustomForm, form_custom(L, unf_mesh, dtype=dtype))
            b = dolfinx.fem.petsc.assemble_vector(L_form, coeffs=L_form.pack_coefficients())
            f += b.array

        return f
    

class Poisson:

    def __init__(
        self,
        kappa: float = 1.0,
        dim: int = 1,
        interior_bc: list[tuple] = [],
        exterior_bc: list[tuple] = [],
        source: Callable = zero_function,
        u: list[Expr] | None = None
    ) -> None:
        
        self.kappa = kappa
        self.dim = dim

        self.interior_bc = interior_bc
        self.exterior_bc = exterior_bc
        self.source = source
        self.u = u

        if u:

            self.u_callable, self.f_callable, self.grad_callable = self._create_manufactured_solution()

    def _create_manufactured_solution(self) -> tuple[Callable[[np.ndarray], np.ndarray]]:

        x, y = sy.symbols('x y')
        
        u = self.u
        a = sy.symbols('a')
        a_val = self.kappa
        
        grad_u = a * sy.Matrix([sy.diff(u, x), sy.diff(u, y)])
        grad_u = grad_u.subs(a, a_val)

        laplacian_u = sy.diff(u, x, 2) + sy.diff(u, y, 2)
        f = -a * laplacian_u
        f = f.subs(a, a_val)

        u_num = sy.lambdify((x, y), u, "numpy")
        f_num = sy.lambdify((x, y), f, "numpy")
        grad_num = [sy.lambdify((x, y), comp, "numpy") for comp in grad_u]

        u_callable = make_callable([u_num])
        f_callable = make_callable([f_num])
        grad_callable = make_callable(grad_num)
            
        return u_callable, f_callable, grad_callable

    def create_function_space(self, unf_mesh: UnfittedCartMesh, degree: int) -> tuple[FunctionSpace, FunctionSpace]:

        V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree))
        V2 = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree, (2,)))

        return V, V2
    
    def assemble_boundary_mass(self, unf_mesh: UnfittedCartMesh, V: FunctionSpace) -> SparseMatrix:

        facet_tags = unf_mesh.create_facet_tags(cut_tag=0, full_tag=0, ext_integral=True)
        ds = ufl.ds(subdomain_id=0, domain=unf_mesh, subdomain_data=facet_tags)

        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        m = u * v * ds

        m_form = cast(CustomForm, form_custom(m, unf_mesh, dtype=dtype))
        M = dolfinx.fem.petsc.assemble_matrix(m_form, coeffs=m_form.pack_coefficients())
        M.assemble()

        ia, ja, a = M.getValuesCSR()
        M = sp.sparse.csr_matrix((a, ja, ia), shape=M.getSize())

        return M

    def assemble_stiffness(self, unf_mesh: UnfittedCartMesh, V: FunctionSpace) -> SparseMatrix:

        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

        kappa = self.kappa

        a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        a_form = cast(CustomForm, form_custom(a, unf_mesh, dtype=dtype))
        A = dolfinx.fem.petsc.assemble_matrix(a_form, coeffs=a_form.pack_coefficients())
        A.assemble()

        ia, ja, a = A.getValuesCSR()
        K = sp.sparse.csr_matrix((a, ja, ia), shape=A.getSize())

        return K
    
    def assemble_mass(self, unf_mesh: UnfittedCartMesh, V: FunctionSpace) -> SparseMatrix:

        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

        a = u * v * ufl.dx
        a_form = cast(CustomForm, form_custom(a, unf_mesh, dtype=dtype))
        A = dolfinx.fem.petsc.assemble_matrix(a_form, coeffs=a_form.pack_coefficients())
        A.assemble()

        ia, ja, a = A.getValuesCSR()
        M = sp.sparse.csr_matrix((a, ja, ia), shape=A.getSize())

        return M

    def assemble_right_hand_side(self, unf_mesh: UnfittedCartMesh, V: FunctionSpace, V2: FunctionSpace) -> np.ndarray: 

        if not self.u:
            return self._assemble_rhs_general(unf_mesh, V)
                
        else: 
            return self._assemble_rhs_manufatured_solution(unf_mesh, V, V2)

    def _assemble_rhs_manufatured_solution(self, unf_mesh: UnfittedCartMesh, V: FunctionSpace, V2: FunctionSpace) -> np.ndarray:

        v = ufl.TestFunction(V)

        fh = dolfinx.fem.Function(V)
        fh.interpolate(self.f_callable)
        
        cell_tags = unf_mesh.create_cell_meshtags(cut_tag=0, full_tag=1)

        dx = ufl.dx(
            subdomain_id=(0, 1),
            domain=unf_mesh,
            subdomain_data=cell_tags,
        )

        sh = dolfinx.fem.Function(V2)
        sh.name = "sh"
        sh.interpolate(self.grad_callable)

        bound_normal = mapped_normal(unf_mesh)

        ds_unf = ds_bdry_unf(subdomain_id=0, domain=unf_mesh, subdomain_data=cell_tags)
        
        # There is a bug when both integrals are computed toghether.

        L = fh * v * dx 
        L_form = cast(CustomForm, form_custom(L, unf_mesh, dtype=dtype))
        b = dolfinx.fem.petsc.assemble_vector(L_form, coeffs=L_form.pack_coefficients())
        f = b.array

        L = self.kappa * ufl.dot(sh, bound_normal) * v * ds_unf
        L_form = cast(CustomForm, form_custom(L, unf_mesh, dtype=dtype))
        b = dolfinx.fem.petsc.assemble_vector(L_form, coeffs=L_form.pack_coefficients())
        f += b.array

        return f
    
    def _assemble_rhs_general(self, unf_mesh: UnfittedCartMesh, V: FunctionSpace) -> np.ndarray:

        v = ufl.TestFunction(V)

        cut_facets = unf_mesh.get_cut_facets(ext_integral=True)
        full_facets = unf_mesh.get_full_facets(ext_integral=True)
        exterior_facets = full_facets.concatenate(cut_facets)

        top = unf_mesh.topology

        top.create_connectivity(2, 1)
        top.create_connectivity(1, 2)

        c_2_e = top.connectivity(2, 1)
        e_2_c = top.connectivity(1, 2)
        
        fh = dolfinx.fem.Function(V)
        fh.interpolate(self.source)
        
        L = fh * v * ufl.dx 
        L_form = cast(CustomForm, form_custom(L, unf_mesh, dtype=dtype))
        b = dolfinx.fem.petsc.assemble_vector(L_form, coeffs=L_form.pack_coefficients())
        f = b.array
        
        facet_data = []
        funs = []

        for (type_, fun, marker, ind) in self.exterior_bc:

            if type_ == 1:

                facets = locate_entities(unf_mesh, 1, marker)
                if facets.size > 0:

                    cell_ids = np.array([e_2_c.links(facet)[0] for facet in facets])
                    local_facet_ids = np.array([np.where(c_2_e.links(c) == e)[0][0] for e, c in zip(facets, cell_ids)])

                    all_facets = qugar.mesh.mesh_facets.MeshFacets(cell_ids, local_facet_ids)
                    bc_facets = all_facets.intersect(exterior_facets)
                    facet_data.append(bc_facets.as_array())
                    funs.append(dolfinx.fem.Function(V))
                    funs[-1].interpolate(fun)

        if facet_data:

            facet_tags = list(enumerate(facet_data))

            for ind, fun in enumerate(funs):

                if ind == 0:
                    L = self.kappa * fun * v * ufl.ds(subdomain_id=ind, domain=unf_mesh, subdomain_data=facet_tags)
                else:
                    L += self.kappa * fun * v * ufl.ds(subdomain_id=ind, domain=unf_mesh, subdomain_data=facet_tags)

            L_form = cast(CustomForm, form_custom(L, unf_mesh, dtype=dtype))
            b = dolfinx.fem.petsc.assemble_vector(L_form, coeffs=L_form.pack_coefficients())
            f += b.array

        for (type_, fun, marker, ind) in self.interior_bc:

            assert type_ == 1

            cell_tags = unf_mesh.create_cell_meshtags(cut_tag=0, full_tag=1)
            ds_unf = ds_bdry_unf(subdomain_id=0, domain=unf_mesh, subdomain_data=cell_tags)

            fh = dolfinx.fem.Function(V)
            fh.interpolate(fun)
            L = self.kappa * fun * v * ds_unf

            L_form = cast(CustomForm, form_custom(L, unf_mesh, dtype=dtype))
            b = dolfinx.fem.petsc.assemble_vector(L_form, coeffs=L_form.pack_coefficients())
            f += b.array

        return f
    