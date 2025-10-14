import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.linalg
import scipy.sparse
import scipy
import scipy.sparse.linalg

import sys

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.fem
import dolfinx

from FLASh.mesh import (
    GlobalDofsManager,
    SplineGeometry
)

from FLASh.pde.solver import BaseSolver

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg, gmres
from scipy.sparse.linalg import splu, spsolve

type SparseMatrix = scipy.sparse._csr.csr_matrix


class residual_tracker:

    """Class for tracking the iterative solver residuals and interations.
    """

    def __init__(self, fun, fprint = False):
        self.niter = 0
        self.res = []
        self.print = fprint
        if self.print:
            print("Solving with CG: ")

    def __call__(self, rk=None):
        self.niter += 1
        self.res.append(rk)
        if self.print:
            sys.stdout.write(f"\033[F\033[KIteration {self.niter}: Residual norm = {np.linalg.norm(rk):.2e}\n")
            sys.stdout.flush()

def monitor(ksp, its, rnorm):
    print(f"Iteration {its}: Residual norm = {rnorm:.2e}")

class Assembler:

    """Class for managing the assembling and use of the fetidp operators.
    """

    def __init__(
            self, 
            gbl_dofs_mngr: GlobalDofsManager
        ) -> None: 
        """Initializes the class.

        Args:
            gbl_dofs_mngr (GlobalDofsManager): Global dofs manager.
        """
        self.gbl_dofs_mngr = gbl_dofs_mngr
        self.assemble_paralelization()
        self.assemble_subdomains()
        self.assemble_S0()
        self.assemble_global_ordering()
        self.assemble_f()

        self.internal_tracker = []

    def assemble_paralelization(self) -> None:

        comm = self.gbl_dofs_mngr.communicators.global_comm
        petsc_comm = self.gbl_dofs_mngr.communicators.petsc_comm
        size = comm.Get_size()
        rank = comm.Get_rank()

        N = self.gbl_dofs_mngr.get_num_subdomains()

        counts = [N // size + (1 if i < N % size else 0) for i in range(size)]
        starts = np.cumsum([0] + counts[:-1])
        self.process_subdomains = np.arange(starts[rank], starts[rank] + counts[rank])
        
        self.comm = comm
        self.petsc_comm = petsc_comm

    def assemble_subdomains(self) -> None:

        """Assembles and stores the subdomain operators.
        """
        N = self.gbl_dofs_mngr.get_num_subdomains()

        self.subdomains_Sbb = []
        self.subdomains_Lu_Sbb = []
        self.subdomains_Piv_Sbb = []
        self.subdomains_Lu_Kii = []
        self.subdomains_Piv_Kii = []

        self.subdomains_Uib = []

        self.subdomains_Kii = []
        self.subdomains_Kib = []
        self.subdomains_Kbb = []
        self.subdomains_fi = []
        self.subdomains_fb = []

        self.subdomains_R = []

        for s_ind, s_id in enumerate(self.process_subdomains):

            subdomain = self.gbl_dofs_mngr.subdomains[s_ind]

            i_dofs = subdomain.interior_dofs
            b_dofs = subdomain.boundary_dofs

            K, f = subdomain.K, subdomain.f

            Ki = K[i_dofs]         
            Kii = Ki[:, i_dofs]     
            Kib = Ki[:, b_dofs]      
            Kb = K[b_dofs]          
            Kbb = Kb[:, b_dofs] 

            Lu_Kii, Piv_Kii = scipy.linalg.lu_factor(Kii)

            Uib = scipy.linalg.lu_solve((Lu_Kii, Piv_Kii), Kib)

            Sbb = Kbb - Kib.T @ Uib

            nvp = subdomain.get_num_local_vertex_primals()

            rSbb = Sbb[nvp:][:,nvp:]
            Lu_Sbb, Piv_Sbb = scipy.linalg.lu_factor(rSbb)

            fi = f[i_dofs]
            fb = f[b_dofs] 

            self.subdomains_Sbb.append(Sbb)
            self.subdomains_Lu_Kii.append(Lu_Kii)
            self.subdomains_Piv_Kii.append(Piv_Kii)
            self.subdomains_Lu_Sbb.append(Lu_Sbb)
            self.subdomains_Piv_Sbb.append(Piv_Sbb)

            self.subdomains_Uib.append(Uib)

            self.subdomains_Kii.append(Kii)
            self.subdomains_Kib.append(Kib)
            self.subdomains_Kbb.append(Kbb)
            self.subdomains_fi.append(fi)
            self.subdomains_fb.append(fb)

            R = self.gbl_dofs_mngr.create_R_inds(s_id)

            self.subdomains_R.append(R)
    
    def assemble_global_ordering(self) -> None: 

        n_act_b = self.gbl_dofs_mngr.get_active_boundary_dofs().size

        N = self.gbl_dofs_mngr.get_num_subdomains()

        active_dofs_local = np.zeros(shape=(N), dtype=np.int32)

        for s_ind, s_id in enumerate(self.process_subdomains):

            subdomain = self.gbl_dofs_mngr.subdomains[s_ind]
            i_dofs = subdomain.interior_dofs
            active_dofs_local[s_id] = i_dofs.size

        active_dofs_global = np.zeros(shape = active_dofs_local.shape, dtype=np.int32)
        self.comm.Allreduce(active_dofs_local, active_dofs_global, MPI.SUM)
        self.total_act_dofs = int(np.sum(active_dofs_global) + n_act_b)

        local_dofs_ordering = []
        count = 0

        for s_ind, s_id in enumerate(self.process_subdomains):
            local_dofs_ordering.append(
                np.array([
                    count,
                    count + active_dofs_local[s_id]
                ])
            )
            count = count + active_dofs_local[s_id]

        rank = self.comm.Get_rank()

        if rank == 0:
            total_local_dofs = int(count + n_act_b)

        else: 
            total_local_dofs = int(count)

        local_dofs_ordering = np.array(local_dofs_ordering)

        self.local_dofs_ord = local_dofs_ordering
        self.total_local_dofs = total_local_dofs

    def apply_S(self, x:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        n_b = self.gbl_dofs_mngr.get_num_boundary_dofs()
        act_boundary_dofs = self.gbl_dofs_mngr.get_active_boundary_dofs()

        w = np.zeros(n_b, dtype=np.float64)
        y = np.zeros(n_b, dtype=np.float64)

        y[act_boundary_dofs] = x

        for s_ind, s_id in enumerate(self.process_subdomains):
            R = self.subdomains_R[s_ind]
            Sbb = self.subdomains_Sbb[s_ind]
            w[R] += Sbb @ y[R]

        w_local = w[act_boundary_dofs]
        w_global = np.zeros(shape = w_local.shape)
        self.comm.Allreduce(w_local, w_global, MPI.SUM)

        return w_global
    
    def assemble_S0(self) -> None:

        """Assembles and stores the primal Schur operator.
        """

        subdomains_R = []
        subdomains_Rc = []
        subdomains_C = []
        subdomains_D = []


        for s_ind, s_id in enumerate(self.process_subdomains):

            subdomain = self.gbl_dofs_mngr.subdomains[s_ind]

            R = self.gbl_dofs_mngr.create_R_inds(s_id)
            Rc = self.gbl_dofs_mngr.create_Rc_inds(s_id)
            D = self.gbl_dofs_mngr.create_D(s_id)
            C = subdomain.create_C().toarray()

            subdomains_R.append(R)
            subdomains_Rc.append(Rc)
            subdomains_C.append(C)
            subdomains_D.append(D)

        self.subdomains_R = subdomains_R
        self.subdomains_Rc = subdomains_Rc
        self.subdomains_C = subdomains_C
        self.subdomains_D = subdomains_D

        subdomains_C_reduced = []
        subdomains_T = []

        for s_ind, s_id in enumerate(self.process_subdomains):

            subdomain = self.gbl_dofs_mngr.subdomains[s_ind]

            C = self.subdomains_C[s_ind]
            Lu_Sbb = self.subdomains_Lu_Sbb[s_ind]
            Piv_Sbb = self.subdomains_Piv_Sbb[s_ind]

            nvp = subdomain.get_num_local_vertex_primals()

            C_reduced = C[nvp:,:]
            C_reduced = C_reduced[:,nvp:]
            T = C_reduced @ scipy.linalg.lu_solve((Lu_Sbb, Piv_Sbb), C_reduced.T)

            subdomains_C_reduced.append(C_reduced)
            subdomains_T.append(T)

        self.subdomains_C_reduced = subdomains_C_reduced
        self.subdomains_T = subdomains_T

        subdomains_Psi = []
        subdomains_Kc = []

        for s_ind, s_id in enumerate(self.process_subdomains):

            subdomain = self.gbl_dofs_mngr.subdomains[s_ind]
            n_b = subdomain.get_num_boundary_dofs()
            n_p = subdomain.get_num_local_primals()

            Psi = np.zeros((n_b, n_p))
            Kc = np.zeros((n_p, n_p))
            I = np.eye(n_p)

            for j in range(n_p):

                psi_j, lambda_j = self.solve_subdomain_problem(
                    s_ind,
                    s_id,
                    np.zeros((n_b)),
                    I[:,j]
                )

                Psi[:,j] = psi_j
                Kc[:,j] = -lambda_j

            subdomains_Psi.append(Psi)
            subdomains_Kc.append(Kc)

        self.subdomains_Psi = subdomains_Psi
        self.subdomains_Kc = subdomains_Kc

        num_primals = self.gbl_dofs_mngr.get_num_primals()
        active_primals = self.gbl_dofs_mngr.get_active_primal_dofs()

        n = len(active_primals)
        Sc_local = scipy.sparse.lil_matrix((n, n))

        active_set = set(active_primals)
        index_map = {p: i for i, p in enumerate(active_primals)}

        for s_ind, s_id in enumerate(self.process_subdomains):
            Rc = self.subdomains_Rc[s_ind]  
            Kc = self.subdomains_Kc[s_ind]  

            full_indices = [i for i, p in enumerate(Rc) if p in active_set]
            active_indices = [index_map[Rc[i]] for i in full_indices]

            if active_indices:
                Kc_sub = Kc[np.ix_(full_indices, full_indices)]

                for i_local, i_global in enumerate(active_indices):
                    for j_local, j_global in enumerate(active_indices):
                        Sc_local[i_global, j_global] += Kc_sub[i_local, j_local]

        N = Sc_local.shape[0]

        Sc_local = Sc_local.tocsr()
        Sc_petsc = PETSc.Mat().createAIJ([N, N], comm=PETSc.COMM_WORLD)
        Sc_petsc.setUp()

        for i in range(N):
            row_start = Sc_local.indptr[i]
            row_end = Sc_local.indptr[i + 1]
            cols = Sc_local.indices[row_start:row_end]
            vals = Sc_local.data[row_start:row_end]
            if len(cols) > 0:
                Sc_petsc.setValues(i, cols, vals, addv=PETSc.InsertMode.ADD_VALUES)

        Sc_petsc.assemblyBegin()
        Sc_petsc.assemblyEnd()

        ksp = PETSc.KSP().create()
        ksp.setOperators(Sc_petsc)
        ksp.setType('preonly')

        pc = ksp.getPC()
        pc.setType('cholesky')
        pc.setFactorSolverType('mumps')

        ksp.setUp()

        # ksp = PETSc.KSP().create()
        # ksp.setOperators(Sc_petsc)

        # ksp.setType('cg')  

        # pc = ksp.getPC()
        # pc.setType('gamg')  

        # ksp.setTolerances(rtol=1e-8)  

        # ksp.setFromOptions()  
        # ksp.setUp()

        self.Sc_sover = ksp
        self.Sc = Sc_petsc

    def apply_S0(self, x:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        N = self.gbl_dofs_mngr.get_num_subdomains()

        num_boundary = self.gbl_dofs_mngr.get_num_boundary_dofs()
        act_boundary_dofs = self.gbl_dofs_mngr.get_active_boundary_dofs()
        num_primals = self.gbl_dofs_mngr.get_num_primals()
        
        y_local = np.zeros(num_boundary, dtype=np.float64)
        x1 = np.zeros(num_boundary, dtype=np.float64)
        x1[act_boundary_dofs] = x

        for s_ind, s_id in enumerate(self.process_subdomains):

            subdomain = self.gbl_dofs_mngr.subdomains[s_ind]
            n_p = subdomain.get_num_local_primals()

            D = self.subdomains_D[s_ind]
            R = self.subdomains_R[s_ind]
            
            w_local, mu = self.solve_subdomain_problem(
                s_ind,
                s_id, 
                D @ x1[R], 
                np.zeros(n_p)
            )

            y_local[R] += D.T @ w_local

        z_local = np.zeros(num_primals, dtype=np.float64)

        for s_ind, s_id in enumerate(self.process_subdomains):

            D = self.subdomains_D[s_ind]
            R = self.subdomains_R[s_ind]
            Rc = self.subdomains_Rc[s_ind]
            Psi = self.subdomains_Psi[s_ind]

            z_local[Rc] += Psi.T @ (D @ x1[R])

        z_global = np.zeros(shape = z_local.shape)
        self.comm.Reduce(z_local, z_global, MPI.SUM, root = 0)

        z_global = self.solve_coarse_problem(z_global)

        self.comm.Bcast(z_global, root = 0)

        for s_ind, s_id in enumerate(self.process_subdomains):

            D = self.subdomains_D[s_ind]
            R = self.subdomains_R[s_ind]
            Rc = self.subdomains_Rc[s_ind]
            Psi = self.subdomains_Psi[s_ind]

            y_local[R] += D @ (Psi @ z_global[Rc])

        y_local = y_local[act_boundary_dofs]
        y_global = np.zeros(shape = y_local.shape)
        self.comm.Allreduce(y_local, y_global, MPI.SUM)

        return y_global 

    def solve_coarse_problem(self, x:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        num_primals = self.gbl_dofs_mngr.get_num_primals()
        active_primals = self.gbl_dofs_mngr.get_active_primal_dofs()

        x_petcs = self.Sc.getVecRight()
        b_petcs = self.Sc.getVecLeft()

        if self.comm.Get_rank() == 0:

            b_petcs.setValues(np.arange(active_primals.size, dtype=np.int32), x[active_primals], addv=PETSc.InsertMode.ADD_VALUES)

        b_petcs.assemblyBegin(); b_petcs.assemblyEnd()
        self.Sc_sover.solve(b_petcs, x_petcs)

        if self.comm.Get_rank() == 0:
            x_seq = PETSc.Vec().createSeq(x_petcs.getSize(), comm=PETSc.COMM_SELF)
        else:
            x_seq = PETSc.Vec().createMPI(0, x_petcs.getSize(), comm=PETSc.COMM_SELF)  

        scatter, _ = PETSc.Scatter.toZero(x_petcs)
        scatter.scatter(x_petcs, x_seq, addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

        y = np.zeros(num_primals, dtype=np.float64)

        if self.comm.Get_rank() == 0:
            y[active_primals] = x_seq.getArray(readonly=True)

        return y
    
    def solve_subdomain_problem(
            self, 
            subdomain_local_id: np.int32,
            subdomain_id: np.int32, 
            x1:npt.NDArray[np.float64],
            x2:npt.NDArray[np.float64]
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:

        subdomain = self.gbl_dofs_mngr.subdomains[subdomain_local_id]

        n_b = subdomain.get_num_boundary_dofs()
        n_p = subdomain.get_num_local_primals()
        nvp = subdomain.get_num_local_vertex_primals()

        Sbb = self.subdomains_Sbb[subdomain_local_id]
        Lu_Sbb = self.subdomains_Lu_Sbb[subdomain_local_id]
        Piv_Sbb = self.subdomains_Piv_Sbb[subdomain_local_id]

        C_reduced = self.subdomains_C_reduced[subdomain_local_id]
        T = self.subdomains_T[subdomain_local_id]

        w = np.zeros(n_b, dtype=np.float64)
        mu = np.zeros(n_p, dtype=np.float64)

        w[:nvp] = x2[:nvp]
        f = x1 - Sbb @ w

        f = f[nvp:]
        g = x2[nvp:]

        mu_reduced = spsolve(T, C_reduced @ scipy.linalg.lu_solve((Lu_Sbb, Piv_Sbb), f) - g)

        w_reduced = scipy.linalg.lu_solve((Lu_Sbb, Piv_Sbb), f - C_reduced.T @ mu_reduced)

        w[nvp:] = w_reduced
        mu[nvp:] = mu_reduced
        mu[:nvp] = x1[:nvp] - Sbb[:nvp] @ w

        return w, mu

    def apply_A(self, x:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        n_b = self.gbl_dofs_mngr.get_num_boundary_dofs()
        act_boundary_dofs = self.gbl_dofs_mngr.get_active_boundary_dofs()

        x_b = np.zeros(shape = (n_b))

        if self.comm.Get_rank() == 0:
            x_b[act_boundary_dofs] = x[self.local_dofs_ord[-1,-1]:]

        self.comm.Bcast(x_b, root = 0)

        y = np.zeros(shape = (self.total_local_dofs))
        y_b_local = np.zeros(shape = (n_b))

        for s_ind, s_id in enumerate(self.process_subdomains):

            Kii = self.subdomains_Kii[s_ind]
            Kib = self.subdomains_Kib[s_ind]
            Kbb = self.subdomains_Kbb[s_ind]

            R = self.subdomains_R[s_ind]

            start, end = self.local_dofs_ord[s_ind,0], self.local_dofs_ord[s_ind,1]
            y[start:end] = Kii @ x[start:end] + Kib @ x_b[R]
            y_b_local[R] += Kib.T @ x[start:end] + Kbb @ x_b[R]

        y_b_global = np.zeros(shape = y_b_local.shape)
        self.comm.Allreduce(y_b_local, y_b_global, MPI.SUM)

        if self.comm.Get_rank() == 0:
            y[self.local_dofs_ord[-1,-1]:] = y_b_global[act_boundary_dofs]

        return y

    def apply_M1(self, x:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        n_b = self.gbl_dofs_mngr.get_num_boundary_dofs()
        act_boundary_dofs = self.gbl_dofs_mngr.get_active_boundary_dofs()

        y_b_local = np.zeros(shape = (n_b))

        for s_ind, s_id in enumerate(self.process_subdomains):

            Uib = self.subdomains_Uib[s_ind]

            R = self.subdomains_R[s_ind]

            start, end = self.local_dofs_ord[s_ind,0], self.local_dofs_ord[s_ind,1]
            y_b_local[R] -= Uib.T @ x[start:end]

        y_b_global = np.zeros(shape = y_b_local.shape)
        self.comm.Reduce(y_b_local, y_b_global, MPI.SUM, root = 0)

        y = x.copy()

        if self.comm.Get_rank() == 0:
            y[self.local_dofs_ord[-1,-1]:] = y[self.local_dofs_ord[-1,-1]:] + y_b_global[act_boundary_dofs]

        return y

    def apply_M2(self, x:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        act_boundary_dofs = self.gbl_dofs_mngr.get_active_boundary_dofs()

        y = np.zeros(shape = (self.total_local_dofs))

        for s_ind, s_id in enumerate(self.process_subdomains):

            Lu_Kii = self.subdomains_Lu_Kii[s_ind]
            Piv_Kii = self.subdomains_Piv_Kii[s_ind]

            start, end = self.local_dofs_ord[s_ind,0], self.local_dofs_ord[s_ind,1]
            y[start:end] = scipy.linalg.lu_solve((Lu_Kii, Piv_Kii),x[start:end])

        n_act = act_boundary_dofs.size

        S0 = LinearOperator((n_act, n_act), matvec = self.apply_S0)
        S = LinearOperator((n_act, n_act), matvec = self.apply_S)

        if self.comm.Get_rank() == 0:
            tracker = residual_tracker(self.apply_S, True)
        else:
            tracker = residual_tracker(self.apply_S)

        b = np.zeros(shape = (n_act))

        if self.comm.Get_rank() == 0:
            b[:] = x[self.local_dofs_ord[-1,-1]:]

        self.comm.Bcast(b, root = 0)

        # if self.comm.Get_rank() == 0:
        #     print("2 norm b: ", np.linalg.norm(b))

        # y_sol, exit_code = gmres(S, b, M = S0, rtol = 1e-16, atol=1e-8, x0 = np.zeros(shape=(n_act,)), callback = tracker)

        y_sol, exit_code = cg(S, b, M = S0, rtol = 1e-12, maxiter= 500, x0 = np.zeros(shape=(n_act,)), callback = tracker)
        self.internal_tracker.append(tracker)

        if self.comm.Get_rank() == 0:
            y[self.local_dofs_ord[-1,-1]:] = y_sol

        return y

    def apply_M3(self, x:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        n_b = self.gbl_dofs_mngr.get_num_boundary_dofs()
        act_boundary_dofs = self.gbl_dofs_mngr.get_active_boundary_dofs()

        x_b = np.zeros(shape = (n_b))
        
        if self.comm.Get_rank() == 0:
            x_b[act_boundary_dofs] = x[self.local_dofs_ord[-1,-1]:]

        self.comm.Bcast(x_b, root = 0)

        y = x.copy()

        for s_ind, s_id in enumerate(self.process_subdomains):

            Uib = self.subdomains_Uib[s_ind]

            R = self.subdomains_R[s_ind]

            start, end = self.local_dofs_ord[s_ind,0], self.local_dofs_ord[s_ind,1]
            y[start:end] -= Uib @ x_b[R]

        return y
    
    def apply_M(self, x:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        y1 = self.apply_M1(x)
        y2 = self.apply_M2(y1)
        y3 = self.apply_M3(y2)

        return y3

    def assemble_f(self) -> None: 

        n_b = self.gbl_dofs_mngr.get_num_boundary_dofs()

        act_boundary_dofs = self.gbl_dofs_mngr.get_active_boundary_dofs()
        d_b_dofs = self.gbl_dofs_mngr.get_dirichlet_boundary_dofs()

        f = np.zeros(shape = (self.total_local_dofs))
        f_b_local = np.zeros(shape = (n_b))

        u_b_d = np.zeros((n_b,))
        u_b_d[d_b_dofs] = self.gbl_dofs_mngr.get_dirichlet_boundary_values()

        for s_ind, s_id in enumerate(self.process_subdomains):

            start, end = self.local_dofs_ord[s_ind,0], self.local_dofs_ord[s_ind,1]
            R = self.subdomains_R[s_ind]

            Kib = self.subdomains_Kib[s_ind]
            Kbb = self.subdomains_Kbb[s_ind]
            fi = self.subdomains_fi[s_ind]
            fb = self.subdomains_fb[s_ind]

            f[start:end] = fi - Kib @ u_b_d[R]
            f_b_local[R] += fb - Kbb @ u_b_d[R]

        f_b_global = np.zeros(shape = f_b_local.shape)
        self.comm.Reduce(f_b_local, f_b_global, MPI.SUM, root = 0)

        if self.comm.Get_rank() == 0:
            f[self.local_dofs_ord[-1,-1]:] = f_b_global[act_boundary_dofs]

        self.f = f
     
    def reconstruct_Us(
            self, 
            ut: npt.NDArray[np.float64],
        ) -> list[npt.NDArray[np.float64]]:
        
        """Reconstructs the full solution vector of every subdomain, once the
        multipliers lambda_ at the interfaces (the solution of the global dual
        problem), and the global primal solution uP have been computed.

        Args:
            uP (npt.NDArray[np.float64]): Global primal solution vector.
            lambda_ (npt.NDArray[np.float64]): Dual problem solution vector.


        Returns:
            list[npt.NDArray[np.float64]]: Vector of solutions for every subdomain.
        """
        n_b = self.gbl_dofs_mngr.get_num_boundary_dofs()
        act_boundary_dofs = self.gbl_dofs_mngr.get_active_boundary_dofs()
        d_dofs = self.gbl_dofs_mngr.get_dirichlet_boundary_dofs()

        u_b = np.zeros(shape = (n_b))

        if self.comm.Get_rank() == 0:
            u_b[act_boundary_dofs] = ut[self.local_dofs_ord[-1,-1]:]
            x = self.gbl_dofs_mngr.get_dirichlet_boundary_values()
            u_b[d_dofs] = self.gbl_dofs_mngr.get_dirichlet_boundary_values()

        self.comm.Bcast(u_b, root = 0)

        us = []

        for s_ind, s_id in enumerate(self.process_subdomains):

            subdomain = self.gbl_dofs_mngr.subdomains[s_ind]

            i_dofs = subdomain.interior_dofs
            b_dofs = subdomain.boundary_dofs
            total_dofs = subdomain.all_dofs.size

            start, end = self.local_dofs_ord[s_ind, 0], self.local_dofs_ord[s_ind, 1]
            R = self.gbl_dofs_mngr.create_R_inds(s_id)

            u = np.zeros((total_dofs,))
            u[b_dofs] = u_b[R]
            u[i_dofs] = ut[start:end]
            
            us.append(u)

        return us

    def reconstruct_from_ub(
            self, 
            ub: npt.NDArray[np.float64],
        ) -> list[npt.NDArray[np.float64]]:

        n_b = self.gbl_dofs_mngr.get_num_boundary_dofs()
        act_boundary_dofs = self.gbl_dofs_mngr.get_active_boundary_dofs()
        d_dofs = self.gbl_dofs_mngr.get_dirichlet_boundary_dofs()

        u_b = np.zeros(shape = (n_b))

        u_b[act_boundary_dofs] = ub
        u_b[d_dofs] = self.gbl_dofs_mngr.get_dirichlet_boundary_values()

        u = np.zeros(self.total_local_dofs)

        if self.comm.Get_rank() == 0:
            u[self.local_dofs_ord[-1,-1]:] = ub

        for s_ind, s_id in enumerate(self.process_subdomains):

            Uib = self.subdomains_Uib[s_ind]
            Lu_Kii = self.subdomains_Lu_Kii[s_ind]
            Piv_Kii = self.subdomains_Piv_Kii[s_ind]
            fi = self.subdomains_fi[s_ind]

            R = self.subdomains_R[s_ind]

            start, end = self.local_dofs_ord[s_ind,0], self.local_dofs_ord[s_ind,1]
            u[start:end] = scipy.linalg.lu_solve((Lu_Kii, Piv_Kii), fi) - Uib @ u_b[R]

        return u
    
    def get_ms(
        self
    ): 
        
        ms = []

        for s_ind, s_id in enumerate(self.process_subdomains):
            ms.append(self.gbl_dofs_mngr.subdomains[s_ind].M)
    
        return ms


class MatrixOperator:
    def __init__(self, assembler: Assembler):
        self.assembler = assembler

    def mult(self, A, x, y):
        x_array = x.getArray(readonly = True)
        y_array = self.assembler.apply_A(x_array)
        y.setArray(y_array)

class PreconditionerOperator:
    def __init__(self, assembler: Assembler):
        self.assembler = assembler

    def apply(self, pc, x, y):
        x_array = x.getArray(readonly = True)
        y_array = self.assembler.apply_M(x_array)
        y.setArray(y_array)


def reconstruct_solutions(
        u: npt.NDArray[np.float64],
        assembler: Assembler
    ) -> list[dolfinx.fem.Function]:
    """Reconstructs the solution function of every subdomain, starting from the
    multipliers lambda_ at the interfaces (the solution of the global dual
    problem).

    Args:
        gbl_dofs_mngr (GlobalDofsManager): Global dofs manager.
        lambda_ (npt.NDArray[np.float64]): Solution of the dual problem.
        assembler (Assembler): Fetidp operators manager.

    Returns:
        list[dolfinx.fem.Function]: List of functions describing the solution
            in every single subdomain. The FEM space of every function has the
            same structure as the one of the reference subdomain, but placed
            at its corresponding position.
    """
    us = assembler.reconstruct_Us(u)

    return us

def reconstruct_solutions_from_ub(
        ub: npt.NDArray[np.float64],
        assembler: Assembler
    ) -> list[dolfinx.fem.Function]:
    """Reconstructs the solution function of every subdomain, starting from the
    multipliers lambda_ at the interfaces (the solution of the global dual
    problem).

    Args:
        gbl_dofs_mngr (GlobalDofsManager): Global dofs manager.
        lambda_ (npt.NDArray[np.float64]): Solution of the dual problem.
        assembler (Assembler): Fetidp operators manager.

    Returns:
        list[dolfinx.fem.Function]: List of functions describing the solution
            in every single subdomain. The FEM space of every function has the
            same structure as the one of the reference subdomain, but placed
            at its corresponding position.
    """
    u = assembler.reconstruct_from_ub(ub)
    us = assembler.reconstruct_Us(u)

    return us


class BDDC(BaseSolver):

    def __init__(self, geometry, linear_pde, communicators, opts=None):

        self.us = None
        self.ms = None

        super().__init__(geometry, linear_pde, communicators, opts)

    def _setup_extra(self):

        start_time = MPI.Wtime()
        
        self.assembler = Assembler(self.gbl_dofs_mngr)

        comm = self.assembler.petsc_comm

        global_size = self.assembler.total_act_dofs
        local_size = self.assembler.total_local_dofs

        mat_op = MatrixOperator(self.assembler)
        pre_op = PreconditionerOperator(self.assembler)

        f_petsc = PETSc.Vec().createMPI((local_size, global_size), comm = comm)
        f_petsc.setArray(self.assembler.f)

        u_petsc = PETSc.Vec().createMPI((local_size, global_size), comm = comm)

        A_petsc = PETSc.Mat().create(comm = comm)
        A_petsc.setSizes([f_petsc.getSizes(), u_petsc.getSizes()])
        A_petsc.setType(PETSc.Mat.Type.PYTHON)
        A_petsc.setPythonContext(mat_op)

        ksp = PETSc.KSP().create()
        ksp.setOperators(A_petsc)
        ksp.setType('cg')
        ksp.setFromOptions()
        pc = ksp.getPC()    
        # pc.setType("none") 
        pc.setType(PETSc.PC.Type.PYTHON) 
        pc.setPythonContext(pre_op)
        # ksp.setTolerances(rtol=1e-14, atol=1e-14, max_it=3000)
        # ksp.setMonitor(monitor)

        assemble_time = MPI.Wtime() - start_time

        self.petsc_solver = ksp
        self.petsc_pc = pc
        self.f_petsc = f_petsc
        self.u_petsc = u_petsc

        self.stats["assemble time"] = assemble_time

    def solve(self):

        self.assembler.internal_tracker = []

        start_time = MPI.Wtime()

        self.petsc_pc.apply(self.f_petsc, self.u_petsc)
        self.solution = self.u_petsc

        solve_time = MPI.Wtime() - start_time

        rank = self.communicators.global_comm.Get_rank()

        self.stats["solve time"] = solve_time
        self.stats["total time"] = self.stats["setup time"] + self.stats["assemble time"] + solve_time
        self.stats["local iterations"] = sum(tracker.niter for tracker in self.assembler.internal_tracker) / len(self.assembler.internal_tracker),
        
        if rank == 0 and self.opts.get("print_stats", True):

            print("#### BDDC Solver ####\n")
            print("Number of iterations:")

            for iter, tracker in enumerate(self.assembler.internal_tracker):
                print(f"Iteration {iter}: {tracker.niter} iterations.")
            print("\n")

            print("Setup time: ", self.stats["setup time"])
            print("Assemble time: ", self.stats["assemble time"])
            print("Solve time: ", solve_time)
            print("Total time: ", self.stats["total time"])
            print("\n")

        # self.assembler.internal_tracker = []

        # start_time = MPI.Wtime()

        # self.petsc_solver.solve(self.f_petsc, self.u_petsc)
        # self.solution = self.u_petsc

        # solve_time = MPI.Wtime() - start_time

        # iterations = self.petsc_solver.getIterationNumber()
        # rank = self.communicators.global_comm.Get_rank()

        # self.stats["solve time"] = solve_time
        # self.stats["total time"] = self.stats["setup time"] + self.stats["assemble time"] + solve_time
        # self.stats["global iterations"] = iterations,
        # self.stats["local iterations"] = sum(tracker.niter for tracker in self.assembler.internal_tracker) / len(self.assembler.internal_tracker),
        
        # if rank == 0 and self.opts.get("print_stats", True):

        #     print("#### BDDC Solver ####\n")
        #     print(f"Number of outer iterations: {iterations}")
        #     print("Number of internal iterations:")

        #     for iter, tracker in enumerate(self.assembler.internal_tracker):
        #         print(f"Iteration {iter}: {tracker.niter} iterations.")
        #     print("\n")

        #     print("Setup time: ", self.stats["setup time"])
        #     print("Assemble time: ", self.stats["assemble time"])
        #     print("Solve time: ", solve_time)
        #     print("Total time: ", self.stats["total time"])
        #     print("\n")

    def get_solution(self):

        if self.solution is None:
            raise ValueError("No solution available. Please run the solver first.")
        
        if self.us is None:
            u_array = self.solution.getArray(readonly = True)
            self.us = reconstruct_solutions(u_array, self.assembler)

        return self.us