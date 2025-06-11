import numpy as np
import numpy.typing as npt
import scipy.sparse
import scipy
import scipy.sparse.linalg

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.fem
import dolfinx

from global_dofs_manager import GlobalDofsManager

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import splu, spsolve

type SparseMatrix = scipy.sparse._csr.csr_matrix


class residual_tracker:

    """Class for tracking the iterative solver residuals and interations.
    """

    def __init__(self):
        self.niter = 0
        self.res = []

    def __call__(self, rk=None):
        self.niter += 1
        self.res.append(rk)

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

        self.process_subdomains = np.arange(rank, N, size)
        self.comm = comm
        self.petsc_comm = petsc_comm

    def assemble_subdomains(self) -> None:

        """Assembles and stores the subdomain operators.
        """
        N = self.gbl_dofs_mngr.get_num_subdomains()

        self.subdomains_Sbb = []
        self.subdomains_Lu_Sbb = []
        self.subdomains_Lu_Kii = []

        self.subdomains_Uib = []

        self.subdomains_Kii = []
        self.subdomains_Kib = []
        self.subdomains_Kbb = []
        self.subdomains_fi = []
        self.subdomains_fb = []

        self.subdomains_R = []

        for s_ind, s_id in enumerate(self.process_subdomains):

            subdomain = self.gbl_dofs_mngr.subdomains[s_ind]

            i_dofs = subdomain.get_interior_dofs()
            b_dofs = subdomain.get_boundary_dofs()

            K, f = subdomain.K, subdomain.f

            Ki = K[i_dofs]         
            Kii = Ki[:, i_dofs]     
            Kib = Ki[:, b_dofs]      
            Kb = K[b_dofs]          
            Kbb = Kb[:, b_dofs] 

            Lu_Kii = splu(Kii)

            Uib = scipy.sparse.csr_matrix(Lu_Kii.solve(Kib.toarray()))

            Sbb = Kbb - Kib.T @ Uib

            fi = f[i_dofs]
            fb = f[b_dofs] 

            self.subdomains_Sbb.append(Sbb)
            self.subdomains_Lu_Kii.append(Lu_Kii)

            self.subdomains_Uib.append(Uib)

            self.subdomains_Kii.append(Kii)
            self.subdomains_Kib.append(Kib)
            self.subdomains_Kbb.append(Kbb)
            self.subdomains_fi.append(fi)
            self.subdomains_fb.append(fb)

            R = self.gbl_dofs_mngr.create_R(s_id)

            self.subdomains_R.append(R)
    
    def assemble_global_ordering(self) -> None: 

        n_act_b = self.gbl_dofs_mngr.get_active_boundary_dofs().size

        N = self.gbl_dofs_mngr.get_num_subdomains()

        active_dofs_local = np.zeros(shape=(N), dtype=np.int32)

        for s_ind, s_id in enumerate(self.process_subdomains):

            subdomain = self.gbl_dofs_mngr.subdomains[s_ind]
            i_dofs = subdomain.get_interior_dofs()
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
            R_i = self.subdomains_R[s_ind]
            Sbb = self.subdomains_Sbb[s_ind]
            w = w + R_i.T @ Sbb @ R_i @ y

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
        subdomains_Lu_cSbb = []

        for s_ind, s_id in enumerate(self.process_subdomains):

            subdomain = self.gbl_dofs_mngr.subdomains[s_ind]

            R = self.gbl_dofs_mngr.create_R(s_id)
            Rc = self.gbl_dofs_mngr.create_Rc(s_id)
            D = self.gbl_dofs_mngr.create_D(s_id)
            C = subdomain.create_C()

            n_p = subdomain.get_num_local_primals()

            Sbb = self.subdomains_Sbb[s_ind]
            O = scipy.sparse.csr_matrix((n_p,n_p)) 
            cSbb = scipy.sparse.bmat([[Sbb, C.T],
                    [C, O]], format='csr')
            
            Lu_cSbb = splu(cSbb)

            subdomains_R.append(R)
            subdomains_Rc.append(Rc)
            subdomains_C.append(C)
            subdomains_D.append(D)
            subdomains_Lu_cSbb.append(Lu_cSbb)

        self.subdomains_R = subdomains_R
        self.subdomains_Rc = subdomains_Rc
        self.subdomains_C = subdomains_C
        self.subdomains_D = subdomains_D
        self.subdomains_Lu_cSbb = subdomains_Lu_cSbb

        subdomains_Psi = []
        subdomains_Kc = []

        for s_ind, s_id in enumerate(self.process_subdomains):

            subdomain = self.gbl_dofs_mngr.subdomains[s_ind]
            n_b = subdomain.get_boundary_dofs().size
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

            subdomains_Psi.append(scipy.sparse.csr_matrix(Psi))
            subdomains_Kc.append(Kc)

        self.subdomains_Psi = subdomains_Psi
        self.subdomains_Kc = subdomains_Kc

        num_primals = self.gbl_dofs_mngr.get_num_primals()
        active_primals = self.gbl_dofs_mngr.get_active_primal_dofs()

        Sc_local = scipy.sparse.csr_matrix((num_primals, num_primals))

        for s_ind, s_id in enumerate(self.process_subdomains):

            Rc = self.subdomains_Rc[s_ind]
            Kc = self.subdomains_Kc[s_ind]

            Sc_local = Sc_local + Rc.T @ Kc @ Rc

        # THis probably can be done better

        Sc_local = scipy.sparse.csr_matrix(Sc_local)
        Sc_local = Sc_local[active_primals,:]
        Sc_local = Sc_local[:,active_primals]
        Sc_local = Sc_local.tocoo()

        local_data = [Sc_local.data, Sc_local.row, Sc_local.col]
        full_data = self.comm.gather(local_data,root=0)

        if self.comm.Get_rank() == 0:
            data = np.concatenate([x[0] for x in full_data])
            rows = np.concatenate([x[1] for x in full_data])
            cols = np.concatenate([x[2] for x in full_data])

            Sc_global = scipy.sparse.csr_matrix(( data, (rows, cols) ) )
            self.Sc = splu(Sc_global)

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
                D @ R @ x1, 
                np.zeros(n_p)
            )

            y_local = y_local + R.T @ D.T @ w_local

        z_local = np.zeros(num_primals, dtype=np.float64)

        for s_ind, s_id in enumerate(self.process_subdomains):

            D = self.subdomains_D[s_ind]
            R = self.subdomains_R[s_ind]
            Rc = self.subdomains_Rc[s_ind]
            Psi = self.subdomains_Psi[s_ind]

            z_local = z_local + Rc.T @ Psi.T @ D @ R @ x1

        z_global = np.zeros(shape = z_local.shape)
        self.comm.Reduce(z_local, z_global, MPI.SUM, root = 0)

        if self.comm.Get_rank() == 0:
            z_global = self.solve_coarse_problem(z_global)

        self.comm.Bcast(z_global, root = 0)

        for s_ind, s_id in enumerate(self.process_subdomains):

            D = self.subdomains_D[s_ind]
            R = self.subdomains_R[s_ind]
            Rc = self.subdomains_Rc[s_ind]
            Psi = self.subdomains_Psi[s_ind]

            RT = R.T.tocsc()

            y_local = y_local + RT @ D @ Psi @ Rc @ z_global

        y_local = y_local[act_boundary_dofs]
        y_global = np.zeros(shape = y_local.shape)
        self.comm.Allreduce(y_local, y_global, MPI.SUM)

        return y_global 

    def solve_coarse_problem(self, x:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        num_primals = self.gbl_dofs_mngr.get_num_primals()
        active_primals = self.gbl_dofs_mngr.get_active_primal_dofs()

        y = np.zeros(num_primals, dtype=np.float64)
        y[active_primals] = self.Sc.solve(x[active_primals])

        return y

    def solve_subdomain_problem(
            self, 
            subdomain_local_id: np.int32,
            subdomain_id: np.int32, 
            y1:npt.NDArray[np.float64],
            y2:npt.NDArray[np.float64]
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:

        subdomain = self.gbl_dofs_mngr.subdomains[subdomain_local_id]
        nb = subdomain.get_boundary_dofs().size

        Lu_cSbb = self.subdomains_Lu_cSbb[subdomain_local_id]

        y = np.hstack((y1, y2))
        x = Lu_cSbb.solve(y)

        return x[:nb], x[nb:]
    
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
            y[start:end] = Kii @ x[start:end] + Kib @ R @ x_b
            y_b_local = y_b_local + R.T @ (Kib.T @ x[start:end] + Kbb @ R @ x_b)

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
            y_b_local = y_b_local - R.T @ Uib.T @ x[start:end]

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

            start, end = self.local_dofs_ord[s_ind,0], self.local_dofs_ord[s_ind,1]
            y[start:end] = Lu_Kii.solve(x[start:end])

        n_act = act_boundary_dofs.size

        S0 = LinearOperator((n_act, n_act), matvec = self.apply_S0)
        S = LinearOperator((n_act, n_act), matvec = self.apply_S)

        tracker = residual_tracker()

        b = np.zeros(shape = (n_act))

        if self.comm.Get_rank() == 0:
            b[:] = x[self.local_dofs_ord[-1,-1]:]

        self.comm.Bcast(b, root = 0)

        y_sol, exit_code = cg(S, b, M = S0, rtol = 1e-16, x0 = np.zeros(shape=(n_act,)), callback = tracker)
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
            y[start:end] = y[start:end] - Uib @ R @ x_b

        return y
 
    def apply_M(self, x:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        y1 = self.apply_M1(x)
        y2 = self.apply_M2(y1)
        y3 = self.apply_M3(y2)

        return y3

    def apply_inverse_diagonal(self, x:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        y = np.zeros(shape = (self.total_local_dofs))

        for s_ind, s_id in enumerate(self.process_subdomains):

            Kii = self.subdomains_Kii[s_ind]

            start, end = self.local_dofs_ord[s_ind,0], self.local_dofs_ord[s_ind,1]
            y[start:end] = x[start:end] / Kii.diagonal()

        if self.comm.Get_rank() == 0:
            a_dofs = self.gbl_dofs_mngr.get_active_boundary_dofs()
            weights = self.gbl_dofs_mngr.get_boundary_weights()
            y[self.local_dofs_ord[-1,-1]:] = x[self.local_dofs_ord[-1,-1]:] / weights[a_dofs]

        return y

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

            f[start:end] = fi - Kib @ R @ u_b_d
            f_b_local += R.T @ (fb - Kbb @ R @ u_b_d)

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
            u_b[d_dofs] = self.gbl_dofs_mngr.get_dirichlet_boundary_values()

        self.comm.Bcast(u_b, root = 0)

        us = []

        for s_ind, s_id in enumerate(self.process_subdomains):

            subdomain = self.gbl_dofs_mngr.subdomains[s_ind]

            i_dofs = subdomain.get_interior_dofs()
            b_dofs = subdomain.get_boundary_dofs()
            total_dofs = subdomain.get_all_dofs(get_active = False).size

            start, end = self.local_dofs_ord[s_ind, 0], self.local_dofs_ord[s_ind, 1]
            R = self.gbl_dofs_mngr.create_R(s_id)

            u = np.zeros((total_dofs,))
            u[b_dofs] = R @ u_b
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
            fi = self.subdomains_fi[s_ind]

            R = self.subdomains_R[s_ind]

            start, end = self.local_dofs_ord[s_ind,0], self.local_dofs_ord[s_ind,1]
            u[start:end] = Lu_Kii.solve(fi) - Uib @ R @ u_b

        return u
    
        

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

class PreconditionerOperator_2:
    def __init__(self, assembler: Assembler):
        self.assembler = assembler

    def apply(self, pc, x, y):
        x_array = x.getArray(readonly = True)
        y_array = self.assembler.apply_inverse_diagonal(x_array)
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

def write_solutions(
    us,
    assembler: Assembler
) -> None:
    
    for s_ind, s_id in enumerate(assembler.process_subdomains):

        subdomain = assembler.gbl_dofs_mngr.subdomains[s_ind]
        subdomain.write_solution(us[s_ind], f"subdomain_{s_id}")


def bddc_solver(
        geometry: dict,
        parameter_function, 
        linear_pde,
        communicators,
        opts: dict = None
) -> None:
    """Solves the Poisson problem with N subdomains per direction using
    a preconditioned conjugate gradient FETI-DP solver.

    The Dirichlet preconditioner is used.

    Every subdomain is considered to have n elements per direction, and
    the input degree is used for discretizing the solution.

    The generated solutions are written to the folder "results" as VTX
    folders named "subdomain_i.pb", with i running from 0 to N-1.
    One file per subdomain. Thy can be visualized using ParaView.

    Args:
        n (list[int]): Number of elements per direction in every single
            subdomain.
        N (list[int]): Number of subdomains per direction.
        degree (int): Discretization space degree.
    """

    N = geometry["N"]
    assert N[0] * N[1] > 1, "Invalid number of subdomains."
    assert geometry["degree"] > 0, "Invalid degree."
    
    opts = opts or {}

    return_stats = opts.get("return_stats", False)
    print_stats = opts.get("print_stats", True)
    write_solution = opts.get("write_solution", False)
    make_plots = opts.get("make_plots", False)
    compute_error = opts.get("compute_error", False)
    compute_cond = opts.get("compute_cond", False)

    stats = {}

    start_time = MPI.Wtime()

    gbl_dofs_mngr = GlobalDofsManager.create_rectangle(geometry, parameter_function, linear_pde, communicators)

    setup_time = MPI.Wtime() - start_time
    start_time = MPI.Wtime()
    
    assembler = Assembler(gbl_dofs_mngr)

    rank = communicators.global_comm.Get_rank()
    comm = assembler.petsc_comm

    global_size = assembler.total_act_dofs
    local_size = assembler.total_local_dofs

    mat_op = MatrixOperator(assembler)
    pre_op = PreconditionerOperator(assembler)

    f_petsc = PETSc.Vec().createMPI((local_size, global_size), comm = comm)
    f_petsc.setArray(assembler.f)

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

    assembler.internal_tracker = []

    assemble_time = MPI.Wtime() - start_time
    start_time = MPI.Wtime()

    ksp.solve(f_petsc, u_petsc)

    solve_time = MPI.Wtime() - start_time
    start_time = MPI.Wtime()

    u_array = u_petsc.getArray(readonly = True)
    iterations = ksp.getIterationNumber()

    if make_plots or write_solution or compute_error:
        us = reconstruct_solutions(u_array, assembler)

    if make_plots:
        gbl_dofs_mngr.plot_solution(us)

    if write_solution:
        write_solutions(us, assembler)

    if compute_error:
        error = gbl_dofs_mngr.compute_error(us)
        stats["error"] = error

    postprocess_time = MPI.Wtime() - start_time

    stats["solve time"] = solve_time
    stats["setup time"] = setup_time
    stats["assemble time"] = assemble_time
    stats["total time"] = setup_time+assemble_time+solve_time+postprocess_time

    stats["global iterations"] = iterations,
    stats["local iterations"] = sum(tracker.niter for tracker in assembler.internal_tracker) / len(assembler.internal_tracker),
    
    if compute_cond:

        def apply_composed(x):
            return assembler.apply_S0(assembler.apply_S(x))

        n = assembler.gbl_dofs_mngr.get_active_boundary_dofs().size

        A = LinearOperator((n, n), matvec = apply_composed)
        
        λ_max = scipy.sparse.linalg.eigsh(A, k=2, which='LA', return_eigenvectors=False)
        λ_min = scipy.sparse.linalg.eigsh(A, k=2, which='SA', return_eigenvectors=False)

        condition_number = λ_max[-1] / λ_min[-1]
        stats["condition number"] = condition_number

        
    if rank == 0 and print_stats:
        print("#### BDDC Solver ####\n")
        print(f"Number of outer iterations: {iterations}")
        print("Number of internal iterations:")
        for iter, tracker in enumerate(assembler.internal_tracker):
            print(f"Iteration {iter}: {tracker.niter} iterations.")
        print("\n")
        print("Setup time: ", setup_time)
        print("Assemble time: ", assemble_time)
        print("Solve time: ", solve_time)
        print("Postprocess time: ", postprocess_time)
        print("Total time: ", setup_time+assemble_time+solve_time+postprocess_time)
        print("\n")

        if compute_error:

            print(f"L2 relative error: {error}\n")

        if compute_cond:

            print("Smallest eigenvalue:", λ_min[-1])
            print("Largest eigenvalue:", λ_max[-1])
            print("Condition number:", condition_number)
            print("\n")


    if return_stats:
        return stats

