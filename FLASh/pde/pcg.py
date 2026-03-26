import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.linalg
import scipy.sparse
import scipy
import scipy.sparse.linalg

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

    def __init__(self, fun, print = False):
        self.niter = 0
        self.res = []
        self.fun = fun
        self.print = print

    def __call__(self, rk=None):
        self.niter += 1
        self.res.append(rk)
        # srk = self.fun(rk)
        # if self.print:
        #     print("2 norm of the residual: ", np.linalg.norm(rk))
        #     print("2 norm of the prec residual: ", np.linalg.norm(srk))
            # print("M norm: ", norm)

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
        self.assemble_global_ordering()
        self.assemble_A()
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

            fi = f[i_dofs]
            fb = f[b_dofs] 

            self.subdomains_Kii.append(Kii)
            self.subdomains_Kib.append(Kib)
            self.subdomains_Kbb.append(Kbb)
            self.subdomains_fi.append(fi)
            self.subdomains_fb.append(fb)

            R = self.gbl_dofs_mngr.create_R(s_id)

            self.subdomains_R.append(R)
    
    def assemble_global_ordering(self) -> None: 

        rank = self.comm.Get_rank()

        n_act_b = self.gbl_dofs_mngr.get_active_boundary_dofs().size

        N = self.gbl_dofs_mngr.get_num_subdomains()

        active_dofs_local = np.zeros(shape=(N), dtype=np.int32)
        local_number_of_dofs = np.zeros(shape=(self.comm.Get_size()), dtype=np.int32)

        # if rank == 0:
        #     local_number_of_dofs[rank] += n_act_b

        for s_ind, s_id in enumerate(self.process_subdomains):

            subdomain = self.gbl_dofs_mngr.subdomains[s_ind]
            i_dofs = subdomain.interior_dofs
            active_dofs_local[s_id] = i_dofs.size
            local_number_of_dofs[rank] += i_dofs.size

        active_dofs_global = np.zeros(shape = active_dofs_local.shape, dtype=np.int32)
        self.comm.Allreduce(active_dofs_local, active_dofs_global, MPI.SUM)

        number_of_dofs = np.zeros(shape = local_number_of_dofs.shape, dtype=np.int32)
        self.comm.Allreduce(local_number_of_dofs, number_of_dofs, MPI.SUM)

        self.offsets = np.cumsum([n_act_b] + list(number_of_dofs[:-1]))
        self.offset = self.offsets[rank]

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

        total_local_dofs = int(count)

        local_dofs_ordering = np.array(local_dofs_ordering)

        self.local_dofs_ord = local_dofs_ordering
        self.total_local_dofs = total_local_dofs
    
    def assemble_A(self) -> None:

        N = self.total_act_dofs

        A = scipy.sparse.lil_matrix((N, N))

        active_boundary = self.gbl_dofs_mngr.get_active_boundary_dofs()
        active_set = set(active_boundary)
        index_map = {p: i for i, p in enumerate(active_boundary)}

        for s_ind, s_id in enumerate(self.process_subdomains):

            start, end = self.local_dofs_ord[s_ind,0], self.local_dofs_ord[s_ind,1]
            offset_start = start + self.offset
            offset_end = end + self.offset

            R = self.subdomains_R[s_ind]  
            K_ii = self.subdomains_Kii[s_ind]
            Kib = self.subdomains_Kib[s_ind]  
            Kbb = self.subdomains_Kbb[s_ind]

            full_indices = [i for i, p in enumerate(R) if p in active_set]
            active_indices = [index_map[R[i]] for i in full_indices]

            A[offset_start:offset_end, offset_start:offset_end] += K_ii

            if active_indices:
                Kbb_sub = Kbb[np.ix_(full_indices, full_indices)]
                Kib_sub = Kib[:, full_indices]

                for i_local, i_global in enumerate(active_indices):

                    A[offset_start:offset_end, i_global] += Kib_sub[:, i_local][None,:].T
                    A[i_global, offset_start:offset_end] += Kib_sub[:, i_local][None,:]

                    for j_local, j_global in enumerate(active_indices):
                        A[i_global, j_global] += Kbb_sub[i_local, j_local]

        N = A.shape[0]

        A = A.tocsr()
        A_petsc = PETSc.Mat().createAIJ([N, N], comm=PETSc.COMM_WORLD)
        A_petsc.setUp()

        for i in range(N):
            row_start = A.indptr[i]
            row_end = A.indptr[i + 1]
            cols = A.indices[row_start:row_end]
            vals = A.data[row_start:row_end]
            if len(cols) > 0:
                A_petsc.setValues(i, cols, vals, addv=PETSc.InsertMode.ADD_VALUES)

        A_petsc.assemblyBegin()
        A_petsc.assemblyEnd()

        ksp = PETSc.KSP().create()
        ksp.setOperators(A_petsc)

        ksp.setType('cg')  

        pc = ksp.getPC()
        
        # Jacobi
        # pc.setType('jacobi')

        # Gauss-Seidel (aka SOR in PETSc)
        pc.setType('sor')
        # pc.setFactorShift(0.0)   # optional fine-tuning

        # ILU (incomplete LU, usually for sequential runs)
        # pc.setType('ilu')

        # AMG
        # pc.setType('gamg')  

        # Solver tolerances
        ksp.setTolerances(rtol=1e-8)
        ksp.setFromOptions()  
        ksp.setUp()

        self.A_sover = ksp
        self.A = A_petsc
        
    def assemble_f(self) -> None: 

        n_b = self.gbl_dofs_mngr.get_num_boundary_dofs()

        act_boundary_dofs = self.gbl_dofs_mngr.get_active_boundary_dofs()
        n_act_b = act_boundary_dofs.size
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

        N = self.total_act_dofs 

        f_petsc = PETSc.Vec().createMPI(N, comm=self.comm)
        f_petsc.setUp()

        for i_local in range(self.total_local_dofs):
            i_global = i_local + self.offset
            value = f[i_local]
            f_petsc.setValue(i_global, value, addv=PETSc.InsertMode.INSERT_VALUES)

        for i_global, i_local in enumerate(act_boundary_dofs):
            value = f_b_local[i_local]
            f_petsc.setValue(i_global, value, addv=PETSc.InsertMode.ADD_VALUES)

        f_petsc.assemblyBegin()
        f_petsc.assemblyEnd()

        self.f = f_petsc
     
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

            i_dofs = subdomain.interior_dofs
            b_dofs = subdomain.boundary_dofs
            total_dofs = subdomain.all_dofs.size

            start, end = self.local_dofs_ord[s_ind, 0], self.local_dofs_ord[s_ind, 1]
            R = self.gbl_dofs_mngr.create_R(s_id)

            u = np.zeros((total_dofs,))
            u[b_dofs] = u_b[R]
            u[i_dofs] = ut[start:end]
            
            us.append(u)

        return us


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


class PCG(BaseSolver):

    def __init__(self, geometry, linear_pde, communicators, opts=None):

        self.us = None
        self.ms = None

        super().__init__(geometry, linear_pde, communicators, opts)

    def _setup_extra(self):

        start_time = MPI.Wtime()
        
        self.assembler = Assembler(self.gbl_dofs_mngr)
        comm = self.assembler.petsc_comm

        assemble_time = MPI.Wtime() - start_time
        self.stats["assemble time"] = assemble_time

    def solve(self):

        start_time = MPI.Wtime()

        self.solution = self.assembler.A.getVecRight()

        self.assembler.A_sover.solve(self.assembler.f, self.solution)

        solve_time = MPI.Wtime() - start_time

        iterations = self.assembler.A_sover.getIterationNumber()
        rank = self.communicators.global_comm.Get_rank()

        self.stats["solve time"] = solve_time
        self.stats["total time"] = self.stats["setup time"] + self.stats["assemble time"] + solve_time
        self.stats["iterations"] = iterations
        

        if rank == 0 and self.opts.get("print_stats", True):
            print("#### PCG Solver ####\n")

            print(f"Number of subdomains: {self.gbl_dofs_mngr.get_num_subdomains()}.")
            print(f"Number of global active dofs: {self.assembler.total_act_dofs}.\n")

            print(f"Number of iterations: {iterations}")
            print("\n")

            print("Setup time: ", self.stats["setup time"])
            print("Assemble time: ", self.stats["assemble time"])
            print("Solve time: ", solve_time)
            print("Total time: ", self.stats["total time"])
            print("\n")

        pass

    def get_solution(self):

        if self.solution is None:
            raise ValueError("No solution available. Please run the solver first.")
        
        if self.us is None:

            local_array = self.solution.getArray(readonly=True)
            u_array_full = self.communicators.global_comm.allgather(local_array)
            u_array_full = np.concatenate(u_array_full)

            if self.communicators.global_comm.Get_rank() == 0:
                
                offset = self.assembler.offset
                size = self.assembler.total_local_dofs
                end = offset + size

                u_array = np.zeros(shape=(size+offset,), dtype=np.float64)
                u_array[:size] = u_array_full[offset:end]
                u_array[size:] = u_array_full[:offset]

            else:

                offset = self.assembler.offset
                size = self.assembler.total_local_dofs
                end = offset + size

                u_array = np.zeros(shape=(size,), dtype=np.float64)
                u_array[:] = u_array_full[offset:end]

            self.us = reconstruct_solutions(u_array, self.assembler)

        return self.us