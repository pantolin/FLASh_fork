from abc import ABC, abstractmethod

from mpi4py import MPI
from FLASh.mesh.global_dofs_manager import GlobalDofsManager    

def write_solutions(
    us,
    gbl_dofs_mngr: GlobalDofsManager
) -> None:
    
    for s_ind, s_id in enumerate(gbl_dofs_mngr.process_subdomains):

        subdomain = gbl_dofs_mngr.subdomains[s_ind]
        subdomain.write_solution(us[s_ind], f"subdomain_{s_id}")

class BaseSolver(ABC):

    def __init__(
            self, 
            geometry, 
            linear_pde, 
            communicators,
            opts = None
    ):
        
        self.geometry = geometry
        self.linear_pde = linear_pde
        self.communicators = communicators
        self.opts = opts or {}
        self.stats = {}
        self.solution = None

    def setup(self):
        
        start_time = MPI.Wtime()

        gbl_dofs_mngr = GlobalDofsManager.create_rectangle(
            self.geometry, 
            self.linear_pde, 
            self.communicators,
            opts = self.opts.get("global_dofs_manager_opts", None)
        )

        self.gbl_dofs_mngr = gbl_dofs_mngr

        self._setup_extra()

        setup_time = MPI.Wtime() - start_time
        self.stats["setup time"] = setup_time

    @abstractmethod
    def _setup_extra(self):
        pass

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def get_solution(self):
        pass
    
    def get_stats(self):
        return self.stats

    def plot_solution(self):

        if self.solution is None:
            raise ValueError("No solution available. Please run the solver first.")
        
        self.gbl_dofs_mngr.plot_solution(self.get_solution())

    def plot_f(self):

        self.gbl_dofs_mngr.plot_solution(self.gbl_dofs_mngr.get_fs())

    def write_solution(self):

        if self.solution is None:
            raise ValueError("No solution available. Please run the solver first.")
        
        write_solutions(self.get_solution())