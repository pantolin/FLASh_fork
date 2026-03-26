"""Solver base classes and helpers.

This module provides the `BaseSolver` abstract class used by the FLASh
framework, as well as helper routines for writing solution fields.
"""

from abc import ABC, abstractmethod

from mpi4py import MPI
from FLASh.mesh.global_dofs_manager import GlobalDofsManager    




class BaseSolver(ABC):
    """Base class for PDE solvers in the FLASh framework.

    Subclasses must implement `_setup_extra`, `solve`, and `get_solution`. The
    base class provides common setup utilities, plotting helpers, and statistics.

    Parameters
    ----------
    geometry:
        Geometry object describing the computational domain.
    linear_pde:
        PDE model instance (e.g., `Elasticity`).
    communicators:
        MPI communicator wrapper (see `FLASh.utils.Communicators`).
    opts: dict, optional
        Solver configuration options.
    """

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

        setup_time = MPI.Wtime() - start_time
        self.stats["setup time"] = setup_time

        self._setup_extra()

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

    def plot_stress(self):

        if self.solution is None:
            raise ValueError("No solution available. Please run the solver first.")
        
        self.gbl_dofs_mngr.plot_stress(self.get_solution())

    def plot_f(self):

        self.gbl_dofs_mngr.plot_solution(self.gbl_dofs_mngr.get_fs())

    def write_solution(self, path="results"):
        """
        Write the solution to files in the specified directory.
        Ensures the directory exists.
        Parameters
        ----------
        path : str, optional
            Directory to save the results in (default is 'results').
        """
        import os
        from pathlib import Path
        if self.solution is None:
            raise ValueError("No solution available. Please run the solver first.")
        Path(path).mkdir(exist_ok=True, parents=True)
        us = self.get_solution()
        for s_ind, s_id in enumerate(self.gbl_dofs_mngr.process_subdomains):
            subdomain = self.gbl_dofs_mngr.subdomains[s_ind]
            subdomain.write_solution(us[s_ind], f"subdomain_{s_id}", path)