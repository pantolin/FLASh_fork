"""
Tests the performance and scalability of the FLASh method on larger problems.
Intended for benchmarking parallel efficiency and computational cost.

Reproduces Figure 18 and the scalability statistics reported in Section 5.1.3,
"Scalability of the method".
"""

import resource
import platform

import numpy as np
from pathlib import Path

from mpi4py import MPI

from FLASh.utils import Communicators

from FLASh.rom import MDEIM

from FLASh.mesh import (
    GlobalDofsManager,
    SplineGeometry,
    gyroid
)
from FLASh.pde import (
    Elasticity,
    BDDC,
    Cholesky
)

dtype = np.float64

import h5py

from _paths import RESULTS_DIR, ROM_DATA_DIR


def collect_scalability_report(solver, communicators, mem_peak_mb, compute_cond):
    """Gather the reviewer-requested scalability quantities for a single solve.

    All collective (reduce/gather) calls are made on every rank; the returned
    dict is populated on rank 0 only (None on the other ranks).
    """

    comm = communicators.global_comm
    rank = comm.Get_rank()
    size = comm.Get_size()

    gdm       = solver.gbl_dofs_mngr
    assembler = solver.assembler
    cm        = gdm.coarse_mesh

    ### Grid counts ###

    n_domains = gdm.get_num_subdomains()
    n_nodes   = len(cm.vertex_coordinates)
    n_edges   = len(cm.edge_vertex_conn)

    # internal edges: shared by exactly 2 subdomains
    n_internal_edges = sum(1 for ec in cm.edge_cell_conn if len(ec) == 2)

    # internal nodes: not connected to any boundary edge
    boundary_edge_set = set(e for e, ec in enumerate(cm.edge_cell_conn) if len(ec) == 1)
    boundary_nodes    = set()
    for e in boundary_edge_set:
        boundary_nodes.update(cm.edge_vertex_conn[e])
    n_internal_nodes = n_nodes - len(boundary_nodes)

    ### DOF counts ###

    # Redundant DOFs: sum of local K sizes (interface DOFs counted per adjacent subdomain)
    local_dofs_redundant = sum(s.K.shape[0] for s in gdm.subdomains)
    n_dofs_redundant = comm.reduce(local_dofs_redundant, op=MPI.SUM, root=0)

    # Interior DOFs: summed across all subdomains (no overlap)
    local_dofs_internal = sum(s.interior_dofs.size for s in gdm.subdomains)
    n_dofs_internal = comm.reduce(local_dofs_internal, op=MPI.SUM, root=0)

    # Skeleton (interface) DOFs — size of the Schur complement system CG solves
    n_dofs_skeleton = gdm.get_active_boundary_dofs().size

    # Non-redundant total = interior (no overlap) + skeleton (counted once)
    n_dofs_non_redundant = assembler.total_act_dofs
    n_total_boundary     = gdm.get_num_boundary_dofs()
    n_dirichlet_boundary = n_total_boundary - n_dofs_skeleton

    # Coarse (primal) DOFs
    n_coarse_dofs = len(gdm.get_active_primal_dofs())

    ### Memory ###

    all_mem       = comm.gather(mem_peak_mb, root=0)
    mem_aggregate = sum(all_mem) if rank == 0 else None
    mem_max_rank  = max(all_mem) if rank == 0 else None

    ### Coarse matrix conditioning (serial only, sparse eigsh) ###

    cond_coarse        = float('nan')
    cond_coarse_scaled = float('nan')
    if compute_cond and rank == 0 and size == 1:
        try:
            import scipy.sparse as sp
            from scipy.sparse.linalg import eigsh, LinearOperator

            n_c = n_coarse_dofs
            indptr  = [0]
            indices = []
            data    = []
            for row in range(n_c):
                cols, vals = assembler.Sc.getRow(row)
                indices.extend(cols)
                data.extend(vals)
                indptr.append(len(indices))

            Sc_scipy = sp.csr_matrix(
                (np.array(data), np.array(indices, dtype=np.int32),
                 np.array(indptr, dtype=np.int32)),
                shape=(n_c, n_c)
            )

            diag       = np.array(Sc_scipy.diagonal())
            d_inv_sqrt = 1.0 / np.sqrt(np.maximum(diag, 1e-300))
            d_sqrt     = np.sqrt(np.maximum(diag, 1e-300))

            x_w = assembler.Sc.createVecRight()
            y_w = assembler.Sc.createVecRight()

            def matvec(x):
                x_w.setArray(x); assembler.Sc.mult(x_w, y_w)
                return y_w.getArray().copy()

            def matvec_inv(x):
                x_w.setArray(x); assembler.Sc_sover.solve(x_w, y_w)
                return y_w.getArray().copy()

            def matvec_scaled(x):
                return d_inv_sqrt * matvec(d_inv_sqrt * x)

            def matvec_scaled_inv(x):
                return d_sqrt * matvec_inv(d_sqrt * x)

            A        = LinearOperator((n_c, n_c), matvec=matvec,            dtype=np.float64)
            A_inv    = LinearOperator((n_c, n_c), matvec=matvec_inv,        dtype=np.float64)
            A_sc     = LinearOperator((n_c, n_c), matvec=matvec_scaled,     dtype=np.float64)
            A_sc_inv = LinearOperator((n_c, n_c), matvec=matvec_scaled_inv, dtype=np.float64)

            lmax        = eigsh(A,        k=1, which='LM', return_eigenvectors=False)[0]
            lmin_inv    = eigsh(A_inv,    k=1, which='LM', return_eigenvectors=False)[0]
            lmax_sc     = eigsh(A_sc,     k=1, which='LM', return_eigenvectors=False)[0]
            lmin_inv_sc = eigsh(A_sc_inv, k=1, which='LM', return_eigenvectors=False)[0]

            cond_coarse        = lmax * lmin_inv
            cond_coarse_scaled = lmax_sc * lmin_inv_sc

        except Exception as e:
            print(f"  [matrix export failed: {e}]")

    if rank != 0:
        return None

    return {
        "n_domains": n_domains,
        "n_nodes": n_nodes,
        "n_internal_nodes": n_internal_nodes,
        "n_edges": n_edges,
        "n_internal_edges": n_internal_edges,
        "n_dofs_non_redundant": n_dofs_non_redundant,
        "n_dofs_redundant": n_dofs_redundant,
        "n_dofs_internal": n_dofs_internal,
        "n_dofs_skeleton": n_dofs_skeleton,
        "n_dirichlet_dofs": n_dirichlet_boundary,
        "n_coarse_dofs": n_coarse_dofs,
        "cond_coarse": cond_coarse,
        "cond_coarse_scaled": cond_coarse_scaled,
        "mem_aggregate_mb": mem_aggregate,
        "mem_max_rank_mb": mem_max_rank,
        "mem_mean_rank_mb": mem_aggregate / size,
    }

if __name__ == "__main__":         

    communicators = Communicators()

    ### Load ROM models ###
    
    epsilon_min = 0.1
    epsilon_max = 0.9

    n_rom = 2
    p_rom = 6
    d_rom = 4

    p0 = np.array([epsilon_min] * d_rom)
    p1 = np.array([epsilon_max] * d_rom)

    k_core_model = MDEIM(n_rom, p_rom, p0, p1)
    k_core_model.set_up_from_files(str(ROM_DATA_DIR / "schwarz_diamond_3" / "K_core"))

    m_core_model = MDEIM(n_rom, p_rom, p0, p1)
    m_core_model.set_up_from_files(str(ROM_DATA_DIR / "schwarz_diamond_3" / "M_core"))

    bm_core_model = MDEIM(n_rom, p_rom, p0, p1)
    bm_core_model.set_up_from_files(str(ROM_DATA_DIR / "schwarz_diamond_3" / "bM_core"))

    K_core_full = np.load(str(ROM_DATA_DIR / "schwarz_diamond_3" / "K_core" / "full_array.npy"))

    ### Set geometry options ###

    P0 = np.array([0.0, 0.0])
    P1 = np.array([1.0, 1.0])

    def map(x, y, r = [0.6, 1.0], theta = [1.5, 2.0]):

        tx = theta[0] + (theta[1]-theta[0])*x
        ty = r[0] + (r[1]-r[0])*y

        return np.stack([ty*np.cos(np.pi*tx), ty*np.sin(np.pi*tx), 0*tx], axis=-1)

    def parameter_function(X):
        val = 0.9 - 0.8*X[0]
        return np.clip(val, 0.1, 0.9)
    
    basis_degree = 8
    spline_degree = 2
    
    geometry_opts = {
            "basis_degree": basis_degree,
            "spline_degree": spline_degree,
            "periodic": False
        }

    ### Set solver source and boundary conditions ###

    def source(X):
        return (0.0+0.0*X[0], 0.0+0.0*X[0])
    
    def h_bc(X):
        return (0.0+0.0*X[0], 0.0+0.0*X[0])

    def nh_bc(X):
        return (0.0+0.0*X[0], -0.1+0.0*X[0])
    
    exterior_bc = [
        (
            0, 
            h_bc, 
            lambda x: np.isclose(x[0], P0[0]), 
            0
        ),
        (
            1, 
            nh_bc, 
            lambda x: np.isclose(x[0], P1[0]), 
            0
        )
    ]

    ### Simulation paramters ###

    stabilization = 5e-4
    stabilize = True

    ### Set pde problems ###

    elasticity_pde = Elasticity(
        exterior_bc = exterior_bc,
        source = source,
        E = 5,
        nu = 0.25,
        K_model = k_core_model,
        M_model = m_core_model,
        bM_model = bm_core_model,
        K_full_core = K_core_full
    )

    ### Set solver options ###

    sbdmn_opts = {
        "stabilize" : True,
        "stabilization": stabilization,
        "assemble" : True
    }

    gdm_opts = {
        "subdomain_opts" : sbdmn_opts
    }

    opts = {
        "global_dofs_manager_opts": gdm_opts
    }

    stats = []
    reports = []

    # macOS returns bytes, Linux returns KB
    mem_scale = 1024**2 if platform.system() == 'Darwin' else 1024

    i_max = 51
    i_step = 5

    for i in range(1, i_max, i_step):

        ### Create geometry ###

        n = [4*i, 2*i]

        P0 = np.array([0.0, 0.0])
        P1 = np.array([1.0, 1.0])

        knots_x = [P0[0]]*spline_degree + list(np.linspace(P0[0],P1[0],n[0]+1)) + [P1[0]]*spline_degree
        knots_y = [P0[1]]*spline_degree + list(np.linspace(P0[1],P1[1],n[1]+1)) + [P1[1]]*spline_degree

        geometry = SplineGeometry.interpolate_map(
            [knots_x, knots_y],
            map,
            gyroid.SchwarzDiamond().make_function(),
            geometry_opts
        )

        geometry.coarse_mesh.set_parameter_field_from_function(parameter_function)
        
        ### Solve with BDDC ###

        mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        solver = BDDC(geometry, elasticity_pde, communicators, opts = opts)
        solver.setup()

        mem_peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / mem_scale

        solver.solve()
        stats.append(solver.get_stats())

        ### Collect detailed scalability report ###

        report = collect_scalability_report(
            solver, communicators, mem_peak_mb, compute_cond=True
        )
        if report is not None:
            reports.append(report)

    if communicators.global_comm.Get_rank() == 0:

        number_of_subdomains = 8 * (np.arange(1, i_max, i_step) ** 2)

        iters = np.array([stat["iterations"][0] for stat in stats])
        setup_time = np.array([stat["setup time"] for stat in stats])
        assemble_time = np.array([stat["assemble time"] for stat in stats])
        solve_time = np.array([stat["solve time"] for stat in stats])

        # Reviewer-requested scalability quantities (from test_5_bis), one per mesh
        def report_array(key):
            return np.array([rep[key] for rep in reports])

        folder = RESULTS_DIR / "test_5"
        folder.mkdir(parents=True, exist_ok=True)

        file_path = folder / "data.h5"

        with h5py.File(file_path, "w") as f:

            f.create_dataset("iters", data=iters)
            f.create_dataset("setup_time", data=setup_time)
            f.create_dataset("assemble_time", data=assemble_time)
            f.create_dataset("solve_time", data=solve_time)

            f.create_dataset("number_of_subdomains", data=number_of_subdomains)

            ### Grid counts ###
            f.create_dataset("n_domains", data=report_array("n_domains"))
            f.create_dataset("n_nodes", data=report_array("n_nodes"))
            f.create_dataset("n_internal_nodes", data=report_array("n_internal_nodes"))
            f.create_dataset("n_edges", data=report_array("n_edges"))
            f.create_dataset("n_internal_edges", data=report_array("n_internal_edges"))

            ### DOF counts ###
            f.create_dataset("n_dofs_non_redundant", data=report_array("n_dofs_non_redundant"))
            f.create_dataset("n_dofs_redundant", data=report_array("n_dofs_redundant"))
            f.create_dataset("n_dofs_internal", data=report_array("n_dofs_internal"))
            f.create_dataset("n_dofs_skeleton", data=report_array("n_dofs_skeleton"))
            f.create_dataset("n_dirichlet_dofs", data=report_array("n_dirichlet_dofs"))
            f.create_dataset("n_coarse_dofs", data=report_array("n_coarse_dofs"))

            ### Coarse matrix conditioning (NaN when run in parallel) ###
            f.create_dataset("cond_coarse", data=report_array("cond_coarse"))
            f.create_dataset("cond_coarse_scaled", data=report_array("cond_coarse_scaled"))

            ### Memory (peak RSS, MB) ###
            f.create_dataset("mem_aggregate_mb", data=report_array("mem_aggregate_mb"))
            f.create_dataset("mem_max_rank_mb", data=report_array("mem_max_rank_mb"))
            f.create_dataset("mem_mean_rank_mb", data=report_array("mem_mean_rank_mb"))

        print(f"Saved to {file_path}")

    

    