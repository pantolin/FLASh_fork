import sys
sys.stdout.isatty = lambda: True

import numpy as np

from FLASh.utils import (
    Communicators
)

from FLASh.mesh import (
    GlobalDofsManager,
    gyroid
)

from FLASh.pde import (
    Elasticity,
    BDDC,
)

from FLASh.rom import (
    MDEIM
)

from example_5_utils import (
    WrenchGeometry
)

from _paths import EXAMPLES_ROOT, ROM_DATA_DIR, RESULTS_DIR

from scipy.io import loadmat

if __name__ == "__main__":        

    
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

    ##3    

    data  = loadmat(str(EXAMPLES_ROOT / "wrench_example" / "Wrench.mat"))

    nodes        = data['nodes']
    eleme_coefs  = data['eleme_coefs']

    conn_eleme_nodes  = data['conn_eleme_nodes'] - 1
    conn_eleme_edges  = data['conn_eleme_edges'] - 1
    conn_edges_nodes  = data['conn_edges_nodes'] - 1

    conn_eleme_nodes = conn_eleme_nodes[:,[0, 2, 1, 3]]
    conn_eleme_edges = conn_eleme_edges[:,[2, 0, 1, 3]]

    nodes_ex  = np.squeeze(data['nodes_ex'] - 1)

    edges_neu  = np.squeeze(data['edges_neu'] - 1)
    edges_dir  = np.squeeze(data['edges_dir'] - 1)

    communicators = Communicators()

    geometry_opts = {
        "basis_degree": 8,
        "spline_degree": 2,
    }

    geometry = WrenchGeometry(
        conn_eleme_nodes,
        conn_eleme_edges,
        conn_edges_nodes,
        nodes,
        eleme_coefs,
        gyroid.SchoenIWP().make_function(),
        geometry_opts
    )
    
    def parameter_function(X):
        num_points = X.shape[1]
        random_vals = 0.1 + 0.8 * np.random.rand(num_points)
        return random_vals
    

    geometry.coarse_mesh.set_parameter_field_from_function(parameter_function)



    nodes_dir = np.unique(np.array(geometry.coarse_mesh.edge_vertex_conn)[edges_dir].flatten())

    points_dir = np.vstack([
        geometry.coarse_mesh.vertex_coordinates[nodes_dir],
        geometry.coarse_mesh.edge_coordinates[edges_dir]
    ]).T

    nodes_neu = np.unique(np.array(geometry.coarse_mesh.edge_vertex_conn)[edges_neu].flatten())

    points_neu = np.vstack([
        geometry.coarse_mesh.vertex_coordinates[nodes_neu],
        geometry.coarse_mesh.edge_coordinates[edges_neu]
    ]).T

    def points_in_array_fast(x: np.ndarray, y: np.ndarray, tol=1e-8) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(2, 1)
            single_point = True
        else:
            single_point = False

        diff = x[:, :, None] - y[:, None, :]  
        dist = np.linalg.norm(diff, axis=0)    
        mask = np.any(dist < tol, axis=1)      

        if single_point:
            return mask[0]  
        return mask
    

    def h_bc(X):
        return (0+0*X[0], 0+0*X[0])
    
    def nh_bc(X):
        return (0+0*X[0], -0.001+0*X[0])

    def source(X):
        return (0.0+0.0*X[0], 0.0+0.0*X[0])

    exterior_bc = [
        (
            0, 
            h_bc, 
            lambda x: points_in_array_fast(x[:2], points_dir[:2]), 
            0
        ),
        (
            1, 
            nh_bc, 
            lambda x: points_in_array_fast(x[:2], points_neu[:2]), 
            1
        ),
    ]
    
    elasticity_pde = Elasticity(
        exterior_bc = exterior_bc,
        source = source,
        E = 5,
        nu = 0.25
    )

    elasticity_pde_rom = Elasticity(
        exterior_bc = exterior_bc,
        source = source,
        E = 5,
        nu = 0.25,
        K_model = k_core_model,
        M_model = m_core_model,
        bM_model = bm_core_model,
        K_full_core = K_core_full
    )

    sbdmn_opts = {
        "stabilize" : True,
        "stabilization": 1e-5,
        "assemble" : True
    }

    gdm_opts = {
        "subdomain_opts" : sbdmn_opts
    }

    opts = {
        "global_dofs_manager_opts": gdm_opts
    }

    GlobalDofsManager.plot(geometry, communicators)

    solver = BDDC(geometry, elasticity_pde_rom, communicators, opts = opts)
    solver.setup()
    solver.solve()
    solver.plot_solution()
    solver.write_solution(str(RESULTS_DIR / "wrench_example_rom"))

    rom_solution = solver.get_solution()

    sbdmn_opts = {
        "stabilize" : False,
        "assemble" : True
    }

    gdm_opts = {
        "subdomain_opts" : sbdmn_opts
    }

    opts = {
        "global_dofs_manager_opts": gdm_opts
    }

    solver = BDDC(geometry, elasticity_pde, communicators, opts = opts)
    solver.setup()
    solver.solve()
    solver.plot_solution()
    solver.write_solution(str(RESULTS_DIR / "wrench_example"))

    solution = solver.get_solution()

    if communicators.global_comm.Get_rank() == 0:
        print("Solution error: ", solver.gbl_dofs_mngr.compute_error(rom_solution, solution))

    








