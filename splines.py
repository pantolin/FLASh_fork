import splipy as sp
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from functools import reduce
from matplotlib.patches import Circle

class BSpline2D:

    def __init__(self, n: list[int], degree: int) -> None:

        self._n = n
        self._degree = degree

        knots_x = [0]*degree + list(np.linspace(0,1,n[0]+1)) + [1]*degree
        knots_y = [0]*degree + list(np.linspace(0,1,n[1]+1)) + [1]*degree

        self._knots = [knots_x, knots_y]

        basis_x = sp.BSplineBasis(degree+1, knots_x) 
        basis_y = sp.BSplineBasis(degree+1, knots_y)

        self._BSpline_basis = [basis_x, basis_y]
        self._number_basis_per_direction = [n[0]+degree, n[1]+degree]
        self._total_number_basis = (n[0]+degree) * (n[1]+degree)

    def evaluate(self, x: np.ndarray) -> np.ndarray:

        fx = self._BSpline_basis[0].evaluate(x[:,0])
        fy = self._BSpline_basis[1].evaluate(x[:,1])

        f = (fy[:,:,None]*fx[:,None,:]).reshape(x.shape[0], self._total_number_basis)
        
        return f
    
    def get_lagrange_extraction(self, x: np.ndarray) -> np.ndarray:

        return sc.sparse.csr_matrix(self.evaluate(x).T)

    def get_lagrange_extraction_connection(self, x: np.ndarray) -> np.ndarray:

        lagrange_extraction = self.get_lagrange_extraction(x)
        
        lagrange_extraction_connection = lagrange_extraction.copy()
        lagrange_extraction_connection.data[:] = 1

        return lagrange_extraction_connection
    
    def get_boundary_dofs(self) -> np.ndarray:

        nx = self._number_basis_per_direction[0]
        ny = self._number_basis_per_direction[1]
        
        ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        indices = (iy * nx + ix).flatten()
        boundary_mask = (
            (ix == 0) | (ix == nx - 1) |
            (iy == 0) | (iy == ny - 1) 
        ).flatten()

        boundary_indices = indices[boundary_mask]

        return np.sort(boundary_indices)
    
    def get_edges_dofs(self) -> np.ndarray:

        nx = self._number_basis_per_direction[0]
        ny = self._number_basis_per_direction[1]

        ix, iy = np.meshgrid(
            np.arange(nx), np.arange(ny), indexing='ij'
        )

        ix, iy = ix.flatten(), iy.flatten()
        node_ids = ix + iy * nx

        edges_dofs = [
            np.sort(node_ids[iy == 0]),
            np.sort(node_ids[ix == 0]),
            np.sort(node_ids[ix == nx - 1]),
            np.sort(node_ids[iy == ny - 1]),
        ]

        return edges_dofs
    
    def get_vertices_dofs(self) -> np.ndarray:

        nx = self._number_basis_per_direction[0]
        ny = self._number_basis_per_direction[1]

        ix, iy = np.meshgrid(
            np.arange(nx), np.arange(ny), indexing='ij'
        )

        ix, iy = ix.flatten(), iy.flatten()
        node_ids = ix + iy * nx

        def mask(*conds):
            return np.logical_and.reduce(list(conds))

        vertices_dofs = [
            node_ids[mask(ix==0, iy==0)],
            node_ids[mask(ix==nx-1, iy==0)],
            node_ids[mask(ix==0, iy==ny-1)],
            node_ids[mask(ix==nx-1, iy==ny-1)],
        ]

        return vertices_dofs

    def plot_basis(self, n: list[int], basis_index: list[int]) -> None:

        x = np.linspace(0, 1, n[0])
        y = np.linspace(0, 1, n[1])

        X, Y = np.meshgrid(x, y)

        b1 = self._BSpline_basis[0].evaluate(x)
        b2 = self._BSpline_basis[0].evaluate(x)

        A = b1[None, :, None, :]
        B = b2[:, None, :, None]
        C = (B * A).reshape(n[0], n[1], self._total_number_basis) 

        for i in basis_index:

            plt.figure(figsize=(6, 5))
            contour = plt.contourf(X, Y, C[:,:,i//2], levels=20, cmap='viridis')

            plt.colorbar(contour)
            plt.title(f'Contour Plot of $B{i}$')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xticks(self._knots[0])
            plt.yticks(self._knots[1])
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def plot_function(self, coefs: np.ndarray, n: list[int]) ->  None:

        x = np.linspace(0, 1, n[0])
        y = np.linspace(0, 1, n[1])

        X, Y = np.meshgrid(x, y)

        b1 = self._BSpline_basis[0].evaluate(x)
        b2 = self._BSpline_basis[0].evaluate(x)

        A = b1[None, :, None, :]
        B = b2[:, None, :, None]
        C = (B * A).reshape(n[0], n[1], self._total_number_basis) 

        D = (C * coefs[None, None, :]).sum(axis=2)
        D_clipped = np.clip(D, 0, 0.1)

        plt.figure(figsize=(6, 5))
        contour = plt.contourf(X, Y, D_clipped, levels=20, cmap='viridis')#, vmin=0, vmax=1)

        plt.colorbar(contour)
        plt.title(f'Contour Plot of $F$')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xticks(self._knots[0])
        plt.yticks(self._knots[1])
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def get_total_number_basis(self) -> None:

        return self._total_number_basis
    
class BSpline3D:

    def __init__(self, n: list[int], degree: int) -> None:

        self._n = n
        self._degree = degree

        knots_x = [0]*degree + list(np.linspace(0,1,n[0]+1)) + [1]*degree
        knots_y = [0]*degree + list(np.linspace(0,1,n[1]+1)) + [1]*degree
        knots_z = [0]*degree + list(np.linspace(0,1,n[2]+1)) + [1]*degree

        self._knots = [knots_x, knots_y, knots_z]

        basis_x = sp.BSplineBasis(degree+1, knots_x) 
        basis_y = sp.BSplineBasis(degree+1, knots_y)
        basis_z = sp.BSplineBasis(degree+1, knots_z)

        self._BSpline_basis = [basis_x, basis_y, basis_z]
        self._number_basis_per_direction = [n[0]+degree, n[1]+degree, n[2]+degree]
        self._total_number_basis = (n[0]+degree) * (n[1]+degree) * (n[2]+degree)

    def evaluate(self, x: np.ndarray) -> np.ndarray:

        fx = self._BSpline_basis[0].evaluate(x[:,0])
        fy = self._BSpline_basis[1].evaluate(x[:,1])
        fz = self._BSpline_basis[2].evaluate(x[:,2])

        f = (fz[:,:,None,None]*fy[:,None,:,None]*fx[:,None,None,:]).reshape(x.shape[0], self._total_number_basis)
        
        return f
    
    def get_lagrange_extraction(self, x: np.ndarray) -> np.ndarray:

        return self.evaluate(x).T
    
    def get_lagrange_extraction_connection(self, x:np.ndarray) -> np.ndarray:

        lagrange_extraction = self.get_lagrange_extraction(x)
        
        mask = np.nonzero(lagrange_extraction)

        lagrange_extraction_connection = np.zeros(shape = lagrange_extraction.shape)
        lagrange_extraction_connection[mask] = 1

        return lagrange_extraction_connection
    
    def get_boundary_dofs(self) -> np.ndarray:

        nx = self._number_basis_per_direction[0]
        ny = self._number_basis_per_direction[1]
        nz = self._number_basis_per_direction[2]
        
        ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
        indices = (iz * ny * nx + iy * nx + ix).flatten()
        boundary_mask = (
            (ix == 0) | (ix == nx - 1) |
            (iy == 0) | (iy == ny - 1) |
            (iz == 0) | (iz == nz - 1)
        ).flatten()

        boundary_indices = indices[boundary_mask]

        return np.sort(boundary_indices)
    
    def get_faces_dofs(self):

        nx = self._number_basis_per_direction[0]
        ny = self._number_basis_per_direction[1]
        nz = self._number_basis_per_direction[2]

        ix, iy, iz = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij'
        )

        ix, iy, iz = ix.flatten(), iy.flatten(), iz.flatten()
        node_ids = ix + iy * nx + iz * nx * ny

        faces_dofs = [
            np.sort(node_ids[ix == 0]),
            np.sort(node_ids[ix == nx - 1]),
            np.sort(node_ids[iy == 0]),
            np.sort(node_ids[iy == ny - 1]),
            np.sort(node_ids[iz == 0]),
            np.sort(node_ids[iz == nz - 1]),
        ]

        return faces_dofs
    
    def get_edges_dofs(self):

        nx = self._number_basis_per_direction[0]
        ny = self._number_basis_per_direction[1]
        nz = self._number_basis_per_direction[2]

        ix, iy, iz = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij'
        )

        ix, iy, iz = ix.flatten(), iy.flatten(), iz.flatten()
        node_ids = ix + iy * nx + iz * nx * ny

        def mask(*conds):
            """Helper to apply multiple conditions"""
            return np.logical_and.reduce(list(conds))


        edges_dofs = [
            np.sort(node_ids[mask(iy==0, iz==0)]),
            np.sort(node_ids[mask(iy==0, iz==nz-1)]),
            np.sort(node_ids[mask(iy==ny-1, iz==0)]),
            np.sort(node_ids[mask(iy==ny-1, iz==nz-1)]),

            np.sort(node_ids[mask(ix==0, iz==0)]),
            np.sort(node_ids[mask(ix==0, iz==nz-1)]),
            np.sort(node_ids[mask(ix==nx-1, iz==0)]),
            np.sort(node_ids[mask(ix==nx-1, iz==nz-1)]),

            np.sort(node_ids[mask(ix==0, iy==0,)]),
            np.sort(node_ids[mask(ix==0, iy==ny-1,)]),
            np.sort(node_ids[mask(ix==nx-1, iy==0,)]),
            np.sort(node_ids[mask(ix==nx-1, iy==ny-1,)]),
        ]

        # --- Vertices ---
        vertices = {
            f"v{i}{j}{k}": node_ids[mask(
                ix == i, iy == j, iz == k
            )][0]
            for i in [0, nx-1]
            for j in [0, ny-1]
            for k in [0, nz-1]
        }
        
        return edges_dofs
    
    def get_vertices_dofs(self):

        nx = self._number_basis_per_direction[0]
        ny = self._number_basis_per_direction[1]
        nz = self._number_basis_per_direction[2]

        ix, iy, iz = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij'
        )

        ix, iy, iz = ix.flatten(), iy.flatten(), iz.flatten()
        node_ids = ix + iy * nx + iz * nx * ny

        def mask(*conds):
            """Helper to apply multiple conditions"""
            return np.logical_and.reduce(list(conds))

        vertices_dofs = [
            node_ids[mask(
                ix == i, iy == j, iz == k
            )]
            for i in [0, nx-1]
            for j in [0, ny-1]
            for k in [0, nz-1]
        ]
        
        return vertices_dofs
    
    



    

# Mesh size
# nx, ny, nz = 5, 6, 7  # replace with your mesh sizes

# # Create grid of node indices
# ix, iy, iz = np.meshgrid(
#     np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij'
# )

# # Flatten to 1D
# ix, iy, iz = ix.flatten(), iy.flatten(), iz.flatten()
# node_ids = ix + iy * nx + iz * nx * ny

# # Mask logic
# def mask(**conds):
#     """Helper to apply multiple conditions"""
#     return np.logical_and.reduce([conds[key] for key in conds])

# # --- Faces ---
# faces = {
#     'x0': node_ids[ix == 0],
#     'x1': node_ids[ix == nx - 1],
#     'y0': node_ids[iy == 0],
#     'y1': node_ids[iy == ny - 1],
#     'z0': node_ids[iz == 0],
#     'z1': node_ids[iz == nz - 1],
# }

# # --- Edges ---
# edges = {
#     'x_y0_z0': node_ids[mask(ix=..., iy=iy==0, iz=iz==0)],
#     'x_y0_z1': node_ids[mask(ix=..., iy=iy==0, iz=iz==nz-1)],
#     'x_y1_z0': node_ids[mask(ix=..., iy=iy==ny-1, iz=iz==0)],
#     'x_y1_z1': node_ids[mask(ix=..., iy=iy==ny-1, iz=iz==nz-1)],

#     'y_x0_z0': node_ids[mask(ix=ix==0, iy=..., iz=iz==0)],
#     'y_x0_z1': node_ids[mask(ix=ix==0, iy=..., iz=iz==nz-1)],
#     'y_x1_z0': node_ids[mask(ix=ix==nx-1, iy=..., iz=iz==0)],
#     'y_x1_z1': node_ids[mask(ix=ix==nx-1, iy=..., iz=iz==nz-1)],

#     'z_x0_y0': node_ids[mask(ix=ix==0, iy=iy==0, iz=...)],
#     'z_x0_y1': node_ids[mask(ix=ix==0, iy=iy==ny-1, iz=...)],
#     'z_x1_y0': node_ids[mask(ix=ix==nx-1, iy=iy==0, iz=...)],
#     'z_x1_y1': node_ids[mask(ix=ix==nx-1, iy=iy==ny-1, iz=...)],
# }

# # --- Vertices ---
# vertices = {
#     f"v{i}{j}{k}": node_ids[mask(
#         ix=ix == i, iy=iy == j, iz=iz == k
#     )][0]
#     for i in [0, nx-1]
#     for j in [0, ny-1]
#     for k in [0, nz-1]
# }

# # Optional: gather all boundary indices
# all_boundary = np.unique(np.concatenate(list(faces.values())))
