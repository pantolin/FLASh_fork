"""FLASh: Fast simulation tools for lattice structures.

This package provides tools for building spline-based geometries, assembling
linear elasticity problems on unfitted meshes, and solving them using domain
decomposition (BDDC) methods.

The package integrates QUGaR for quadrature and DolfinX/FEniCS for visualization.
"""

from FLASh import (
    mesh, 
    pde, 
    rom,
    utils
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "mesh",
    "pde",
    "rom",
    "utils"
]