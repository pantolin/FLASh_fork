PDE module API
==============

This section documents the public classes and functions within `FLASh.pde`.

Classes
-------

.. automodule:: FLASh.pde.linear_pde
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members:
        make_unit_square_quadrature
        zero_function
        compute_K
        compute_K_core
        create_facet_quadrature

.. automodule:: FLASh.pde.solver
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members:
        write_solutions

.. automodule:: FLASh.pde.bddc
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members:
        residual_tracker

Functions
---------

.. autofunction:: FLASh.pde.linear_pde.make_unit_square_quadrature
.. autofunction:: FLASh.pde.linear_pde.zero_function
.. autofunction:: FLASh.pde.linear_pde.create_facet_quadrature
.. autofunction:: FLASh.pde.solver.write_solutions
