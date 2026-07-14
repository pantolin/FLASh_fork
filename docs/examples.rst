Examples
========

The repository includes a set of example scripts and test scripts to illustrate
usage and to verify functionality.

Example scripts (5 total):

- `examples/example_1.py` - Minimal BDDC elasticity solve on a gyroid-like domain.
- `examples/example_2.py` - Periodic wheel geometry example.
- `examples/example_wing.py` - Wing geometry with ROM models.
- `examples/example_4.py` - Additional ROM demonstration example.
- `examples/example_wrench_coarse.py` - ROM vs non-ROM comparison for a complex geometry.

Test scripts (6 total):

- `examples/test_solver_comparison.py` - Direct (Cholesky), PCG, and BDDC solver comparison.
- `examples/test_rom_accuracy.py` - BDDC convergence and error with reduced-order models.
- `examples/test_3.py` - Internal precursor to `test_acceleration_efficiency.py`.
- `examples/test_acceleration_efficiency.py` - Error vs. solver acceleration choices.
- `examples/test_scalability.py` - Parallel performance and scalability.
- `examples/test_fast_assembly_accuracy.py` - Fast assembly accuracy and convergence.

  The detailed test descriptions and mathematical setup are provided in the
  accompanying paper.

ROM-related helper scripts (3 total):

- `examples/create_rom.py` - ROM model generation from training data.
- `examples/test_rom_basis.py` - Basis error verification.
- `examples/single_cell_rom.py` - Single cell ROM evaluation and comparison.

Plotting script (1):

- `examples/plot_results.py` - Plotting utility for test and simulation results.

Run one example in serial with:

.. code-block:: bash

   python examples/example_wing.py

Run in parallel with MPI (when using MPI-aware solvers and libraries):

.. code-block:: bash

   mpirun -n 4 python examples/example_wing.py

Adjust the process count (`-n 4`) to your available CPUs.
