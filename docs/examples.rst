Examples
========

The repository includes a set of example scripts and test scripts to illustrate
usage and to verify functionality.

Example scripts (5 total):

- `examples/example_1.py` - Minimal BDDC elasticity solve on a gyroid-like domain.
- `examples/example_2.py` - Periodic wheel geometry example.
- `examples/example_3.py` - Wing geometry with ROM models.
- `examples/example_4.py` - Additional ROM demonstration example.
- `examples/example_5.py` - ROM vs non-ROM comparison for a complex geometry.

Test scripts (6 total):

- `examples/test_1.py` through `examples/test_6.py` - Regression and functional tests
  for assembly, solving, geometry handling, and ROM behavior.

  The detailed test descriptions and mathematical setup are provided in the
  accompanying paper.

ROM-related helper scripts (3 total):

- `examples/create_rom.py` - ROM model generation from training data.
- `examples/rom_basis_test.py` - Basis error verification.
- `examples/single_cell_rom.py` - Single cell ROM evaluation and comparison.

Plotting script (1):

- `examples/plot_results.py` - Plotting utility for test and simulation results.

Run one example in serial with:

.. code-block:: bash

   python examples/example_3.py

Run in parallel with MPI (when using MPI-aware solvers and libraries):

.. code-block:: bash

   mpirun -n 4 python examples/example_3.py

Adjust the process count (`-n 4`) to your available CPUs.
