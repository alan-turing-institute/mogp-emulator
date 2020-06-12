.. _benchmarks:

**********************************
``mogp_emulator`` Benchmarks
**********************************

.. toctree::
   rosenbrock
   branin
   tsunami
   mcmc_benchmark
   mice_benchmark
   gkdr_benchmark
   histmatch_benchmark

Benchmarks
----------

The code includes a series of benchmarks that illustrate various pieces of the implementation. Benchmarks
can be run from the ``mogp_emulator/tests`` directory by entering ``make all`` or ``make benchmarks`` to
run all benchmarks, or ``make rosenbrock``, ``make branin``, ``make tsunami``, ``make mcmc``, ``make mice``, ``make gKDR``, or ``make histmatch`` to run the individual benchmarks.

Single Emulator Convergence Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first benchmark examines the convergence of a single emulator applied to the Rosenbrock function in
several different dimensions (more details can be found `here <https://www.sfu.ca/~ssurjano/rosen.html>`_).
This illustrates how the emulator predictions improve as the number of training points is increased
for different numbers of input parameters. The benchmark evaluates the Rosenbrock function in 4, 6, and
8 dimensions and shows that the mean squared prediction error and the mean variance improve with the
number of training points used. Matplotlib can optionally be used to visualize the results.

Multi-Output Convergence Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The second benchmark examines the convergence of multiple emulators derived from the same input values.
This benchmark is based on the 2D Branin function (more details on this function can be found `here <https://www.sfu.ca/~ssurjano/branin.html>`_)
. The code uses 8 different realizations of the Branin
function using different parameter values, and then examines the convergence of the 8 different
emulators fit using different number of parameter values based on the prediction errors and
variance values. The results can optionally be visualized using Matplotlib.

Multi-Output Performance Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A performance benchmark is included that uses a set of Tsunami simulation results to examine the
speed at which the code fits multiple emulators in parallel. The code fits 8, 16, 32, and 64 emulators
using 1, 2, 4, and 8 processess and notes the time required to perform the fitting. Note that the results
will depend on the number of cores on the computer -- once you exceed the number of cores, the performance
will degrade. As with the other benchmarks, Matplotlib can optionally be used to plot the results.

MCMC Benchmark
~~~~~~~~~~~~~~

A benchmark applying the software to fitting an emulator with MCMC sampling is included. The code
draws hyperparameter samples and compares the resulting posterior distributions with the values
found via maximum likelihood estimation. If Matplotlib is installed, a histogram of the parameter
samples is shown.

MICE Benchmark
~~~~~~~~~~~~~~

A benchmark comparing the MICE Sequential design method to Latin Hypercube sampling is also available.
This creates designs of a variety of sizes and computes the error on unseen data for the 2D Branin
function. It compares the accuracy of the sequential design to the Latin Hypercube for both the
predictions and uncertainties.

Dimension Reduction Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A benchmark illustrating gradient-based kernel dimension reduction is available in ``tests/benchmark_kdr_GP.py``
This problem contains a 100 dimension function with a single active dimension and shows how the loss depends
on the number of dimensions included in the reduced dimension space.

History Matching Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~

A benchmark illustrating use of the ``HistoryMatching`` class to rule out parts of the input space
using a GP emulator to make predictions. The benchmark contains a 1D and a 2D example.
