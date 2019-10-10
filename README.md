## Multi-Output Gaussian Process Emulator

`mogp_emulator` is a Python package for fitting Gaussian Process Emulators to computer simulation results.
The code contains routines for fitting GP emulators to simulation results with a single or multiple target
values, optimizing hyperparameter values, and making predictions on unseen data. The code also has utilities
that implement experimental design and cross-validation routines for constructing simulation runs and
verifying results.

[![Build Status](https://travis-ci.com/alan-turing-institute/mogp_emulator.svg?branch=master)](https://travis-ci.com/alan-turing-institute/mogp_emulator)
[![codecov](https://codecov.io/gh/alan-turing-institute/mogp_emulator/branch/master/graph/badge.svg)](https://codecov.io/gh/alan-turing-institute/mogp_emulator)
[![Documentation Status](https://readthedocs.org/projects/mogp-emulator/badge/?version=latest)](https://mogp-emulator.readthedocs.io/en/latest/?badge=latest)

### Overview

This Python package includes code implementing a fairly standard Gaussian Process regression. Given a set
of input variables and target values, the Gaussian Process interpolates those values using a multivariate
Gaussian distribution using a given mean and covariance functions. For simplicity, this implementation
assumes a zero mean function and a squared-exponential covariance function, though future code improvements
may allow for other choices. Fitting the Gaussian process requires inverting the covariance matrix computed
from the training data, which is done using Cholesky decomposition as the covariance matrix is symmetric
and positive definite (complexity is O(n^3), where n is the number of training points). The squared
exponential covariance function contains several hyperparameters, which includes a length scale for each
input variable and an overall variance. These hyperparameters can be set manually, or chosen automatically
by minimizing the negative logarithm of the marginal likelihood function. Once the hyperparameters are fit,
predictions can be made efficiently (complexity O(n) for each prediction, where n is again the number of
training points), and the variance computed (complexity O(n^2) for each prediction).

Simulations with multiple outputs can be fit by assuming that each output is fit by an independent emulator.
The code allows this to be done in parallel using the Python multiprocessing library. This is implemented
in the `MultiOutputGP` class, which exhibits an interface that is nearly identical to that of the main
`GaussianProcess` class.

The code assumes that the simulations are exact and attempts to interpolate between them. However, in some
cases, if two training points are too close to one another the resulting covariance matrix is singular due
to the co-linear points. In this case, the matrix inversion is stabilized by iteratively adding noise to
the diagonal of the matrix until the matrix can be successfully inverted. However, this procedure reduces 
the accuracy of the resulting predictions.

### Installation

#### Requirements

The code requires Python 3.6 or later, and working Numpy and Scipy installations are required. You should
be able to install these packages using `pip` if you do not have them already available on your system.
The code includes a full suite of unit tests and several benchmark problems. Running the test suite
requires pytest and the benchmarks make use of Matplotlib if you would like to visualize the results, 
though it is not required.

#### Download

You can download the code as a zipped archive from the Github repository. This will download all files
on the `master` branch, which can be unpacked and then used to install following the instructions
below.

If you prefer to check out the Github repository, you can download the code using:

	git clone https://github.com/alan-turing-institute/mogp_emulator/
	
This will clone the entire git history of the software and check out the `master` branch by default.
The `master` branch is the most stable version of the code, but will not have all features as the
code is under active development. The `devel` branch is the more actively developed branch, and all new
features will be available here as they are completed. All code in the `devel` branch is well tested and
documented to the point that results can be trusted, but may still have some minor bugs and issues. Any
other branch is used to develop new features and should be considered untested and experimental. Please
get in touch with one of the team members (see below) if you are unsure if a particular feature is
available.

#### Installation

To install the dependencies, in the main `mopg_emulator` directory enter the following into the shell:

	pip3 install -r requirements.txt
	
Then to install the main code, run the following command:

	python3 setup.py install
	
This will install all dependencies and install the main code in the system Python installation. You may
need adminstrative priveleges to install the dependencies or the software itself, depending on your
system configuration. However, any updates to the code cloned through the github repository (particularly
if you are using the devel branch, which is under more active development) will not be reflected in the
system installation using this method. If you would like to always have the most active development
version, install using:

	python3 setup.py develop
	
This will insert symlinks to the repository files into the system Python installation so that files
are updated whenever there are changes to the code files.

### Testing the Installation

#### Unit Tests

`mogp_emulator` includes a full set of unit tests. To run the test suite, you will need to install pytest.
The tests can be run from the `mogp_emulator/tests` directory by entering `make tests` or `pytest`, which
will run all tests and print out the results to the console. `pytest` can be installed using `pip` if you
would like to run the tests.

#### Benchmarks

The code includes a series of benchmarks that further illustrate the implementation. Benchmarks can be
run from the `mogp_emulator/tests` directory by entering `make benchmarks` or `make rosenbrock`,
`make branin`, or `make tsunami` to run the individual benchmarks.

##### Single Emulator Convergence Tests

The first benchmark examines the convergence of a single emulator applied to the Rosenbrock function in
several different dimensions (more details can be found at https://www.sfu.ca/~ssurjano/rosen.html).
This illustrates how the emulator predictions improve as the number of training points is increased
for different numbers of input parameters. The benchmark evaluates the Rosenbrock function in 4, 6, and
8 dimensions and shows that the mean squared prediction error and the mean variance improve with the
number of training points used. Matplotlib can optionally be used to visualize the results.

##### Multi-Output Convergence Tests

The second benchmark examines the convergence of multiple emulators derived from the same input values.
This benchmark is based on the 2D Branin function (more details on this function can be found at
https://www.sfu.ca/~ssurjano/branin.html). The code uses 8 different realizations of the Branin
function using different parameter values, and then examines the convergence of the 8 different
emulators fit using different number of parameter values based on the prediction errors and
variance values. The results can optionally be visualized using Matplotlib.

##### Performance Benchmark

A performance benchmark is included that uses a set of Tsunami simulation results to examine the
speed at which the code fits multiple emulators in parallel. The code fits 8, 16, 32, and 64 emulators
using 1, 2, 4, and 8 processess and notes the time required to perform the fitting. Note that the results
will depend on the number of cores on the computer -- once you exceed the number of cores, the performance
will degrade. As with the other benchmarks, Matplotlib can optionally be used to plot the results.

##### MCMC Benchmark

A benchmark applying the software to fitting an emulator with MCMC sampling is included. The code
draws hyperparameter samples and compares the resulting posterior distributions with the values
found via maximum likelihood estimation. If Matplotlib is installed, a histogram of the parameter
samples is shown.

##### MICE Benchmark

A benchmark comparing the MICE Sequential design method to Latin Hypercube sampling is also available.
This creates designs of a variety of sizes and computes the error on unseen data for the 2D Branin
function. It compares the accuracy of the sequential design to the Latin Hypercube for both the
predictions and uncertainties.


### Documentation

Building the documentation requires Sphinx/autodoc, which can be installed using `pip`. To build the documentatation, first install Sphinx and change to the `docs` directory. There is a Makefile in the
`docs` directory that will build the documentation for you. To build the HTML version, enter the
following into the shell from the `docs` directory:

	make html
	
This will build the HTML version of the documentation. A standalone PDF version can be built, which
requires a standard LaTeX installation, via:

	make latexpdf
	
In both cases, the documentation files can be found in the corresponding directories in the `docs/_build`
directory. Note that if these directories are not found on your system, you may need to create them in
order for the build to finish correctly. A version of the documentation can also be found at the link
above on Read the Docs.

### References

(1) Rasmussen, C. E.; Williams, C. K. I. Gaussian Processes for Machine Learning, 3. print.; Adaptive computation and machine learning; MIT Press: Cambridge, Mass., 2008.

### Contact

This package is under active development by the Research Engineering Group at the Alan Turing Institute
as part of several projects on Uncertainty Quantification. Feedback on the usability and features that
you would find useful can be sent to Eric Daub (edaub@turing.ac.uk). If you encounter any bugs or
problems with installing the software, please see the Issues tab on the Github page, and if the issue is
not present, create a new one.
