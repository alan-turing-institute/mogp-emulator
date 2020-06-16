.. _methoddetails:

Details on the Method
=====================

The UQ workflow described in the :ref:`overview` section has three main components: Experimental Design,
a Gaussian Process Emulator, and History Matching, each of which we describe
in more detail on this page. For more specifics on the software implementation, see the linked pages
to the individual classes provided with this library.

Experimental Design
-------------------

To run the simulator, we must first select the inputs given some broad knowledge of the parameter
space. This is done using the various :ref:`ExperimentalDesign <ExperimentalDesign>` classes, which
require that the user specifies the distribution from which parameter values are drawn. Depending
on the particular design used, the design computes a desired number of points to sample for the
experimental design.

The simplest approach is to use a :ref:`Monte Carlo Design <MonteCarloDesign>`, which simply randomly
draws points from the underlying distributions. However, in practice this does not usually give the best
performance, as no attempt is made to draw points from the full range of the distribution. To improve
upon this, we can use a :ref:`Latin Hypercube Design <LatinHypercubeDesign>`, which guarantees that
the every sample is drawn from a different quantile of the underlying distribution.

In practice, however, because the computation required to fit a surrogate model is usually small
when compared to the computational effort in running the simulator, a :ref:`Sequential Design
<SequentialDesign>` can provide improved performance. A sequential design is more intelligent
about the next point to sample by determining what new point from a set of options will improve
the surrogate model. This package provides an implementation of the
:ref:`Mutual Information for Computer Experiments (MICE) <MICEDesign>` sequential design algorithm,
which has been shown to outperform Latin Hypercubes with only a small additional computational overhead.

Gaussian Process Emulator
-------------------------

The central component of the UQ method is :ref:`Gaussian Process <GaussianProcess>` regression, which
serves as the surrogate model in the UQ workflow adopted here. Given a set
of input variables and target values, the Gaussian Process interpolates those values using a multivariate
Gaussian distribution using user-specified mean and covariance functions and priors on the hyperparameter
values (if desired). Fitting the Gaussian process requires inverting the covariance matrix computed
from the training data, which is done using Cholesky decomposition as the covariance matrix is symmetric
and positive definite (complexity is :math:`\mathcal{O}(n^3)`, where :math:`n` is the number of training
points). The squared exponential covariance function contains several hyperparameters, which includes a
length scale for each input variable and an overall variance. These hyperparameters can be set manually,
or chosen automatically by minimizing the negative logarithm of the posterior marginalized over the data.
Once the hyperparameters are fit, predictions can be made efficiently (complexity :math:`\mathcal{O}(n)`
for each prediction, where n is again the number of training points), and the variance computed (complexity :math:`\mathcal{O}(n^2)` for each prediction).

The code assumes that the simulations are exact and attempts to interpolate between them. However, in some
cases, if two training points are too close to one another the resulting covariance matrix is singular due
to the co-linear points. A "nugget" term (noise added to the diagonal of the covariance matrix) can be added
to prevent a singular covariance matrix. This nugget can be specified in advance (if the observations
have a fixed uncertainty associated with them), or can be estimated. Estimating the nugget can treat the
nugget as a hyperparameter that can be optimised, or find the nugget adaptively by attempting to make the
nugget as small as necessary in order to invert the covariance matrix. In practice, the adaptive nugget
is the most robust, but requires additional computational effort as the matrix is factored multiple times
during each optimisation step.

Covariance Functions
~~~~~~~~~~~~~~~~~~~~

The library implements two stationary :ref:`covariance functions <Kernel>`: :ref:`Squared Exponential <squaredexponential>` and :ref:`Matern 5/2 <matern52>`. These
can be specified when creating a new emulator.

Mean Functions
~~~~~~~~~~~~~~

A :ref:`Mean Function <MeanFunction>` can be specified using R-style :Ref:`formulas <formula>`. By default, these are parsed with the ``patsy``
library (if it is installed), so R users are encouraged to install this pacakge. However, ``mogp_emulator``
has its own built-in parser to construct mean functions from a string. Mean functions can also be
constructed directly from a rich language mean function classes.

Hyperparameters
~~~~~~~~~~~~~~~

There are two types of hyperparameters for a GP emulator, those associated with mean functions, and
those associated with the covariance kernel. All mean function hyperparameters are treated on a linear
scale, while the kernel hyperparameters are on a logarithmic scale as all kernel parameters are constrained
to be positive. The first part of the hyperparameter array contains the :math:`n_{mean}`
mean function parameters (i.e. if the mean function has 4 parameters, then the first 4 entries in the
hyperparameter array belong to the mean function), then come the :math:`D` correlation length hyperparameters
(the same as the number of inputs), followed by the covariance and nugget hyperparameters. This means that the
total number of hyperparameters depends on the mean function specification and is :math:`n_{mean}+D+2`.

To interpret the correlation length hyperparameters, the relationship between the reported hyperparameter
:math:`\theta` and the correlation length :math:`d` is :math:`\exp(-\theta)=d^2`. Thus, a large
positive value of :math:`\theta` indicates a small correlation length, and a large negative value of
:math:`\theta` indicates a large correlation length.

The covariance scale :math:`\sigma^2` can be interpreted as :math:`\exp(\theta)=\sigma^2`. Thus, in
this case a large positive value of :math:`\theta` indicates a large overall variance scale and a
large negative value of :math:`\theta` indicates a small variance scale.

If the nugget is estimated via hyperparmeter optimisation, the nugget is determined by
:math:`\exp(\theta) = \delta`, where :math:`\delta` is added to the diagonal of the covariance matrix.
Large positive values of :math:`\theta` indicates a large nugget and a
large negative value of :math:`\theta` indicates a small nugget. The nugget value can always be extracted
on a linear scale via the ``nugget`` attribute of a GP regardless of how it was fit, so this is the
most reliable way to determine the nugget.

Hyperparameter Priors
~~~~~~~~~~~~~~~~~~~~~

:ref:`Prior beliefs <Priors>` can be specified on hyperparameter values. Exactly how these are
interpreted depends on the type of hyperparameter and the type of prior distribution. For
:ref:`normal prior <NormalPrior>` distributions, these are applied directly to the hyperparameter
values with no transformation. Thus, for mean function hyperparameters, a normal distribution is
assumed for a normal prior, while for kernel parameters a lognormal distribution is assumed.

For the :ref:`Gamma <GammaPrior>` and :ref:`Inverse Gamma <InvGammaPrior>` priors, the distribution is
only defined over positive hyperparameter values, so all parameters are exponentiated and then the
exponentiated value is used when computing the log PDF.

Multi-Output GP
---------------

Simulations with multiple outputs can be fit by assuming that each output is fit by an independent emulator.
The code allows this to be done in parallel using the Python multiprocessing library. This is implemented
in the :ref:`MultiOutputGP class <MultiOutputGP>`, which exhibits an interface that is nearly identical
to that of  the main :ref:`GaussianProcess <GaussianProcess>` class.

Estimating Hyperparameters
--------------------------

For regular and Multi-Output GPs, hyperparameters are fit using the ``fit_GP_MAP`` function in
the :ref:`fitting module <fitting>`, using L-BFGS optimisation on the negative log posterior.
This modifies the hyperparameter values of the GP or MOGP object, returning a fit object that
can be used for prediction.

History Matching
----------------

The final component of the UQ workflow is the calibration method. This library implements
:ref:`History Matching <historymatching>` to perform model calibration
to determine which points in the input space are plausible given a set of observations.
Performing History Matching requires a fit GP emulator to a set of simulator runs and an observation
associated with the simulator output. The emulator is then used to efficiently estimate the simulator
output, accounting for all uncertainties, to compare with observations and points that are unlikely
to produce the observation can then be "ruled out" and deemed implausible, reducing the input space to
better understand the system under question.

At the moment, History Matching is only implemented for a single output and a single set of simulation
runs. Future work will extend this to multiple outputs and multiple waves of simulations.