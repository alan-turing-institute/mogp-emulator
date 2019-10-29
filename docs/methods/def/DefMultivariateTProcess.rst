.. _DefMultivariateTProcess:

Definition of Term: Multivariate t-process
==========================================

The :ref:`univariate t-process<DefTProcess>` is a probability
distribution for a function, in which the joint distribution of any set
of points on that function is multivariate t. In MUCM it arises in the
fully :ref:`Bayesian<DefBayesian>` approach as the underlying
distribution (after integrating out a variance
:ref:`hyperparameter<DefHyperparameter>`) of an
:ref:`emulator<DefEmulator>` for the output of a
:ref:`simulator<DefSimulator>`, regarding the simulator output as a
function :math:`f(x)` of its input(s) :math:`x`; see the procedure for
building a Gaussian process emulator for the core problem
(:ref:`ProcBuildCoreGP<ProcBuildCoreGP>`). The t-process generalises
the :ref:`Gaussian process<DefGP>` (GP) in the same way that a t
distribution generalises the normal (or Gaussian) distribution.

Most simulators in practice produce multiple outputs (e.g. {temperature,
pressure, wind speed, ...}) for any given input configuration. If the
simulator has :math:`r` outputs then :math:`f(x)` is :math:`r\times 1`. In this
context, an emulator may be based on the :ref:`multivariate
GP<DefMultivariateGP>` or a multivariate t-process.

Formally, the multivariate t-process is a probability model over
functions with multivariate values. It is characterised by a degrees of
freedom :math:`b`, a mean function :math:`m(\cdot) = \\textrm{E}[f(\cdot)]`
and a covariance function :math:`v(\cdot,\cdot) =
\\textrm{Cov}[f(\cdot),f(\cdot)]`. Under this model, the function
evaluated at a single input :math:`x` has a multivariate t distribution
with :math:`b` degrees of freedom, where:

-  :math:`m(x)` is the :math:`r \\times 1` mean vector of :math:`f(x)` and
-  :math:`v(x,x)` is the :math:`r\times r` scale matrix of :math:`f(x)`.

Furthermore, if we stack the vectors :math:`f(x_1), f(x_2),\cdots,f(x_n)`
at an arbitrary set of :math:`n` outputs :math:`D = (x_1,\cdots,x_n)` into a
vector of :math:`rn` elements, then this also has a multivariate t
distribution.

The multivariate t-process usually arises when we use a multivariate GP
model with a :ref:`separable<DefSeparable>` covariance. We therefore
shall assume unless specified otherwise that a multivariate t-process
also has a :ref:`separable<DefSeparable>` covariance, that is

:math:`v(\cdot,\cdot) = \\Sigma c(\cdot, \\cdot)`

where :math:`\Sigma` is a covariance matrix between outputs and :math:`c(\cdot,
\\cdot)` is a correlation function between input points.

With a covariance function of this form the multivariate t-process has
an important property. If instead of stacking the output vectors into a
long vector we form instead the :math::ref:`r\times n` matrix :math:`f(D)`
(following the conventions in the `notation<MetaNotation>` page)
then :math:`f(D)` has a matrix-variate t distribution with :math:`b` degrees
of freedom, mean matrix :math:`m(D)`, between-rows covariance matrix
:math:`\Sigma` and between-columns covariance matrix :math:`c(D,D)`.

References
----------

Information on matrix variate distributions can be found in:

-  `Matrix variate
   distributions <http://www.crcpress.com/ecommerce_product/product_detail.jsf?catno=LM06108&isbn=0000000000000&af=W1129>`__,
   Arjun K. Gupta, D. K. Nagar, CRC Press, 1999
