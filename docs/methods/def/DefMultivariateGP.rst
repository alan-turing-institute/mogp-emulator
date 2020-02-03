.. _DefMultivariateGP:

Definition of Term: Multivariate Gaussian process
=================================================

The :ref:`univariate Gaussian process<DefGP>` (GP) is a probability
distribution for a function, in which the joint distribution of any set
of points on that function is multivariate normal. In MUCM it is used to
develop an :ref:`emulator<DefEmulator>` for the output of a
:ref:`simulator<DefSimulator>`, regarding the simulator output as a
function :math:`f(x)` of its input(s) :math:`x`.

Most simulators in practice produce multiple outputs (e.g. temperature,
pressure, wind speed, ...) for any given input configuration. The
multivariate GP is an extension of the univariate GP to the case where
:math:`f(x)` is not a scalar but a vector. If the simulator has :math:`r`
outputs then :math:`f(x)` is :math:`r\times 1`.

Formally, the multivariate GP is a probability model over functions with
multivariate values. It is characterised by a mean function :math:`m(\cdot)
= \textrm{E}[f(\cdot)]` and a covariance function :math:`v(\cdot,\cdot) =
\textrm{Cov}[f(\cdot),f(\cdot)]`. Under this model, the function
evaluated at a single input :math:` x` has a multivariate normal
distribution :math:`f(x) \sim N\left(m(x),v(x,x)\right)`, where:

-  :math:`m(x)` is the :math:`r \times 1` mean vector of :math:`f(x)`
   and
-  :math:`v(x,x)` is the :math:`r\times r` covariance matrix of
   :math:`f(x)`.

Furthermore, the joint distribution of :math:`f(x)` and :math:`f(x')` is
also multivariate normal:

.. math::
   \left( \begin{array}{c}f(x) \\ f(x') \end{array} \right) \sim
   N \left( \,\left(\begin{array}{c}m(x) \\ m(x') \end{array}
   \right), \left(\begin{array}{cc}v(x,x) & v(x,x') \\ v(x',x) &
   v(x',x') \end{array} \right) \,\right)

so that :math:`v(x,x')` is the :math:`r\times r` matrix of
covariances between elements of :math:`f(x)` and :math:`f(x')`. In general,
the multivariate GP is characterised by the fact that if we stack the
vectors :math:`f(x_1), f(x_2),\ldots,f(x_n)` at an arbitrary set of
:math:`n` outputs :math:`D = (x_1,\ldots,x_n)` into a vector of
:math:`rn` elements, then this has a multivariate normal
distribution. The only formal constraint on the covariance function
:math:`v(.,.)` is that the :math:`rn\times rn` covariance matrix
for any arbitrary set :math:`D` should always be non-negative
definite.

The general multivariate GP is therefore very flexible. However, a
special case which has the following more restrictive form of covariance
function can be very useful:

.. math::
   v(\cdot,\cdot) = \Sigma c(\cdot, \cdot),

where :math:`\Sigma` is a covariance matrix between outputs and
:math:`c(\cdot, \cdot)` is a correlation function between input points.

This restriction implies a kind of :ref:`separability<DefSeparable>`
between inputs and outputs. With a covariance function of this form the
multivariate GP has an important property, which is exploited in the
procedure for emulating multiple outputs
:ref:`ProcBuildMultiOutputGPSep<ProcBuildMultiOutputGPSep>`. One
aspect of this separable property is that when we stack the vectors
:math:`f(x_1),\ldots,f(x_n)` into an :math:`rn\times 1` vector its
covariance matrix has the Kronecker product form :math:`\Sigma\otimes
c(D,D)`, where :math:`c(D,D)` is the between inputs correlation matrix
with elements :math:`c(x_i,x_{i'})` defined according to the notational
conventions in the :ref:`notation<MetaNotation>` page.

Another aspect is that if instead of stacking the output vectors into a
long vector we form instead the :math:`r\times n` matrix :math:`f(D)`
(again following the conventions in the :ref:`notation<MetaNotation>`
page) then :math:`f(D)` has a matrix-variate normal distribution with mean
matrix :math:`m(D)`, between-rows covariance matrix :math:`\Sigma` and
between-columns covariance matrix :math:`c(D,D)`.

References
----------

Information on matrix variate distributions can be found in:

-  `Matrix variate
   distributions <http://www.crcpress.com/ecommerce_product/product_detail.jsf?catno=LM06108&isbn=0000000000000&af=W1129>`__,
   Arjun K. Gupta, D. K. Nagar, CRC Press, 1999
