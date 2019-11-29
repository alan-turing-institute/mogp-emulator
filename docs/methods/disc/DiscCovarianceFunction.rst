.. _DiscCovarianceFunction:

Discussion: The GP covariance function
======================================

Description and Background
--------------------------

In the fully :ref:`Bayesian<DefBayesian>` approach an
:ref:`emulator<DefEmulator>` is created based on a :ref:`Gaussian
process<DefGP>` (GP), and one of the principal steps in building
the emulator is to specify the GP's covariance function. The covariance
function also has an analogous role in the :ref:`Bayes
linear<DefBayesLinear>` approach. See the thread for the
analysis of the core model using Gaussian process methods
(:ref:`ThreadCoreGP<ThreadCoreGP>`) and the thread for Bayes linear
emulation for the core model (:ref:`ThreadCoreBL<ThreadCoreBL>`) for
emulation of the :ref:`core problem<DiscCore>` from the fully
Bayesian and Bayes linear approaches, respectively.

Discussion
----------

The covariance function gives the prior covariance between the simulator
output at any given vector of input values and the output at any other
given input vector. When these two vectors are the same, the covariance
function specifies the prior variance of the output at that input
vector, and so quantifies prior uncertainty about the simulator output.
In principal, this function can take a very wide variety of forms.

Within the :ref:`MUCM<DefMUCM>` toolkit, we make an assumption of
constant prior variance, so that in the case of a single output (as in
the core problem) the covariance function has the form :math:`\sigma^2
c(\cdot, \cdot)`, where :math:`c(\cdot, \cdot)` is a correlation
function having the property that, for any input vector :math:`x`,
:math:`c(x,x)=1`. Hence the interpretation of :math:`\sigma^2` is as the
variance of the output (at any input). The correlation function
generally depends on a set of
:ref:`hyperparameters<DefHyperparameter>` denoted by :math:`\delta`,
while :math:`\sigma^2` is another hyperparameter.

When building a :ref:`multivariate GP<DefMultivariateGP>` emulator
for a simulator with multiple outputs, the covariance between the sets
of outputs at inputs :math:`x` and :math:`x'` is a matrix which is assumed to
have the :ref:`separable<DefSeparable>` form :math:`\Sigma c(\cdot,
\cdot)`, where now :math:`\Sigma` is a between-outputs covariance matrix
but :math:`c(\cdot, \cdot)` is a correlation function over the input space
with the same interpretation for each output as in the single output
case. See the variant thread for the analysis of a simulator with
multiple outputs using Gaussian process methods
(:ref:`ThreadVariantMultipleOutputs<ThreadVariantMultipleOutputs>`).

Additional Comments
-------------------

More generally, we might relax the assumption that the variance is
constant across the input space. However, it is important for the
resulting covariance function to be valid, in the sense that the implied
variance for any linear combination of outputs at different input points
remains positive.

One valid covariance function in which the variance is not constant
takes the form :math:`u(x) u(x') c(x,x')`, where :math:`c(\cdot, \cdot)` is a
correlation function (see the alternatives page on emulator prior
correlation function
(:ref:`AltCorrelationFunction<AltCorrelationFunction>`)) as before,
while :math:`u(x)^2` is the variance of the output at inputs :math:`x`, and
can now vary with :math:`x`. In general, :math:`u(\cdot)` would be modelled
using further hyperparameters.
