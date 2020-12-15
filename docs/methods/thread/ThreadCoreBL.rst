.. _ThreadCoreBL:

Thread: Bayes linear emulation for the core model
=================================================

Overview
--------

This page takes the user through the construction and analysis of an
emulator for a simple univariate computer simulator -- the
:ref:`core<DiscCore>` problem. The approach described here employs
:ref:`Bayes linear methods<DefBayesLinear>`.

Requirements
------------

The method and techniques described in this page are applicable when we
satisfy the following requirements:

-  We are considering a core problem with the following features:

   -  We are only concerned with one :ref:`simulator<DefSimulator>`.
   -  The simulator only produces one output, or (more realistically) we
      are only interested in one output.
   -  The output is :ref:`deterministic<DefDeterministic>`.
   -  We do not have observations of the real world process against
      which to compare the simulator.
   -  We do not wish to make statements about the real world process.
   -  We cannot directly observe derivatives of the simulator.

-  We are prepared to represent our beliefs about the simulator with a
   :ref:`second-order specification<DefSecondOrderSpec>` and so are
   following the :ref:`Bayes linear<DefBayesLinear>` approach

The Bayes linear emulator
-------------------------

The Bayes linear approach to emulation is (comparatively) simple in
terms of belief specification and analysis, requiring only mean,
variance and covariance specifications for the uncertain output of the
computer model rather than a full joint probability distribution for the
entire collection of uncertain computer model output. For a detailed
discussion of Bayes linear methods, see
:ref:`DiscBayesLinearTheory<DiscBayesLinearTheory>`; for discussion
of Bayes linear methods in comparison to the Gaussian process approach
to emulation, see :ref:`AltGPorBLEmulator<AltGPorBLEmulator>`.

Our belief specification for the univariate deterministic simulator is
given by the Bayes linear :ref:`emulator<DefEmulator>` of the
simulator :math:`f(x)` which takes the a linear :ref:`mean
function<AltMeanFunction>` in the following structural form:

.. math::
   f(x) = \sum_j \beta_{j}\, h_{j}(x) + w(x)

In this formulation, :math:`\beta=(\beta_{1}, \dots,\beta_{p})` are
unknown scalars, :math:`h(x)=(h_{1}(x), \dots,h_{p}(x))` are known
deterministic functions of :math:`x`, and :math:`w(x)` is a stochastic
residual process. Thus our mean function has the linear form
:math:`m(x)=h(x)^T\beta`.

Thus our belief specification for the computer model can be expressed in
terms of beliefs about two components. The component :math:`h^T(x) \beta`
is a linear trend term that expresses our beliefs about the global
variation in :math:`f`, namely that portion of the variation in :math:`f(x)`
which we can resolve without having to make evaluations for :math:`f` at
input choices which are near to :math:`x`. The residual :math:`w(x)` expresses
local variation, which we take to be a weakly stationary stochastic
process with constant variance :math:`\sigma^2` (for a discussion on the
covariance function see
:ref:`DiscCovarianceFunction<DiscCovarianceFunction>`), and a
specified :ref:`correlation function<AltCorrelationFunction>`
:math:`c(x,x')` which is parametrised by correlation hyperparameters
:math:`\delta`. We treat :math:`\beta` and :math:`w(x)` as being uncorrelated a
priori. The advantages of including a structured mean function, such as
the linear form used here, are discussed in
:ref:`DiscStructuredMeanFunction<DiscStructuredMeanFunction>`.

Emulator prior specification
----------------------------

Given the emulator structure described above, in order to construct a
Bayes linear emulator for a given simulator :math:`f(x)` we require the
following ingredients:

-  The form of the trend :ref:`basis functions<DefBasisFunctions>`
   :math:`h(x)`
-  Expectations, variances, and covariances for the trend coefficients
   :math:`\beta`
-  Expectation of the residual process :math:`w(x)` at a given input :math:`x`
-  The form of the residual covariance function :math:`c(x,x')`
-  One of:

   #. Specified values for the residual variance :math:`\sigma^2` and
      correlation hyperparameters :math:`\delta`,
   #. Expectations, variances and covariances for :math:`(\sigma^2,\delta)`
   #. A sufficiently large number of model evaluations to estimate
      :math:`(\sigma^2,\delta)` empirically

These specifications are used to represent our prior beliefs that we
have about the simulator before incorporating information from the
:ref:`training sample<DefTrainingSample>`. We now discuss obtaining
appropriate specifications for each of these quantities.

Choosing the form of :math:`h(x)`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Bayes linear emulation, the emphasis of the emulator is often placed
on a :ref:`detailed structural
representation<DiscStructuredMeanFunction>` of the simulator's
mean behaviour. Therefore the choice of trend basis function is a key
component of the BL emulator. This choice can be made directly by an
expert or by empirical investigation of a large sample of simulator
evaluations. Methods for determining appropriate choices of :math:`h(x)`
are discussed in the alternatives page on basis functions for the
emulator mean (:ref:`AltBasisFunctions<AltBasisFunctions>`).

Choosing the form of :math:`c(x,x')`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the simulator's behaviour is well-captured by the chosen mean
function, then the proportion of variation in the simulator output that
is explained by the residual stochastic process is quite small making
the choice of the form for :math:`c(x,x')` less influential in subsequent
analyses. Nonetheless, alternatives on the emulator prior correlation
function are considered in
:ref:`AltCorrelationFunction<AltCorrelationFunction>`. A typical
choice is the Gaussian correlation function for the residuals.

If we have chosen to work with :ref:`active inputs<DefActiveInput>`
in the mean function, then the covariance function often includes a
:ref:`nugget<DefNugget>` term, representing the variation in the
output of the simulator which is not explained by the active inputs. See
the discussion page on active and inactive inputs
(:ref:`DiscActiveInputs<DiscActiveInputs>`).

Belief specifications for :math:`\beta`, :math:`\sigma^2`, and :math:`\delta`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The emulator modelling stage will have described the form of the mean
and covariance structures in terms of some hyperparameters. A Bayes
linear approach now requires that we express our prior beliefs about
these hyperparameters.

Given the specified trend functions :math:`h(x)`, we now require an
expectation and variance for each coefficient :math:`\beta_j` and a
covariance between every pair :math:`(\beta_j,\beta_k)`. We additionally
require a specification of values for the residual variance
:math:`\sigma^2` and the correlation function parameters :math:`\delta`.
Depending on the availability of expert information and the level of
detail of the specification, this may take the form of (a)
expert-specified point values, (b) expert-specified expectations and
variances, (c) empirically obtained numerical estimates.

As with the basis functions, these specifications can either be made
from expert judgement or via data analysis when there are sufficient
simulator evaluations. Further details on making these specifications
are described in the alternatives page on prior specification for BL
hyperparameters (:ref:`AltBLPriors<AltBLPriors>`).

Design
------

The next step is to create a :ref:`design<DefDesign>`, which consists
of a set of points in the input space at which the simulator is to be
run to create the training sample. Alternative choices on training
sample design for the core problem are given in
:ref:`AltCoreDesign<AltCoreDesign>`.

The result of applying one of the design procedures described there is a
matrix of :math:`n` points :math:`X=(x_1,\dots,x_n)^T`. The simulator is then
run at each of these input configurations, producing an :math:`n`-vector
:math:`f(X)` of elements, whose i-th element is the output :math:`f(x_i)`
produced by the simulator from the run with inputs :math:`x_i`.

Building the emulator
---------------------

Empirical construction from runs only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the prior information is weak and the amount of available data is
large, then any Bayesian posterior would be dominated by the data. Thus
given a specified form for the simulator mean function, we can estimate
:math:`\beta` and :math:`\sigma^2` via standard regression techniques. This
will give estimates :math:`\hat{\beta}` and :math:`\hat{\sigma}^2` which can
be treated as adjusted/posterior values for those parameters given the
data. The procedure for the empirical construction of a Bayes linear
emulator is described in
:ref:`ProcBuildCoreBLEmpirical<ProcBuildCoreBLEmpirical>`.

Bayes linear assessment of the emulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given the output :math:`f(X)`, we make a Bayes linear adjustment of the
trend coefficients :math:`\beta` and the residual function :math:`w(x)`. This
adjustment requires the specification of a prior mean and variance
:math:`\beta`, a covariance specification for :math:`w(x)`, and specified
values for :math:`\sigma^2` and :math:`\delta`. Given the design, model runs
and the prior BL emulator the process of
:ref:`adjusting<DefBLAdjust>` :math:`\beta` and :math:`w(x)` is described
in the procedure page for building a BL emulator for the core problem
(:ref:`ProcBuildCoreBL<ProcBuildCoreBL>`).

Bayes linear adjustment for residual variance and correlation functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before carrying out the Bayes linear assessment as described above, we
may learn about the residual variance via Bayes linear :ref:`variance
learning<DefBLVarianceLearning>`. Consequently, we additionally
require a second-order prior specification for :math:`\sigma^2` which may
come from expert elicitation or analysis of fast approximate models. The
procedure for adjusting our beliefs about the emulator residual variance
is described in
:ref:`ProcBLVarianceLearning<ProcBLVarianceLearning>`.

We may similarly use Bayes linear variance learning methods for updating
our beliefs about the correlation function (and hence :math:`\delta`.)

Bayes linear emulator construction with uncertain variance and
correlation hyperparameters will be developed in a later version of the
Toolkit.

Diagnostics and validation
--------------------------

Although the fitted emulator will correctly represent the information in
the simulator runs, it is always important to validate it against
additional model evaluations runs. We assess this by applying the
diagnostic checks and, if necessary, rebuilding the emulator using runs
from an additional design.

The procedure page on validating a Gaussian process emulator
(:ref:`ProcValidateCoreGP<ProcValidateCoreGP>`) describes diagnostics
and validation for GP emulators. This approach is generally applicable
to the BL case and so can be used to validate a Bayes linear emulator.
However unlike the GP diagnostic process, the Bayes linear approach
would not consider the diagnostic values to have particular distribution
forms. Specific Bayes linear diagnostics will be developed in a future
version.

Post-emulation tasks
--------------------

Having obtained a working emulator, the MUCM methodology now enables
efficient analysis of a number of tasks that regularly face users of
simulators.

Prediction
~~~~~~~~~~

The simplest of these tasks is to use the emulator as a fast surrogate
for the simulator, i.e. to predict what output the simulator would
produce if run at a new point :math:`x` in the input space. The procedure
for predicting one or more new points using a BL emulator is set out in
:ref:`ProcBLPredict<ProcBLPredict>`.

Uncertainty analysis
~~~~~~~~~~~~~~~~~~~~

Uncertainty analysis is the process of predicting the computer model
output, when the inputs to the computer model are also uncertain,
thereby exposing the uncertainty in model outputs that is attributable
to uncertainty in the inputs. The Bayes linear approach to such a
prediction problem is described in the procedure page on Uncertainty
analysis for a Bayes linear emulator (:ref:`ProcUABL<ProcUABL>`).

Sensitivity analysis
~~~~~~~~~~~~~~~~~~~~

In :ref:`sensitivity analysis<DefSensitivityAnalysis>` the objective
is to understand how the output responds to changes in individual inputs
or groups of inputs. In general, when the mean function of the emulator
accounts for a large proportion of the variation of the simulator then
the sensitivity of the simulator to changes in the inputs can be
investigated by examination of the basis functions of :math:`m(x)` and
their corresponding coefficients. In the case where the mean function
does not explain much of the simulator variation and the covariance
function is Gaussian then the methods of the procedure page on variance
based sensitivity analysis (:ref:`ProcVarSAGP<ProcVarSAGP>`) are
broadly applicable if we are willing to ascribe a prior distributional
form to the simulator input.
