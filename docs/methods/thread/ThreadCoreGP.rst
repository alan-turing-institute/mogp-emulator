.. _ThreadCoreGP:

Thread: Analysis of the core model using Gaussian Process methods
=================================================================

Overview
--------

The principal user entry points to the :ref:`MUCM<DefMUCM>` toolkit
are the various *threads*, as explained in the :ref:`Toolkit
Structure<MetaToolkitStructure>`. The main threads give detailed
instructions for building and using :ref:`emulators<DefEmulator>` in
various contexts.

This thread takes the user through the analysis of the most basic kind
of problem, using the fully :ref:`Bayesian<DefBayesian>` approach
based on a Gaussian process (GP) emulator. We characterise a core
problem or model as follows:

-  We are only concerned with one :ref:`simulator<DefSimulator>`.
-  The simulator only produces one output, or (more realistically) we
   are only interested in one output.
-  The output is :ref:`deterministic<DefDeterministic>`.
-  We do not have observations of the real world process against which
   to compare the simulator.
-  We do not wish to make statements about the real world process.
-  We cannot directly observe derivatives of the simulator.

Each of these aspects of the core problem is discussed further in page
:ref:`DiscCore<DiscCore>`.

The fully Bayesian approach has a further restriction:

-  We are prepared to represent the simulator as a :ref:`Gaussian
   process<DefGP>`.

See also the discussion page on the Gaussian assumption
(:ref:`DiscGaussianAssumption<DiscGaussianAssumption>`).

This thread comprises a number of key stages in developing and using the
emulator.

Active inputs
-------------

Before beginning to develop the emulator, it is necessary to decide what
inputs to the simulator will be varied. Complex simulators often have
many inputs and many outputs. In the core problem, only one output is of
interest and we assume that this has already been identified. It may
also be necessary to restrict the number of inputs that will be
represented in the emulator. The distinction between :ref:`active
inputs<DefActiveInput>` and :ref:`inactive
inputs<DefInactiveInput>` is considered in the discussion page
:ref:`DiscActiveInputs<DiscActiveInputs>`.

Once the active inputs have been determined, we will refer to these
simply as the inputs, and we denote the number of (active) inputs by
:math:`p`.

The GP model
------------

The first stage in building the emulator is to model the mean and
covariance structures of the Gaussian process that is to represent the
simulator. As explained in the definition of a Gaussian process
(:ref:`DefGP<DefGP>`), a GP is characterised by a mean function and a
covariance function. We model these functions to represent prior beliefs
that we have about the simulator, i.e. beliefs about the simulator prior
to incorporating information from the :ref:`training
sample<DefTrainingSample>`.

The choice of a mean function is considered in the alternatives page
:ref:`AltMeanFunction<AltMeanFunction>`. In general, the choice will
lead to the mean function depending on a set of
:ref:`hyperparameters<DefHyperparameter>` that we will denote by
:math:`\beta`.

The most common approach is to define the mean function to have the
linear form :math:` m(x) = h(x)^{\rm T}\beta \`, where :math:`h(\cdot)` is a
vector of regressor functions, whose specification is part of the choice
to be made. For appropriate ways to model the mean, both generally and
in linear form, see :ref:`AltMeanFunction<AltMeanFunction>`.

The GP covariance function is discussed in page
:ref:`DiscCovarianceFunction<DiscCovarianceFunction>`. Within the
toolkit we will assume that the covariance function takes the form
:math:`\sigma^2 c(\cdot,\cdot)`, where :math:`\sigma^2` is an unknown scale
hyperparameter and :math:`c(\cdot, \\cdot)` is called the correlation
function indexed by a set of correlation hyperparameters :math:`\delta`.
The choice of the emulator prior correlation function is considered in
the alternatives page
:ref:`AltCorrelationFunction<AltCorrelationFunction>`.

The most common approach is to define the correlation function to have
the Gaussian form :math:`c(x,x') = \\exp\{-(x-x')^{\rm T}C(x-x')\}`, where
:math:`C` is a diagonal matrix with elements the inverse squares of the
elements of the :math:`\delta` vector. A slightly more complex form is the
Gaussian with nugget, :math:`c(x,x') = \\nu I_{x=x'} +
(1-\nu)\exp\{-(x-x')^{\rm T}C(x-x')\}:ref:`, where the
`nugget<DefNugget>` :math:`\nu` may represent effects of inactive
variables and the expression :math:`I_{x=x'}` takes the value 1 if
:math::ref:`x=x'` and otherwise is 0. See
`AltCorrelationFunction<AltCorrelationFunction>` for more
details.

The techniques that follow in this thread will be expressed as far as
possible in terms of the general forms of the mean and covariance
functions, depending on general hyperparameters :math:`\beta`,
:math:`\sigma^2` and :math:`\delta`. However, in many cases, simpler formulae
and methods can be developed when the linear and Gaussian forms are
chosen, and some techniques in this thread may only be available in the
special cases.

Prior distributions
-------------------

The GP modelling stage will have described the mean and covariance
structures in terms of some hyperparameters. A fully Bayesian approach
now requires that we express probability distributions for these that
are again *prior* distributions. Possible forms of prior distribution
are discussed in the alternatives page on prior distributions for GP
hyperparameters (:ref:`AltGPPriors<AltGPPriors>`). The result is in
general a joint distribution :math:`\pi(\beta,\sigma^2,\delta)`. Where
required, we will denote the marginal distribution of :math:`\delta` by
:math:`\pi_\delta (\cdot)`, and similarly for marginal distributions of
other groups of hyperparameters.

Design
------

The next step is to create a :ref:`design<DefDesign>`, which consists
of a set of points in the input space at which the simulator is to be
run to create the training sample. Design options for the core problem
are discussed in the alternatives page on training sample design for the
core problem (:ref:`AltCoreDesign<AltCoreDesign>`).

The result of applying one of the design procedures described there is
an ordered set of points :math:`D = \\{x_1, x_2, \\ldots, x_n\}`. The
simulator :math:`f(\cdot)` is then run at each of these input
configurations, producing a vector :math:`f(D)` of :math:`n` elements, whose
i-th element :math:`f(x_i)` is the output produced by the simulator from
the run with inputs :math:`x_i`.

One suggestion that is commonly made for the choice of the sample size
:math:`n` is :math:`n=10p`, where :math:`p` is the number of inputs. (This may
typically be enough to obtain an initial fit, but additional simulator
runs are likely to be needed for the purposes of
:ref:`validation<DefValidation>`, and then to address problems raised
in the validation diagnostics as discussed below.)

Fitting the emulator
--------------------

Given the training sample and the GP prior model, the procedure for
building a GP emulator for the core problem is theoretically
straightforward, and is set out in page
:ref:`ProcBuildCoreGP<ProcBuildCoreGP>`. Nevertheless, there are
several computational difficulties that are discussed there.

The result of :ref:`ProcBuildCoreGP<ProcBuildCoreGP>` is the
emulator, fitted to the prior information and training data. As
discussed fully in the page on forms of GP based emulators
(:ref:`DiscGPBasedEmulator<DiscGPBasedEmulator>`), the emulator has
two parts, an updated GP (or a related process called a
:ref:`t-process<DefTProcess>`) conditional on hyperparameters, plus
one or more sets of representative values of those hyperparameters.
Addressing the tasks below will then consist of computing solutions for
each set of hyperparameter values (using the GP or t-process) and then
an appropriate form of averaging of the resulting solutions.

Although the fitted emulator will correctly represent the information in
the training data, it is always important to validate it against
additional simulator runs. The procedure for validating a Gaussian
process emulator is described in page
:ref:`ProcValidateCoreGP<ProcValidateCoreGP>`. It is often necessary,
in response to the validation diagnostics, to rebuild the emulator using
additional training runs.

Tasks
-----

Having obtained a working emulator, the MUCM methodology now enables
efficient analysis of a number of tasks that regularly face users of
simulators.

Prediction
~~~~~~~~~~

The simplest of these tasks is to use the emulator as a fast surrogate
for the simulator, i.e. to predict what output the simulator would
produce if run at a new point in the input space. The procedure for
predicting the simulator's output in one or more new points is set out
in page :ref:`ProcPredictGP<ProcPredictGP>`.

For some of the tasks considered below, we require to predict the output
not at a set of discrete points, but in effect the entire output
function as the inputs vary over some range. This can be achieved also
using simulation, as discussed in the procedure page for simulating
realisations of an emulator
(:ref:`ProcSimulationBasedInference<ProcSimulationBasedInference>`).

Uncertainty analysis
~~~~~~~~~~~~~~~~~~~~

:ref:`Uncertainty analysis<DefUncertaintyAnalysis>` is the process of
predicting the simulator output when one or more of the inputs are
uncertain. The procedure page for performing uncertainty analysis using
a GP emulator (:ref:`ProcUAGP<ProcUAGP>`) explains how this is done.

Sensitivity analysis
~~~~~~~~~~~~~~~~~~~~

In :ref:`sensitivity analysis<DefSensitivityAnalysis>` the objective
is to understand how the output responds to changes in individual inputs
or groups of inputs. The procedure page for variance based sensitivity
analysis using a GP emulator (:ref:`ProcVarSAGP<ProcVarSAGP>`) gives
details of carrying out :ref:`variance based<DefVarianceBasedSA>`
sensitivity analysis.

Examples
--------

-  :ref:`One dimensional example<ExamCoreGP1Dim>`
-  :ref:`Two dimensional example<ExamCoreGP2Dim>` with uncertainty
   and sensitivity analysis

Additional Comments, References, and Links
------------------------------------------

Other tasks that can be addressed include optimisation (finding the
values of one or more inputs that will minimise or maximise the output)
and decision analysis (finding an optimal decision according to a formal
description of utilities). A related task is
:ref:`decision-based<DefDecisionBasedSA>` sensitivity analysis. We
expect to add procedures for these tasks for the core problem in due
course.

Another task that is very often required is
:ref:`calibration<DefCalibration>`. This requires us to think about
the relationship between the simulator and reality, which is dealt with
in
:ref:`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`.
Tasks involving observations of the real process are explicitly excluded
from the core problem.
