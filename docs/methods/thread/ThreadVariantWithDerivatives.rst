.. _ThreadVariantWithDerivatives:

Thread: Emulators with derivative information
=============================================

Overview
--------

This thread describes how we can use derivative information in addition
to standard function output to build an :ref:`emulator<DefEmulator>`.
As in the core thread for analysing the core problem using GP methods
(:ref:`ThreadCoreGP<ThreadCoreGP>`) the following apply:

-  We are only concerned with one :ref:`simulator<DefSimulator>`.
-  The simulator only produces one output, or (more realistically) we
   are only interested in one output.
-  The output is :ref:`deterministic<DefDeterministic>`.
-  We do not have observations of the real world process against which
   to compare the simulator.
-  We do not wish to make statements about the real world process.

Each of these aspects of the core problem is discussed further in page
:ref:`DiscCore<DiscCore>`.

However now, we also assume we can directly observe first derivatives of
the simulator either through an :ref:`adjoint<DefAdjoint>` model or
some other technique. This thread describes the use of derivative
information when building an emulator for the fully Bayesian approach
only, thus we require the further restriction:

-  We are prepared to represent the simulator as a :ref:`Gaussian
   process<DefGP>`.

There is discussion of this requirement in page
:ref:`DiscGaussianAssumption<DiscGaussianAssumption>`. If we want to
adopt the :ref:`Bayes linear<DefBayesLinear>` approach it is still
possible to include derivative information and this may be covered in a
future release of the toolkit. Further information on this can be found
in Killeya, M.R.H., (2004) "Thinking Inside The Box" Using Derivatives
to Improve Bayesian Black Box Emulation of Computer Simulators with
Applications to Compartmental Models. Ph.D. thesis, Department of
Mathematical Sciences, University of Durham.

Readers should be familiar with :ref:`ThreadCoreGP<ThreadCoreGP>`,
before considering including derivative information.

Active inputs
-------------

As in :ref:`ThreadCoreGP<ThreadCoreGP>`

The GP model
------------

As in :ref:`ThreadCoreGP<ThreadCoreGP>` the first stage in building
the emulator is to model the mean and covariance structures of the
Gaussian process that is to represent the simulator. As explained in the
definition page of a Gaussian process (:ref:`DefGP<DefGP>`), a GP is
characterised by a mean function and a covariance function. We model
these functions to represent prior beliefs that we have about the
simulator, i.e. beliefs about the simulator prior to incorporating
information from the :ref:`training sample<DefTrainingSample>`. The
derivatives of a Gaussian process remain a Gaussian process and so we
can use a similar approach to :ref:`ThreadCoreGP<ThreadCoreGP>` here.

The choice of the emulator prior mean function is considered in the
alternatives page :ref:`AltMeanFunction<AltMeanFunction>`. However
here we must ensure that the chosen function is differentiable. In
general, the choice will lead to the mean function depending on a set of
:ref:`hyperparameters<DefHyperparameter>` that we will denote by
:math:`\beta`.

The most common approach is to define the mean function to have the
linear form :math:`m(x) = h(x)^{\rm T}\beta`, where :math:`h(\cdot)` is a
vector of regressor functions, whose specification is part of the choice
to be made. As we are including derivative information in the training
sample we must ensure that :math:`h(\cdot)` is differentiable. This will
then lead to the derivative of the mean function:
:math:`\frac{\partial}{\partial x}m(x) = \frac{\partial}{\partial
x}h(x)^{\rm T}\beta`. For appropriate ways to model the mean, both
generally and in linear form, see
:ref:`AltMeanFunction<AltMeanFunction>`.

The covariance function is considered in the discussion page
:ref:`DiscCovarianceFunction<DiscCovarianceFunction>` and here must
be twice differentiable. Within the toolkit we will assume that the
covariance function takes the form :math:`\sigma^2 c(\cdot,\cdot)`, where
:math:`\sigma^2` is an unknown scale hyperparameter and :math:`c(\cdot,
\cdot)` is called the correlation function indexed by a set of
correlation hyperparameters :math:`\delta`. The correlation then between a
point, :math:`x_i`, and a derivative w.r.t input :math:`k` at
:math:`x_j`, (denoted by :math:`x_j^{(k)}`), is :math:`\frac{\partial}{\partial
x_j^{(k)}} c(x_i,x_j)`. The correlation between a derivative w.r.t
input :math:`k` at :math:`x_i`, (denoted by :math:`x_i^{(k)}`), and
a derivative w.r.t input :math:`l` at :math:`x_j`, (denoted by
:math:`x_j^{(l)}`), is :math:`\frac{\partial^2}{\partial x_i^{(k)}\partial
x_j^{(l)}} c(x_i,x_j)`. The choice of correlation function is
considered in the alternatives page
:ref:`AltCorrelationFunction<AltCorrelationFunction>`.

The most common approach is to define the correlation function to have
the Gaussian form :math:`c(x_i,x_j) = \exp\{-(x_i-x_j)^{\rm
T}C(x_i-x_j)\}`, where :math:`C` is a diagonal matrix with elements
the inverse squares of the elements of the :math:`\delta` vector. The
correlation then between a point, :math:`x_i`, and a derivative
w.r.t input :math:`k` at point :math:`j`, :math:`x_j^{(k)},` is:

.. math::
   \frac{\partial}{\partial x_j^{(k)}} c(x_i,x_j) =
   \frac{2}{\delta^2\{k\}}\left(x_i^{(k)}-x_j^{(k)}\right)\,\exp\{-(x_i-x_j)^{\rm
   T}C(x_i-x_j)\},

the correlation between two derivatives w.r.t input :math:`k` but at
points :math:`i` and :math:`j` is:

.. math::
   \frac{\partial^2}{\partial x_i^{(k)} \partial x_j^{(k)}} c(x_i,x_j)
   = \left(\frac{2}{\delta^2\{k\}} -
   \frac{4\left(x_i^{(k)}-x_j^{(k)}\right)^2}{\delta^4\{k\}}\right)\exp\{-(x_i-x_j)^{\rm
   T}C(x_i-x_j)\},

and finally the correlation between two derivatives w.r.t inputs
:math:`k` and :math:`l`, where :math:`k \ne l`, at points
:math:`i` and :math:`j` is:

.. math::
   \frac{\partial^2}{\partial x_i^{(k)} \partial x_j^{(l)}} c(x_i,x_j)
   =\frac{4}{\delta^2\{k\}\delta^2\{l\}}\left(x_j^{(k)}-x_i^{(k)}\right)\left(x_i^{(l)}-x_j^{(l)}\right)\exp\{-(x_i-x_j)^{\rm
   T}C(x_i-x_j)\}

Prior distributions
-------------------

As in :ref:`ThreadCoreGP<ThreadCoreGP>`

Design
------

The next step is to create a :ref:`design<DefDesign>`, which consists
of a set of points in the input space at which the simulator or adjoint
is to be run to create the training sample. Design options for the core
problem are discussed in the alternatives page on training sample design
(:ref:`AltCoreDesign<AltCoreDesign>`). Here though, we also need to
decide at which of these points we want to obtain function output and at
which points we want to obtain partial derivatives. This adds a further
consideration when choosing a design option but as yet we don't have any
specific design procedures which take into account the inclusion of
derivative information.

If one of the design procedures described in
:ref:`AltCoreDesign<AltCoreDesign>` is applied, the result is an
ordered set of points :math:`D = \{x_1, x_2, \ldots, x_n\}`. Given
:math:`D`, we would now need to choose at which of these points we
want to obtain function output and at which we want to obtain partial
derivatives. This information is added to :math:`D` resulting in the
design, :math:`\tilde{D}` of length :math:`\tilde{n}`. A point
in :math:`\tilde{D}` has the form :math:`(x,d)`, where :math:`d`
denotes whether a derivative or the function output is to be included at
that point. The simulator, :math:`f(\cdot)`, or the adjoint of the
simulator, :math:`\tilde{f}(\cdot)`, (depending on the value of each
:math:`d`), is then run at each of the input configurations.

One suggestion that is commonly made for the choice of the sample size,
:math:`n`, for the core problem is :math:`n=10p`, where :math:`p`
is the number of inputs. (This may typically be enough to obtain an
initial fit, but additional simulator runs are likely to be needed for
the purposes of :ref:`validation<DefValidation>`, and then to address
problems raised in the validation diagnostics.) There is not, however,
such a guide for what :math:`\tilde{n}` might be. If we choose to
obtain function output and the first derivatives w.r.t to all inputs at
every location in the design, then we would expect that fewer than
:math:`10p` locations would be required; how many fewer though, is
difficult to estimate.

Fitting the emulator
--------------------

Given the training sample of function output and derivatives, and the GP
prior model, the process of building the emulator is given in the
procedure page :ref:`ProcBuildWithDerivsGP<ProcBuildWithDerivsGP>`

The result of :ref:`ProcBuildWithDerivsGP<ProcBuildWithDerivsGP>` is
the emulator, fitted to the prior information and training data. As with
the core problem, the emulator has two parts, an updated GP (or a
related process called a :ref:`t-process<DefTProcess>`) conditional
on hyperparameters, plus one or more sets of representative values of
those hyperparameters. Addressing the tasks below will then consist of
computing solutions for each set of hyperparameter values (using the GP
or t-process) and then an appropriate form of averaging of the resulting
solutions.

Although the fitted emulator will correctly represent the information in
the training data, it is always important to validate it against
additional simulator runs. For the :ref:`core problem<DiscCore>`, the
process of validation is described in the procedure page
:ref:`ProcValidateCoreGP<ProcValidateCoreGP>`. Here, we are
interested in predicting function output, therefore as in
:ref:`ProcValidateCoreGP<ProcValidateCoreGP>` we will have a
validation design :math:`D^\prime` which only consists of points for
function output; no derivatives are required and as such the simulator,
:math:`f(\cdot)`, not the adjoint, :math:`\tilde{f}(\cdot)`, is run at each
:math:`x_j^\prime` in :math:`D^\prime`. Then in the case of a
linear mean function, weak prior information on hyperparameters
:math:`\beta` and :math:`\sigma`, and a single posterior
estimate of :math:`\delta`, the predictive mean vector, :math:`m^*`,
and the predictive covariance matrix, :math:`\strut V^*`, required
in :ref:`ProcValidateCoreGP<ProcValidateCoreGP>`, are given by the
functions :math:`m^*(\cdot)` and :math:`v^*(\cdot,\cdot)` which are given in
:ref:`ProcBuildWithDerivsGP<ProcBuildWithDerivsGP>`. We can therefore
validate an emulator built with derivatives using the same procedure as
that which we apply to validate an emulator of the core problem. It is
often necessary, in response to the validation diagnostics, to rebuild
the emulator using additional training runs which can of course, include
derivatives. We hope to extend the validation process using derivatives
as we gain more experience in validation diagnostics and emulating with
derivative information.

Tasks
-----

Having obtained a working emulator, the MUCM methodology now enables
efficient analysis of a number of tasks that regularly face users of
simulators.

Prediction
~~~~~~~~~~

The simplest of these tasks is to use the emulator as a fast surrogate
for the simulator, i.e. to predict what output the simulator would
produce if run at a new point in the input space. In this thread we are
concerned with predicting the function output of the simulator. The
prediction of derivatives of the simulator output w.r.t the inputs, at a
new point in the input space is covered in the thread
:ref:`ThreadGenericEmulateDerivatives<ThreadGenericEmulateDerivatives>`.
The process of predicting function output at one or more new points for
the core problem is set out in :ref:`ProcPredictGP<ProcPredictGP>`.
When we have derivatives in the training sample the process of
prediction is the same as for the core problem, but anywhere :math:`D, t, A,
e` etc are required, they should be replaced with :math:`\tilde{D},
\tilde{t}, \tilde{A}, \tilde{e}`.

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
uncertain. The procedure page on uncertainty analysis using a GP
emulator (:ref:`ProcUAGP<ProcUAGP>`) explains how this is done for
the core problem. We hope to extend this procedure to cover an emulator
built with derivative information in a later release of the toolkit.

Sensitivity analysis
~~~~~~~~~~~~~~~~~~~~

In :ref:`sensitivity analysis<DefSensitivityAnalysis>` the objective
is to understand how the output responds to changes in individual inputs
or groups of inputs. The procedure page
:ref:`ProcVarSAGP<ProcVarSAGP>` gives details of carrying out
:ref:`variance based<DefVarianceBasedSA>` sensitivity analysis for
the core problem. We hope to extend this procedure to cover an emulator
built with derivative information in a later release of the toolkit.

Examples
--------

:ref:`One dimensional example<ExamVariantWithDerivatives1Dim>`

Additional Comments, References, and Links
------------------------------------------

If we are interested in emulating multiple outputs of a simulator, there
are various approaches to this discussed in the alternatives page
:ref:`AltMultipleOutputsApproach<AltMultipleOutputsApproach>`. If the
approach chosen is to build a :ref:`multivariate
GP<DefMultivariateGP>` emulator and derivatives are available,
then they can be included using the methods described in this page
combined with the methods described in the thread for the analysis of a
simulator with multiple outputs
(:ref:`ThreadVariantMultipleOutputs<ThreadVariantMultipleOutputs>`).
A variant thread on multiple outputs with derivatives
(ThreadVariantMultipleOutputsWithDerivatives) page may be included in a
later release of the toolkit.
