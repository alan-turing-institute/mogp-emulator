.. _ThreadVariantMultipleOutputs:

Thread: Analysis of a simulator with multiple outputs using Gaussian Process methods
====================================================================================

Overview
--------

The multivariate emulator
~~~~~~~~~~~~~~~~~~~~~~~~~

The principal user entry points to the :ref:`MUCM<DefMUCM>` toolkit
are the various *threads*, as explained in the Toolkit structure page
(:ref:`MetaToolkitStructure<MetaToolkitStructure>`). The main threads
give detailed instructions for building and using
:ref:`emulators<DefEmulator>` in various contexts.

This thread takes the user through the analysis of a variant of the most
basic kind of problem, using the fully Bayesian approach based on a
Gaussian process (GP) emulator. We characterise the basic multi-output
model as follows:

-  We are only concerned with one :ref:`simulator<DefSimulator>`.
-  The output is :ref:`deterministic<DefDeterministic>`.
-  We do not have observations of the real world process against which
   to compare the simulator.
-  We do not wish to make statements about the real world process.
-  We cannot directly observe derivatives of the simulator.

Each of these requirements is also a part of the core problem, and is
discussed further in :ref:`DiscCore<DiscCore>`. However, the core
problem further assumes that the simulator only produces one output, or
that we are only interested in one output. We relax that assumption
here. The core thread :ref:`ThreadCoreGP<ThreadCoreGP>` deals with
the analysis of the core problem using a GP emulator. This variant
thread extends the core analysis to the case of a simulator with more
than one output.

The fully Bayesian approach has a further restriction:

-  We are prepared to represent the simulator as a :ref:`Gaussian
   process<DefGP>`.

There is discussion of this requirement in
:ref:`DiscGaussianAssumption<DiscGaussianAssumption>`.

Alternative approaches to emulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are various approaches to tackling the problems raised by having
multiple outputs, which are discussed in the alternatives page on
emulating multiple outputs
(:ref:`AltMultipleOutputsApproach<AltMultipleOutputsApproach>`). Some
approaches reduce or transform the multi-output model so that it can be
analysed by the methods in :ref:`ThreadCoreGP<ThreadCoreGP>`.
However, others employ a :ref:`multivariate GP<DefMultivariateGP>`
emulator that is described in detail in the remainder of this thread.

The GP model
------------

The first stage in building the emulator is to model the mean and
covariance structures of the Gaussian process that is to represent the
simulator. As explained in the definition of a :ref:`multivariate Gaussian
process<DefMultivariateGP>`, a GP is characterised by a mean
function and a covariance function. We model these functions to
represent *prior* beliefs that we have about the simulator, i.e. beliefs
about the simulator prior to incorporating information from the
:ref:`training sample<DefTrainingSample>`.

Alternative choices of the emulator prior mean function are considered
in :ref:`AltMeanFunction<AltMeanFunction>`, with specific discussion
on the multivariate case in
:ref:`AltMeanFunctionMultivariate<AltMeanFunctionMultivariate>`. In
general, the choice will lead to the mean function depending on a set of
:ref:`hyperparameters<DefHyperparameter>` that we will denote by
:math:`\beta`. We will generally write the mean function as :math:`m(\cdot)`
where the dependence on :math:`\beta` is implicit. Note that if we have
:math:`r` outputs, then :math:`m(\cdot)` is a vector of :math:`1 \times r`
elements comprising the mean functions of the various outputs.

The most common approach is to define the mean function to have the
linear form :math:`m(x) = h^T (x)\beta`, where :math:`h(\cdot)` is a :math:`q
\times 1` vector of regressor (or basis) functions whose specification
is part of the choice to be made. Note that :math:`\beta` is a :math:`q \times
r` matrix.

The covariance function for a multivariate GP specifies the :math:`r\times
r` covariance matrix between the :math:`r` outputs of the simulator at an
input configuration :math:`x` and the :math:`r` outputs at input :math:`x'`. A
number of options of varying complexity are available for the covariance
function, which are discussed in
:ref:`AltMultivariateCovarianceStructures<AltMultivariateCovarianceStructures>`.
The hyperparameters in a general covariance function, including the
hyperparameters in the correlation function and scale parameters in the
covariance function are denoted by :math:`\omega`.

The techniques that follow in this thread will be expressed as far as
possible in terms of the general forms of the mean and covariance
functions, depending on general hyperparameters :math:`\beta` and
:math:`\omega`. However, in many cases, simpler formulae and methods can be
developed when the linear mean function and the separable covariance
function with a Gaussian form of correlation function are are chosen,
and some techniques in this thread may only be available in the special
cases.

Prior distributions
-------------------

The GP modelling stage will have described the mean and covariance
structures in terms of some hyperparameters. A fully Bayesian approach
now requires that we express probability distributions for these that
are again *prior* distributions. Alternative forms of prior
distributions for GP hyperparameters are discussed in
:ref:`AltGPPriors<AltGPPriors>`, with some specific suggestions for
the covariance function hyperparameters :math:`\omega` given in
:ref:`AltMultivariateCovarianceStructures<AltMultivariateCovarianceStructures>`.
The result is in general a joint (prior) distribution
:math:`\pi(\beta,\omega)`. Where required, we will denote the marginal
distribution of :math:`\omega` by :math:`\pi_\omega(\cdot)`, and similarly for
marginal distributions of other groups of hyperparameters. This
alternatives for multivariate GP priors for the input-output
:ref:`separable<DefSeparable>` case are further discussed in
:ref:`AltMultivariateGPPriors<AltMultivariateGPPriors>`.

Design
------

The next step is to create a :ref:`design<DefDesign>`, which consists
of a set of points in the input space at which the simulator is to be
run to create the training sample. Design options for the core problem
are discussed in :ref:`AltCoreDesign<AltCoreDesign>`. Design for the
multiple output problem has not been explicitly studied, but we believe
that designs for the core problem will be good also for the multi-output
problem, although it seems likely that a larger number of design points
could be required.

The result of applying one of the design procedures described there is
an ordered set of points :math:`D = \{x_1, x_2, \ldots, x_n\}`. The
simulator is then run at each of these input configurations, producing a
:math:`n\times r` matrix of outputs. The i-th column of this matrix is the
output produced by the simulator from the run with inputs :math:`x_i`.

One suggestion that is commonly made for the choice of the sample size
:math:`n` is :math:`n=10p`, where :math:`p` is the number of inputs. (This may
typically be enough to obtain an initial fit, but additional simulator
runs are likely to be needed for the purposes of
:ref:`validation<DefValidation>`, and then to address problems raised
in the validation diagnostics as discussed below. In general there are
no sure rules of thumb for this choice and careful validation is
critical in building an emulator.)

Fitting the emulator
--------------------

Given the training sample and the GP prior model, the process of
building the emulator is theoretically straightforward, and is set out
in the procedure page for building a multivariate Gaussian process
emulator for the core problem
(:ref:`ProcBuildMultiOutputGP<ProcBuildMultiOutputGP>`). If a
separable covariance function is chosen, then there are various
simplifications to the procedure, set out in
:ref:`ProcBuildMultiOutputGPSep<ProcBuildMultiOutputGPSep>`.

The result of :ref:`ProcBuildMultiOutputGP<ProcBuildMultiOutputGP>`
is the emulator, fitted to the prior information and training data. It
is in the form of an updated multivariate GP (or, in the separable
covariance case, a related process called a :ref:`multivariate
t-process<DefMultivariateTProcess>`) conditional on
hyperparameters, plus one or more sets of representative values of those
hyperparameters. Addressing the tasks below will then consist of
computing solutions for each set of hyperparameter values (using the
multivariate GP or t-process) and then an appropriate form of averaging
of the resulting solutions (see the procedure page on predicting
simulator outputs using a GP emulator
(:ref:`ProcPredictGP<ProcPredictGP>`)).

Although the fitted emulator will correctly represent the information in
the training data, it is always important to validate it against
additional simulator runs. The procedure of validating a Gaussian
process emulator is described in
:ref:`ProcValidateCoreGP<ProcValidateCoreGP>`. It is often necessary,
in response to the validation diagnostics, to rebuild the emulator using
additional training runs.

Validating multi-output emulators is more challenging. The most simple
approach is to validate on individual outputs (ignoring any correlations
between them implied by the covariance function) using the methods
defined in :ref:`ProcValidateCoreGP<ProcValidateCoreGP>`. This is not
the full answer, however there is relatively little experience of
validating multivariate emulators in the literature. We hope to develop
this and include insights in future releases of the toolkit.

Tasks
-----

Having obtained a working emulator, the MUCM methodology now enables
efficient analysis of a number of tasks that regularly face users of
simulators.

Prediction
~~~~~~~~~~

The simplest of these tasks is to use the emulator as a fast surrogate
for the simulator, i.e. to predict what output the simulator would
produce if run at a new point :math:`x^\prime` in the input space. The
process of predicting one or more new points is set out in
:ref:`ProcPredictGP<ProcPredictGP>`.

For some of the tasks considered below, we require to predict the output
not at a set of discrete points, but in effect the entire output
function as the inputs vary over some range. This can be achieved also
using simulation, as discussed in the procedure page for simulating
realisations of an emulator
(:ref:`ProcSimulationBasedInference<ProcSimulationBasedInference>`).

Sometimes interest will be in a deterministic function of one or more of
the outputs. If your only interest is in a function of a set of outputs
which is a pre-determined mapping, building a direct single output
emulator is probably the most efficient approach. In other situations,
such as when you are interested in both the raw outputs and one or more
functions of the outputs, or when you are interested in function(s) that
depend some auxiliary variables other than just the raw outputs of the
simulator, then it is better to build the multivariate emulator first,
then use the procedure for obtaining a sample from the predictive
distribution of the function set out in
:ref:`ProcPredictMultiOutputFunction<ProcPredictMultiOutputFunction>`.

It is worth noting that the predictive distribution (i.e. emulator) of
any **linear** transformation of the outputs is also a multivariate GP,
or t-process. In particular, the emulator of a single linear combination
of outputs is a regular univariate GP emulator, so all the core theory
applies whenever we want to do anything with a single output or with a
single linear combination of outputs. It is important to realise that
having built a multivariate emulator, these single linear output
functions are derived from it, not by fitting a univariate emulator
separately (which will almost certainly produce slightly different
results).

Uncertainty analysis
~~~~~~~~~~~~~~~~~~~~

:ref:`Uncertainty analysis<DefUncertaintyAnalysis>` is the process of
predicting the simulator output when one or more of the inputs are
uncertain.

Sensitivity analysis
~~~~~~~~~~~~~~~~~~~~

In :ref:`sensitivity analysis<DefSensitivityAnalysis>` the objective
is to understand how the output responds to changes in individual inputs
or groups of inputs. The most common approach is a :ref:`variance
based<DefVarianceBasedSA>` sensitivity analysis.

Examples
--------

:ref:`ExamMultipleOutputs<ExamMultipleOutputs>` is an example
demonstrating the multivariate emulator with a number of different
covariance functions.
:ref:`ExamMultipleOutputsPCA<ExamMultipleOutputsPCA>` is a more
complex example showing a reduced dimension multivariate emulator
applied to a chemometrics model using PCA to reduce the dimension of the
output space.

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
Although the calibration task itself is not covered in this release of
the Toolkit we hope to include it in a future release.
