.. _ThreadGenericEmulateDerivatives:

Thread: Generic methods to emulate derivatives
==============================================

Overview
--------

This thread describes how we can build an
:ref:`emulator<DefEmulator>` with which we can predict the
derivatives of the model output with respect to the inputs. If we have
derivative information available, either from an
:ref:`adjoint<DefAdjoint>` model or some other means, we can include
that information when emulating derivatives. This is similar to the
variant thread on emulators with derivative information
(:ref:`ThreadVariantWithDerivatives<ThreadVariantWithDerivatives>`)
which includes derivative information when emulating function output. If
the adjoint to a simulator doesn't exist and we don't wish to obtain
derivative information through another method, it is still possible to
emulate model derivatives with just the function output.

The emulator
------------

The derivatives of a posterior Gaussian process remain Gaussian
processes with mean and covariance functions obtained by the relevant
derivatives of the posterior mean and covariance functions. This can be
applied to any Gaussian process emulator. The process of building an
emulator of derivatives with the fully Bayesian approach is given in the
procedure page
:ref:`ProcBuildEmulateDerivsGP<ProcBuildEmulateDerivsGP>`. This
covers building a Gaussian process emulator of derivatives with just
function output, an extension of the core thread
:ref:`ThreadCoreGP<ThreadCoreGP>`, and a Gaussian process emulator of
derivatives built with function output and derivative information, an
extension of
:ref:`ThreadVariantWithDerivatives<ThreadVariantWithDerivatives>`.

The result is a Gaussian process emulator of derivatives which will
correctly represent any derivatives in the training data, but it is
always important to validate the emulator against additional derivative
information. For the :ref:`core problem<DiscCore>`, the process of
validation is described in the procedure page
:ref:`ProcValidateCoreGP<ProcValidateCoreGP>`. Although here we are
interested in emulating derivatives, as we know the derivatives of a
Gaussian process remain a Gaussian process, we can apply the same
validation techniques as for the core problem. We require a validation
design :math::ref:`\strut D^\prime` which consists of points where we want to
obtain validation derivatives. An `adjoint<DefAdjoint>` is then
run at these points; if an appropriate adjoint does not exist the
derivatives are obtained through another technique, for example finite
differences. If any local sensitivity analysis has already been
performed on the simulator, some derivatives may already have been
obtained and can be used here for validation. Then in the case of a
linear mean function, weak prior information on hyperparameters
:math:`\strut \\beta` and :math:`\strut \\sigma`, and a single posterior
estimate of :math:`\strut \\delta`, the predictive mean vector, :math:`\strut
m^*:ref:`, and the predictive covariance matrix, :math:`\strut V^*`, required
in `ProcValidateCoreGP<ProcValidateCoreGP>`, are given by the
functions :math:`\tilde{m}^*(\cdot)` and :math:`\tilde{v}^*(\cdot,\cdot)`
which are given in
:ref:`ProcBuildEmulateDerivsGP<ProcBuildEmulateDerivsGP>`. We can
therefore validate an emulator of derivatives using the same procedure
as that which we apply to validate an emulator of the core problem. It
is often necessary, in response to the validation diagnostics, to
rebuild the emulator using additional training runs which can of course,
include derivatives. We hope to extend the validation process using
derivatives as we gain more experience in validation diagnostics and
emulating with derivative information.

The :ref:`Bayes linear<DefBayesLinear>` approach to emulating
derivatives may be covered in a future release of the toolkit.

Tasks
-----

Having obtained a working emulator, the MUCM methodology now enables
efficient analysis of a number of tasks that regularly face users of
simulators.

Prediction
~~~~~~~~~~

The simplest of these tasks is to use the emulator as a fast surrogate
for the adjoint, i.e. to predict what derivatives the adjoint would
produce if run at a new point in the input space. The process of
predicting function output at one or more new points for the core
problem is set out in the prediction page
:ref:`ProcPredictGP<ProcPredictGP>`. Here we are predicting
derivatives and the process of prediction is the same as for the core
problem. If the procedure in
:ref:`ProcBuildEmulateDerivsGP<ProcBuildEmulateDerivsGP>` is
followed, :math:`\tilde{D}, \\tilde{t}, \\tilde{A}, \\tilde{e}` etc are
used in replace of :math::ref:`D, t, A, e`, as required in
`ProcPredictGP<ProcPredictGP>`.

Sensitivity analysis
~~~~~~~~~~~~~~~~~~~~

In :ref:`sensitivity analysis<DefSensitivityAnalysis>` the objective
is to understand how the output responds to changes in individual inputs
or groups of inputs. Local sensitivity analysis uses derivatives to
study the effect on the output, when the inputs are perturbed by a small
amount. Emulated derivatives could replace adjoint produced derivatives
in this analysis if the adjoint is too expensive to execute or in fact
does not exist.

Other tasks
~~~~~~~~~~~

Derivatives can be informative in optimization problems. If we want to
find which sets of input values results in either a maximum or a minimum
output then knowledge of the gradient of the function, with respect to
the inputs, may result in a more efficient search. Derivative
information is also useful in :ref:`data
assimilation<DefDataAssimilation>`.
