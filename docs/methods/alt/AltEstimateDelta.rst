.. _AltEstimateDelta:

Alternatives: Estimators of correlation hyperparameters
=======================================================

Overview
--------

As discussed in the procedure for building a Gaussian process emulator
for the core problem (:ref:`ProcBuildCoreGP<ProcBuildCoreGP>`), one
approach to dealing with uncertainty in the
:ref:`hyperparameters<DefHyperparameter>` :math:`\delta` of the
correlation function in a :ref:`Gaussian process<DefGP>`
:ref:`emulator<DefEmulator>` is to ignore it, using a single estimate
of :math:`\delta` instead of a sample from its posterior distribution. We
discuss here a number of alternative ways of obtaining a suitable
estimate.

Choosing the Alternatives
-------------------------

The chosen value should be a representative central value in the
posterior distribution :math:`\pi^*(\cdot)`. Where the parameters have the
interpretation of correlation lengths, as is the case in most of the
correlation functions discussed in the alternatives page for the
emulator prior correlation function
(:ref:`AltCorrelationFunction<AltCorrelationFunction>`), the estimate
should preferably err on the side of under-estimating these parameters
rather than over-estimating them. This will lead to an emulator with a
little more uncertainty in its predictions, which may compensate
somewhat for not formally accounting for uncertainty in :math:`\delta`.

The Nature of the Alternatives
------------------------------

The most widely used estimator, as mentioned in
:ref:`ProcBuildCoreGP<ProcBuildCoreGP>`, is the posterior mode. This
is the value of :math:`\delta` at which :math:`\pi^*(\delta)` is maximised.
Although any convenient algorithm or utility might be used for this
purpose, care should be taken because it is not uncommon to find
multiple local maxima in :math:`\pi^*(\cdot)`. If several modes are found, the
choice between them might, for instance, be made using the
cross-validation approach mentioned below, or by seeing which choice
leads to an emulator with better :ref:`validation<DefValidation>`
diagnostics (see the procedure page for validating a Gaussian process
emulator (:ref:`ProcValidateCoreGP<ProcValidateCoreGP>`)).

A related estimator is :math:`\exp(\bar\lambda)`, where :math:`\bar\lambda` is
defined in the procedure for a multivariate lognormal approximation for
correlation hyperparameters
(:ref:`ProcApproxDeltaPosterior<ProcApproxDeltaPosterior>`) as the
mode of :math:`\ln\delta`. It should be noted, though, that this will tend
to produce larger values of correlation length parameters than the
simple mode of :math:`\delta`.

Another method that has been suggested, and is widely used in spatial
statistics, although there is little experience about how it will
perform in the emulator setting, is cross-validation. The idea here is
to compare alternative candidate values of :math:`\delta` by how well they
predict the training data outputs when the emulator is built from a
subset of the training data. In its simplest form, the cross-validation
measure for a given candidate value of :math:`\delta` is computed as
follows.

#. For :math:`j=1,2,\ldots,n` compute the posterior mean :math:`\hat
   f_j=m^*(x_j)` of the j-th training sample output using (a) the
   formula for the posterior mean given in
   :ref:`ProcBuildCoreGP<ProcBuildCoreGP>`, (b) the candidate value
   of :math:`\delta` and (c) the :math:`n-1` training sample runs obtained by
   leaving out the j-th run.
#. Then the cross-validation measure is
   :math:`{\scriptstyle\sum_{j=1}^n}(f(x_j)-\hat f_j)^2`.

Small values of the cross-validation measure are best. In the full
cross-validation method, all possible values of :math:`\delta` are
candidates, with an algorithm being applied to search for the value
minimising the cross-validation measure.

Another approach that is very widely used in spatial statistics is
variogram fitting, as described in the procedure for variogram
estimation of covariance function hyperparameters
(:ref:`ProcVariogram<ProcVariogram>`). Although this is advocated in
the :ref:`Bayes linear<DefBayesLinear>` approach (see the thread for
the Bayes linear emulation for the core model
(:ref:`ThreadCoreBL<ThreadCoreBL>`)), in the fully
:ref:`Bayesian<DefBayesian>` approach using a :ref:`Gaussian
process<DefGP>` emulator it is not recommended because it
involves inappropriate approximations/simplifications.

Experience in spatial statistics may be unreliable for our purposes
because that experience is typically limited to two or three dimensions,
equivalent in MUCM terms to emulating simulators with only two or three
inputs.

Additional Comments, References, and Links
------------------------------------------

Further research is being conducted in :ref:`MUCM<DefMUCM>` into this
question, and findings will be reported in this page in due course.
