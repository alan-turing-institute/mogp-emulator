.. _DiscCoreValidationDesign:

Discussion: Design of a validation sample
=========================================

Description and Background
--------------------------

Having built an :ref:`emulator<DefEmulator>`, it is important to
:ref:`validate<DefValidation>` it by checking that the statistical
predictions it makes about the :ref:`simulator<DefSimulator>` outputs
agree with the actual simulator outputs. This validation process is
described for the :ref:`core problem<DiscCore>` in the procedure for
validating a Gaussian process emulator
(:ref:`ProcValidateCoreGP<ProcValidateCoreGP>`), and involves
comparing emulator predictions with data from a validation sample. We
discuss here the principles of :ref:`designing<DefDesign>` a
validation study.

Discussion
----------

A valid emulator should predict the simulator output with an appropriate
degree of uncertainty. Validation testing checks whether, for instance,
when the emulator's predictions are more precise (i.e. have a smaller
predictive variance) the actual simulator output is closer to the
predictive mean than when the emulator predictions are more uncertain.
The predictions will be most precise when the emulator is asked to
predict for an input point :math:`x^\prime` that is close to the training
sample points that were used to build the emulator; they will be largest
when predicting at a point :math:`x^\prime` that is far from all training
sample points (relative to the estimated correlation lengths). Hence a
good validation design will mix design points close to training sample
points with points far apart. In particular, design points close to
training sample points (or to each other) are highly sensitive to the
emulator's estimated correlation function hyperparameters (such as
correlation length parameters), which is an important aspect of
validation.

A validation design comprises a set of :math:`n^\prime` design points
:math:`D^\prime=\{x^\prime_1,x^\prime_2,\ldots,x^\prime_{n^\prime}\}`.
There is little practical understanding yet about how big :math:`n^\prime`
should be, or how best to achieve the required mix of design points, but
some considerations are as follows.

When the simulator is slow to run, it is important to keep the size of
the validation sample as small as possible. Nevertheless, we typically
need enough validation points to validate against possibly poor
estimation of the different hyperparameters, and we should expect to
have at least as many points as there are hyperparameters. Whereas it is
often suggested that the training sample size, :math:`n`, should be about
10 times the number of inputs, :math:`p`, we tentatively suggest that for
validation we need :math:`n^\prime\ge 3p`.

To achieve a mix of points with high and low predictive variances, one
approach is to generate a design in which the predictive variances of
the :ref:`pivoted Cholesky<ProcPivotedCholesky>` decomposition is
maximised. This could be done by using this criterion in an :ref:`optimised
Latin hypercube<ProcOptimalLHC>` design procedure.

Another idea is to randomly select a number of the original training
design points, and place a validation design point close to each. For
instance, if we selected a training sample point :math:`x` then we could
place the validation point :math:`x^\prime` randomly within the region
where the correlation function :math:`c(x,x^\prime)` (with estimated values
for :math:`\delta`) exceeds 0.8, say. The number of these points should be
at least :math::ref:`p`. Then the remaining points could be chosen as a random
`Latin hypercube<ProcLHC>`.

Additional Comments
-------------------

This is a topic of ongoing research in :ref:`MUCM<DefMUCM>`, and we
expect to add to this page in due course.
