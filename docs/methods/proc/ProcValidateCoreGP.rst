.. _ProcValidateCoreGP:

Procedure: Validate a Gaussian process emulator
===============================================

Description and Background
--------------------------

Once an :ref:`emulator<DefEmulator>` has been built, under the fully
:ref:`Bayesian<DefBayesian>` :ref:`Gaussian process<DefGP>`
approach, using the procedure in page
:ref:`ProcBuildCoreGP<ProcBuildCoreGP>`, it is important to
:ref:`validate<DefValidation>` it. Validation involves checking
whether the predictions that the emulator makes about the
:ref:`simulator<DefSimulator>` output accord with actual observation
of runs of the simulator. Since the emulator has been built using a
:ref:`training sample<DefTrainingSample>` of runs, it will inevitably
predict those correctly. Hence validation uses an additional set of
runs, the validation sample.

We describe here the process of setting up a validation sample, using
the validation data to test the emulator and interpreting the results of
the tests.

We consider here an emulator for the :ref:`core problem<DiscCore>`,
and in particular we are only concerned with one simulator output.

Inputs
------

-  Emulator, as derived in page
   :ref:`ProcBuildCoreGP<ProcBuildCoreGP>`.
-  The input configurations :math:`D=\{x_1,x_2,\ldots,x_n\}` at which the
   simulator was run to produce the training data from which the
   emulator was built.

Outputs
-------

-  A conclusion, either that the emulator is valid or that it is not
   valid.
-  If the emulator is deemed not valid, then indications for how to
   improve it.

Procedure
---------

The validation sample
~~~~~~~~~~~~~~~~~~~~~

The validation sample must be distinct from the training sample that was
used to build the emulator. One approach is to reserve part of the
training data for validation, and to build the emulator only using the
rest of the training data. However, the usual approach to designing a
training sample (typically to use points that are well spread out,
through some kind of space-filling design, see the alternatives page on
training sample design (:ref:`AltCoreDesign<AltCoreDesign>`)) does
not generally provide subsets that are good for validation. It is
preferable to develop a validation sample design after building the
emulator, taking into account the training sample design :math:`D` and the
estimated values of the correlation function
:ref:`hyperparameters<DefHyperparameter>` :math:`\delta`.

Validation sample design is discussed in page
:ref:`DiscCoreValidationDesign<DiscCoreValidationDesign>`. We denote
the validation design by
:math:`D^\prime=\{x^\prime_1,x^\prime_2,\ldots,x^\prime_{n^\prime}\}`, with
:math:`n^\prime` points. The simulator is run at each of the validation
points to produce the output vector
:math:`f(D^\prime)=(f(x^\prime_1),f(x^\prime_2),\ldots
f(x^\prime_{n^\prime})^T)`, where :math:`f(x^\prime_j)` is the simulator
output from the run with input vector :math:`x^\prime_j`.

We then need to evaluate the emulator's predictions for
:math:`f(D^\prime)`. For the purposes of our diagnostics, it will be enough
to evaluate means, variances and covariances. The procedure for
computing these moments is given in the procedure page for predicting
simulator outputs (:ref:`ProcPredictGP<ProcPredictGP>`). The
procedure is particularly simple in the case of a linear mean function,
weak prior information on hyperparameters :math:`\beta` and :math:`\sigma^2`,
and a single posterior estimate of :math:`\delta`, since then the required
moments are simply given by the functions :math::ref:`m^*(.)` and :math:`v^*(.,.)`
given in `ProcBuildCoreGP<ProcBuildCoreGP>` (evaluated at the
estimate of :math:`\delta`). In fact, where we have a linear mean function
and weak prior information on the other hyperparameters, it is
recommended that only a single estimate of :math:`\delta` is computed prior
to validation. If the validation tests declare the emulator to be valid,
*then* it may be worthwhile to go back and derive a sample of
:math:`\delta` values for subsequent use.

We denote the predictive means and covariances of the validation data by
:math:`m^*(x^\prime_j)` and :math:`v^*(x^\prime_j,x^\prime_{j^\prime})`,
noting that the predictive variance of the j-th point is
:math:`v^*(x^\prime_j,x^\prime_j)`. We let :math:`m^*` be the mean vector
:math:`(m^*(x^\prime_1),
m^*(x^\prime_2),\ldots,m^*(x^\prime_{n^\prime}))^T` and :math:`V^*` be the
covariance matrix with :math:`(j,j^\prime)`-th element
:math:`v^*(x^\prime_j,x^\prime_{j^\prime})`.

Possible causes of validation failure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before presenting the diagnostics it is useful to consider the various
ways in which an emulator may fail to make valid predictions. Although
the GP is a very flexible way to represent prior knowledge about the
computer model, the GP emulator can give poor predictions of simulator
outputs for at least two basic reasons. First, the assumption of
particular mean and correlation functions may be inappropriate. Second,
even if these assumptions are reasonable there are various
hyperparameters to be estimated, and a bad or unfortunate choice of
training dataset may suggest inappropriate values for these parameters.
In the case of the correlation function parameters :math:`\delta`, where we
condition on fixed estimates, we may also make a poor choice of
estimate.

If the assumed form of the mean function is wrong, for instance because
inappropriate regressors have been used in a linear form (see the
alternatives page on emulator prior mean function
(:ref:`AltMeanFunction<AltMeanFunction>`)), or if the hyperparameters
:math:`\beta` have been poorly estimated, then the emulator predictions may
be systematically too low or too high in some regions of the input
space.

In the various forms of correlation function considered in the
discussion page on GP covariance function
(:ref:`DiscCovarianceFunction<DiscCovarianceFunction>`), and in the
alternatives page on emulator prior correlation function
(:ref:`AltCorrelationFunction<AltCorrelationFunction>`) all involve
stationarity, implying that we expect the simulator output to respond
with similar degrees of smoothness and variability at all points in the
input space. In practice, simulators may respond much more rapidly to
changes in the inputs at some parts of the space than others. In case of
such non-stationarity, credible intervals of emulator predictions can be
too wide in regions of low responsiveness or too narrow in regions where
the response is more dynamic.

Finally, although the form of the correlation function may be
appropriate, we may estimate the parameters :math:`\sigma^2` and
:math:`\delta` poorly. When we have incorrect estimation of the variance
(\(\sigma^2`), the credible intervals of the emulator predictions are
systematically too wide or too narrow. Poor estimation of the
correlation parameters (\(\delta`) leads to credible intervals that are
too wide or too narrow in the neighbourhood of the training data points.

Validation diagnostics
~~~~~~~~~~~~~~~~~~~~~~

We present here a basic set of validation diagnostics. In each case we
present the diagnostic itself and a reference probability distribution
against which the observed value of the diagnostic should be compared.
If the observed value is extreme relative to that distribution, i.e. it
is far out in one or other tail of the reference distribution, then this
indicates a validation failure. It is a matter of judgement how extreme
a validation diagnostic needs to be before declaring a validation
failure. It is common to use the upper and lower 5% points of the
reference distribution as suggestive of a failure, with the upper and
lower 0.1% points corresponding to clear evidence of failure.

We discuss the implications and interpretations of each possible
validation failure and the extent to which these should lead to a
decision that the emulator is not valid.

Reference distributions are approximate, but the approximations are good
enough for the purposes of identifying validation failures.

 Mahalanobis distance
^^^^^^^^^^^^^^^^^^^^

The Mahalanobis distance diagnostic is

:math:`M = (f(D^\prime)-m^*)^T(V^*)^{-1}(f(D^\prime)-m^*)\,.`

The reference distribution for :math:`M` is the scaled F-Snedecor
distribution with :math:`n^\prime` and :math:`(n - q)` degrees of freedom,
where :math:`q` is the dimension of the :math:`h(\cdot)` function. The mean of
this reference distribution is

:math:`\textrm{E}[M] = n^\prime`

and the variance is

:math:`\textrm{Var}[M] = \\frac{2n^{\prime}(n^{\prime}+n-q-2)}{n-q-4}`

:math:`M` is a measure of overall fit. If too large it suggests that the
emulator is over-confident, in the sense that the uncertainty expressed
in :math:`V^*` is too low compared to the observed differences between the
observed :math:`f(D^\prime)` and the predictive means :math:`m^*`. This in
turn may suggest poor estimation of :math:`\beta`, under-estimation of
:math:`\sigma^2` or generally over-estimated correlation length parameters
:math:`\delta`.

Conversely, if :math:`M` is too small it suggests that the emulator is
underconfident, which in turn suggests over-estimation of :math:`\sigma^2`
or generally under-estimated correlation length parameters.

An extreme value of this diagnostic should be investigated further
through the following more targeted diagnostics. Whilst a moderate value
of :math:`M` generally suggests that the emulator is valid, it is prudent
to engage anyway in these further diagnostic checks, because they may
bring out areas of concern.

 Individual standardised errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The individual standardised errors are, for :math:`j=1,2,\ldots,n^\prime`,

:math:`e_j =
\\frac{f(x^\prime_j)-m^*(x_j^\prime)}{\sqrt{v^*(x^\prime_j,x^\prime_j)}}\,.`

Each of these is a validation diagnostic in its own right with reference
distribution the standard normal distribution, :math:`{\cal N}(0,1)`. When
comparing with the reference distribution, it is important to remember
that we are making many tests and if :math:`n^\prime` is large enough then
we certainly expect some moderately extreme values by pure chance even
if the emulator is valid. We are therefore looking for individual very
extreme values (larger than 3 in absolute value, say) or patterns of
extreme values.

Isolated very extreme :math:`e_j` values suggest a local irregular
behaviour of the simulator in the region of :math:`x^\prime_j`. Clusters of
extreme values whose input values :math:`x^\prime_j` lie in a particular
region of the input space suggest non-stationarity of the simulator in
that region.

If large values tend to correspond to :math:`x^\prime_j` values close to
training sample design points this suggests over-estimation of
correlation lengths. It should be noted that groups of unusually *small*
values of :math:`e_j` close to training sample design points suggest
under-estimation of correlation lengths.

It is important to note, however, that the :math:`e_j` values are not
independent, and this makes interpretation of apparent patterns of
individual errors difficult. The next group of diagnostics, the pivoted
Cholesky errors, are the most promising of a number of ways to generate
independent standardised errors.

 Pivoted Cholesky errors
^^^^^^^^^^^^^^^^^^^^^^^

The well-known Cholesky decomposition of a positive-definite matrix
yields a kind of square-root matrix. In our diagnostics we use a version
of this called the pivoted Cholesky decomposition. The procedure for
this is given in page
:ref:`ProcPivotedCholesky<ProcPivotedCholesky>`. Let :math:`C` be the
pivoted Cholesky decomposition of :math:`V^*` and let

:math:`t = C^{-1} (f(D^\prime)-m^*)\,.`

Then we consider each of the individual elements :math:`t_k` of this vector
to be a validation diagnostics, for :math:`k=1,2,\ldots,n^\prime`. The
reference distribution for each :math:`t_k` is standard normal.

A property of the pivoted Cholesky decomposition is that each :math:`t_k`
is associated with a particular validation sample value, but the
ordering of these diagnostics is different from the ordering of the
validation dataset. Thus, for instance, the first diagnostic :math:`t_1`
will not generally correspond to the first validation data point
:math:`x^\prime_1`. The ordering instead assists with identifying
particular kinds of emulator failure.

Extreme values of :math:`t_k` early in the sequence (low :math:`k`) suggest
under-estimation of :math:`\sigma^2`, while if the values early in the
sequence are unusually small then this suggests over-estimation of
:math:`\sigma^2`. When these extremes or unusually low values cluster
instead at the end of the sequence (high :math:`k`) it suggests
over-/under-estimation of correlation lengths.

Response to diagnostics
~~~~~~~~~~~~~~~~~~~~~~~

If there are no validation failures, or only relatively minor failures,
we will generally declare the emulator to be valid. This does not, of
course, constitute proof of validity. Subsequent usage may yet uncover
problems with the emulator. Nevertheless, we would proceed on the basis
that the emulator appears to be valid. In practice, it is rare to have
no validation failures - local inhomogeneity of the simulator's
behaviour will almost always lead to some emulation difficulties.
Declaring validity when minor validation errors have arisen is a
pragmatic decision.

When failures cannot be ignored because they are too extreme or too
numerous, the emulator should not be used as it stands. Instead, it
should be rebuilt with more data. Changes to the assumed mean and
correlation functions may also be indicated.

Rebuilding with additional data is, in one sense at least,
straightforward since we have the validation sample data which can
simply be added to the original training sample data. We now regard the
combined data as our training sample and proceed to rebuild the
emulator. However, it should be noted that we will need additional
validation data with which the validate the rebuilt emulator.

Also, the diagnostics may indicate adding new data in particular regions
of the input space, if problems have been noted in those regions.
Problems with the correlation parameters may suggest including extra
training data points that are relatively close to either the original
training data points or the validation points.

Additional Comments
-------------------

These diagnostics, and some others, were developed in
:ref:`MUCM<DefMUCM>` and presented in

Bastos, L. S. and O'Hagan, A. (2008). Diagnostics for Gaussian process
emulators. MUCM Technical Report 08/02. (May be downloaded from the
:ref:`MUCM
website<http://mucm.group.shef.ac.uk/Pages/Dissemination/Dissemination_Papers_Technical>`.)

Validation is something of an evolving art. We hope to extend the
discussion here as we gain more experience in :ref:`MUCM<DefMUCM>`
with the diagnostics.
