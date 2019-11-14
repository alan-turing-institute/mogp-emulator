.. _AltMultipleOutputsApproach:

Alternatives: Approaches to emulating multiple outputs
======================================================

Overview
--------

The simplest analyses in the :ref:`MUCM<DefMUCM>` toolkit relate to
the :ref:`core problem<DiscCore>`. One of the simplifying
restrictions in the core problem is that we are only interested in one
output from the :ref:`simulator<DefSimulator>`. In the core threads
that use either Gaussian process methods
(:ref:`ThreadCoreGP<ThreadCoreGP>`) or Bayes linear methods
(:ref:`ThreadCoreBL<ThreadCoreBL>`), the core problem is addressed
using the MUCM methodology of first building an
:ref:`emulator<DefEmulator>` and then answering relevant questions
about the simulator output by means of the emulator. In reality,
simulators almost always produce multiple outputs, and we are often
interested in addressing questions concerning more than one output.

One of the most common tasks is to predict multiple outputs of the
simulator at input points where the output has not yet been observed. In
a sense, this can be considered as a question of predicting each output
separately, reducing the problem to several individual core problems.
However, prediction inevitably involves uncertainty and in predicting
several outputs we are concerned about joint uncertainty, represented by
covariances as well as variances. If we only predict each output
separately by the core problem methods of
:ref:`ThreadCoreGP<ThreadCoreGP>` or
:ref:`ThreadCoreBL<ThreadCoreBL>`, we are ignoring covariances
between the outputs. This could cause significant over-estimation of our
uncertainty, and the problem will become more serious as the correlation
between outputs increases. If the outputs are uncorrelated (or
independent) then using separate emulators will give good results.

A related task is to predict one or more functions of the outputs. For
instance, a simulator may have outputs that represent amounts of
rainfall at a number of locations, but we wish to predict the total
rainfall over a region, which is the sum of the rainfall outputs at the
various locations (or perhaps a weighted sum if the locations represent
subregions of different sizes). Or if the simulator outputs the
probability of a natural disaster and its consequence in loss of lives,
then the product of these two outputs is the expected loss of life,
which may be of primary interest. For questions involving functions of
outputs, in the fully Bayesian approach we can use a simulation based
inference procedure, as detailed in
:ref:`ProcPredictMultiOutputFunction<ProcPredictMultiOutputFunction>`.
(This is not generally an option in the Bayes linear approach.) For
linear functions, it may be possible to provide specific methods for
prediction and related tasks.

In general, we can tackle questions of multiple outputs using the same
basic MUCM methodology as for a single output (i.e. we emulate the
various outputs and then use the resulting emulator(s) to address the
relevant tasks) but there are now a number of alternative ways in which
we might emulate multiple simulator outputs.

Consistency
~~~~~~~~~~~

A consideration that might be important in the choice of alternative is
consistency. For instance, if we emulate multiple outputs together by
one of these methods, would the predictions for any single output be the
same as we would have produced if we had just emulated that one output
as in :ref:`ThreadCoreGP<ThreadCoreGP>` or
:ref:`ThreadCoreBL<ThreadCoreBL>`?

More generally, if we build an emulator for multiple outputs and use it
to predict one or more functions of the outputs, would this give the
same predictions as if we had taken the output vectors from our training
sample, computed the relevant function(s) for each output vector and
then fitted an emulator to these? For instance, if we take the training
sample outputs and sum the values of the first two outputs, then fit a
single-output emulator to those sums, would this give the same result as
fitting an emulator for the multiple outputs and then making predictions
about the sum of the first two outputs?

We cannot hope to achieve consistency of this kind in all cases. For
instance, we should not expect to have consistency for predicting
nonlinear functions of outputs. A Gaussian process emulator for the
product of two outputs, for example, will not look like the product of
two Gaussian processes. Some kinds of multiple output emulators, though,
will give consistency for prediction either of individual outputs or of
linear functions of outputs.

Choosing the Alternatives
-------------------------

We present here a number of alternative approaches to emulation of
multiple outputs. In each case we consider the complexity of building
the relevant emulator(s) and of using them to address tasks such as
prediction, uncertainty analysis or sensitivity analysis of some
combination of outputs.

It is worth noting that whatever approach we choose for emulation the
training data will comprise a set of output vectors :math:`f(x_i),
i=1,2,\ldots,n`. Each output vector contains values for all the outputs
of interest.

Independent outputs
~~~~~~~~~~~~~~~~~~~

A simple approach is to assume that the outputs are independent (or in
the Bayes linear framework, just uncorrelated). Then we can build
separate, independent emulators for the different outputs using the
methods set out in the core threads :ref:`ThreadCoreGP<ThreadCoreGP>`
and :ref:`ThreadCoreBL<ThreadCoreBL>`. This allows different mean and
correlation functions to be fitted for each output, and in particular
different outputs can have different correlation length parameters in
their hyperparameter vectors :math:`\delta`. Therefore this approach does
not assume that the input and output correlations are
:ref:`separable<DefSeparable>`.

Using the independent emulators for tasks such as prediction,
uncertainty and sensitivity analyses is set out in
:ref:`ThreadGenericMultipleEmulators<ThreadGenericMultipleEmulators>`,
and is also relatively simple.

The independent outputs approach can also been seen as a special case of
the general multivariate emulator (see below and
:ref:`ThreadVariantMultipleOutputs<ThreadVariantMultipleOutputs>`),
in which the multivariate covariance function is diagonal (see
:ref:`AltMultivariateCovarianceStructures<AltMultivariateCovarianceStructures>`).

A hybrid approach of breaking the outputs into groups such that there is
near independence between groups but we expect correlation within groups
might be considered. Then we emulate each group separately using one of
the methods discussed below. It is not difficult to adapt the procedures
to accommodate this case, but we do not provide explicit procedures in
this version of the toolkit.

Transformation of outputs to independence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Another approach is to transform the outputs so that the resulting
transformed outputs can be considered to be independent (or
uncorrelated), and can then be emulated using independent emulators as
discussed above. If such a transformation can be found, then we can
proceed as above. Since the original outputs are functions of the
transformed outputs, we can address any required task, such as
prediction or sensitivity analysis, using the same methods as described
in the generic thread for combining two or more emulators
(:ref:`ThreadGenericMultipleEmulators<ThreadGenericMultipleEmulators>`).

The only additional questions to address with this approach are (a) how
to find a suitable transformation and (b) how to allow for uncertainty
in the choice of transformation.

We will refer to the transformed outputs which will be emulated
independently as latent outputs, to distinguish these from the original
outputs of the simulator.

 Transformation methods
^^^^^^^^^^^^^^^^^^^^^^

The simplest transformations to consider are linear. In principle, we
could use nonlinear transformations but we do not consider that
possibility here.

One approach is to use the method of principal components, or some other
variance decomposition method, as described in
:ref:`ProcOutputsPrincipalComponents<ProcOutputsPrincipalComponents>`.
These methods in general produce a one-to-one linear transformation,
with the same number of latent outputs as original outputs of interest.
Thus, we can represent the latent outputs as a linear transformation of
original outputs, use this transformation to compute the latent output
values for each output vector in the training sample and proceed to
develop independent single-output emulators for the latent outputs. The
original outputs are then derived using the inverse linear
transformation and we can apply the methods of
:ref:`ThreadGenericMultipleEmulators<ThreadGenericMultipleEmulators>`
for prediction, uncertainty or sensitivity analysis.

We can also consider linear transformations in which there are fewer
latent outputs than original outputs. This possibility arises naturally
when we use principal components, since the essence of that method is to
find a set of latent outputs such that as much as possible of the
variability in the original outputs is explained by the first few latent
outputs. If the remaining latent outputs are very insensitive to
variation in the simulator inputs (over the range explored in the
training sample), then we may consider that there is little value in
emulating these. In practice there will be no guarantee that lower
variance components might be less structured and some sort of method for
looking at correlation structure, for example variograms and
cross-variograms for the latent outputs could be used to provide a
qualitative assessment of the stability of this model.

Formally, we represent the latent outputs that we do not emulate fully
as zero mean GPs with a correlation function that consists only of a
nugget (setting :math::ref:`\nu=1` in
`AltCorrelationFunction<AltCorrelationFunction>`). The variance
hyperparameter :math:`\sigma^2` is equated in each case to the estimated
variance of that principal component. We can then apply the methods of
:ref:`ThreadGenericMultipleEmulators<ThreadGenericMultipleEmulators>`
for prediction, etc.

This kind of dimension reduction can be a very powerful way of managing
the emulation of large numbers of outputs, and remains the only widely
explored method for tackling high dimensional outputs. This form of
dimension reduction is likely to be effective when there might be
expected to be strong correlation between the outputs of the simulator,
as for example is often the case where the output of the simulator is
composed of a grid (or other tessellation / projection) of a spatial or
spatio-temporal field. Indeed, there is often much redundancy in outputs
on a grid since in order for the simulator to resolve a spatial feature
effectively it must be larger than around 5-10 grid lengths (depending
on the simulator details), otherwise it can induce numerical
instabilities in the simulator, or be poorly modelled.

 Uncertainty in transformation choice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When we have applied one of the above methods to find a suitable
transformation, we may wish to acknowledge formally that the
transformation has only been estimated. We are supposing here that there
is a 'true' transformation such that the latent outputs would be
independent, but that we are unsure of what that transformation would
be. We then need to treat the transformation as providing additional
uncertain hyperparameters. For instance, if we are selecting a linear
transformation, the coefficients of that transformation become the new
hyperparameters. We can in principle represent this uncertainty through
sample values of the transformation hyperparameters, with which we
augment the sampled values of any other hyperparameters. Although in
principle the necessary hyperparameter samples can be derived from the
statistical procedure used to estimate the transformation, and this
uncertainty can then be taken account of as described in
:ref:`ThreadGenericMultipleEmulators<ThreadGenericMultipleEmulators>`,
this will not be simple in practice. Furthermore, the estimation of the
transformation hyperparameters should be done jointly with the fitting
of the emulator hyperparameters to avoid double use of the data. We do
not offer detailed procedures for this in the current version of the
toolkit, although we may do so in a future release.

The alternative is to ignore such additional uncertainty. We have
selected a transformation without any particular belief that there is a
'true' transformation that would make the latent outputs independent.
Rather, we think that this transformation will simply make the
assumption of independence that we necessarily make when developing
separate emulators a better representation of the simulator. Under this
view questions of double use of the data are also ignored.

Outputs as extra input dimension(s)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some simulators, the outputs represent values of some generic
real-world property at different locations and/or times. For instance, a
simulator of a vehicle crash may output velocities or forces on the
vehicle as a time series at many time points during the simulated crash.
As another example, a simulator of the spread of an oil-spill in the
ocean may output the concentration of oil at different geographic
locations in the ocean and at different times.

In such cases, an alternative way to treat the multiple outputs is as a
single output but with one or more extra inputs.

-  In the crash simulator, we add the time as an input. Then if we are
   interested in the velocity output we represent the simulator as
   producing a single velocity output when given all the original inputs
   plus a time input value. This is now a single output simulator and
   can be tackled simply by the methods of
   :ref:`ThreadCoreGP<ThreadCoreGP>` or
   :ref:`ThreadCoreBL<ThreadCoreBL>`.
-  Similarly, if we consider the oil-spill simulator, we need to add
   three or four inputs - the time value and the geographic coordinates
   (either 2 of these or, if the simulation also covers depth within the
   ocean, 3). Then again we have a single output simulator and can use
   one of the core threads to emulate it.
-  Note also that if we are interested in both velocity and force in the
   crash simulator, then adding a time input leaves a simulator with two
   outputs, velocity and force. We therefore still need to use methods
   for emulating multiple inputs, but the device of making time an input
   has reduced the number of outputs enormously (from twice the number
   of time points to just two).

This approach is clearly very simple, but we need also to consider what
restrictions it implies for the covariance structure in terms of the
original outputs. In theory, there need not be any restriction, since in
principle the covariance function of the single-output emulator with the
new inputs can be defined to represent any joint covariance structure.
In practice, however, the covariance structures that we generally assume
make quite strong restrictive assumptions. As discussed in
:ref:`DiscCovarianceFunction<DiscCovarianceFunction>` we invariably
assume a constant variance, which implies that all of the original
outputs have the same variance. This may be reasonable, but often is not
- for instance, we may expect changes in the other simulator inputs to
have increasing effects on oil-spill concentrations over time.
Furthermore, the usual choices of correlation function considered in
:ref:`AltCorrelationFunction<AltCorrelationFunction>` make strong
assumptions. For instance, the Gaussian and power exponential
correlation functions have the effect of (a) assuming separability
between the original inputs and outputs, and (b) assuming a form of
correlation function between the original outputs that is unlikely to be
appropriate, particularly in the case of time series.

So the great simplicity of the additional input approach is balanced by
restrictive assumptions that may lead to poor emulation (or may require
much larger training samples to achieve adequate emulation).

The multivariate emulator
~~~~~~~~~~~~~~~~~~~~~~~~~

In order to relax some of the restrictions imposed by the additional
input approach, we need to directly emulate all the outputs
simultaneously using a multivariate emulator. The multivariate emulator
is an extension of the :ref:`Gaussian process emulator for the core
problem<ProcBuildCoreGP>`, based on a :ref:`multivariate Gaussian
process<DefMultivariateGP>`. The key differences between the
multivariate emulator and the additional input approach are (a) the
multivariate emulator does not restrict the outputs to all having the
same variance, and (b) the multivariate emulator does not impose a
parametric form (e.g. Gaussian or power exponential) on the
between-output correlations.

The methodology for building and using a multivariate emulator is
presented in
:ref:`ThreadVariantMultipleOutputs<ThreadVariantMultipleOutputs>`. A
key issue is in choosing the covariance function for the :ref:`multivariate
Gaussian process<DefMultivariateGP>`. The various multivariate
covariance functions that are available are discussed in
:ref:`AltMultivariateCovarianceStructures<AltMultivariateCovarianceStructures>`.

The dynamic emulator
~~~~~~~~~~~~~~~~~~~~

A quite different approach in the case of :ref:`dynamic<DefDynamic>`
simulators which produce time series of outputs by recursively updating
a state vector is to emulate the single step simulator. A separate
thread, :ref:`ThreadVariantDynamic<ThreadVariantDynamic>`, deals with
dynamic emulation.

Additional Comments, References, and Links
------------------------------------------

The multivariate emulator with a separable covariance function is
developed and contrasted with the approach of treating outputs as an
extra input in

Conti, S. and O'Hagan, A. (2009). Bayesian emulation of complex
multi-output and dynamic computer models. Journal of Statistical
Planning and Inference. `doi:
10.1016/j.jspi.2009.08.006 <http://dx.doi.org/10.1016/j.jspi.2009.08.006>`__

In the following paper, the authors develop a dynamic linear model
representation for outputs that are in the form of a time series. A
special case of this is equivalent to (a) formulating the outputs as a
time input and (b) the multivariate emulator with a separable covariance
function, but with a proper time series structure for the covariance
between outputs. However, their model is made more general by allowing
the time series model parameters to vary.

Liu, F, and West, M. (2009). A dynamic modelling strategy for Bayesian
computer model emulation. Bayesian Analysis 4, 393-412. Published online
at http://ba.stat.cmu.edu/journal/2009/vol04/issue02/liu.pdf.
