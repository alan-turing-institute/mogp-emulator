.. _AltBLPriors:

Alternatives: Prior specification for BL hyperparameters
========================================================

Overview
--------

In the fully :ref:`Bayes linear<DefBayesLinear>` approach to
emulating a complex :ref:`simulator<DefSimulator>`, the
:ref:`emulator<DefEmulator>` is formulated to represent prior
knowledge of the simulator in terms of a :ref:`second-order belief
specification<DefSecondOrderSpec>`. The BL prior specification
requires the specification of beliefs about some
:ref:`hyperparameters<DefHyperparameter>`, as discussed in the
alternatives page on emulator prior mean function
(:ref:`AltMeanFunction<AltMeanFunction>`), the discussion page on the
GP covariance function
(:ref:`DiscCovarianceFunction<DiscCovarianceFunction>`) and the
alternatives page on emulator prior correlation function
(:ref:`AltCorrelationFunction<AltCorrelationFunction>`).
Specifically, in the :ref:`core problem<DiscCore>` that is the
subject of the core threads (:ref:`ThreadCoreBL<ThreadCoreBL>`,
:ref:`ThreadCoreGP<ThreadCoreGP>`) a vector :math:`\beta` defines the
detailed form of the mean function, a scalar :math:`\sigma^2` quantifies
the uncertainty or variability of the simulator around the prior mean
function, while :math:`\delta` is a vector of hyperparameters defining
details of the correlation function. Threads that deal with variations
on the basic core problem may introduce further hyperparameters.

A Bayes linear analysis requires hyperparameters to be given prior
expectations, variances and covariances. We consider here ways to
specify these prior beliefs for the hyperparameters of the core problem.
Prior specifications for other hyperparameters are addressed in the
relevant variant thread. Hyperparameters may be handled differently in
the fully :ref:`Bayesian<DefBayesian>` approach - see
:ref:`ThreadCoreGP<ThreadCoreGP>`.

Choosing the Alternatives
-------------------------

The prior beliefs should be chosen to represent whatever prior knowledge
the analyst has about the hyperparameters. However, the prior
distributions will be updated with the information from a set of
training runs, and if there is substantial information in the training
data about one or more of the hyperparameters then the prior information
about those hyperparameters may be irrelevant.

In general, a Bayes linear specification requires statements of
second-order beliefs for all uncertain quantities. In the current
version of this Toolkit, the Bayes linear emulation approach does not
consider the situation where :math:`\sigma^2` and :math:`\delta` are
uncertain, and so we require the following:

-  :math:`\text{E}[\beta_i]`, :math:`\text{Var}[\beta_i]`,
   :math:`\text{Cov}[\beta_i,\beta_j]` - expectations, variances and
   covariances for each coefficient :math:`\beta_i`, and covariances
   between every pair of coefficients :math:`(\beta_i,\beta_j), i\neq j`
-  :math:`\sigma^2=\text{Var}[w(x)]` - the variance of the residual
   stochastic process
-  :math:`\delta` - a value for the hyperparameters of the correlation
   function

The Nature of the Alternatives
------------------------------

Priors for :math:`\beta`
~~~~~~~~~~~~~~~~~~~~~~~~~

Given a specified form for the basis functions :math:`h(x)` of :math:`m(x)` as
described in the alternatives page on basis functions for the emulator
mean (:ref:`AltBasisFunctions<AltBasisFunctions>`), we must specify
expectation and variance for each coefficient :math:`\beta_i` and a
covariance between every pair :math:`(\beta_i,\beta_j)`.

As with the basis functions :math:`h(x)`, there are two primary means of
obtaining a belief specification for :math:`\beta`.

#. **Expert-led specification** - the specification can be made directly
   by an expert using methods such as

   a. Intuitive understanding of the magnitude and impact of the
      physical effects represented by :math:`h(x)` leading to a direct
      quantification of expectations, variances and covariances.
   b. Assessing the difference between the model under study and another
      well-understood model such as a fast approximate version or an
      earlier version of the same simulator. In this approach, we can
      combine the known information about the mean behaviour of the
      second simulator with the belief statements about the differences
      between the two simulator to construct an appropriate belief
      specification for the hyperparameters -- see :ref:`multilevel
      emulation<DefMultilevelEmulation>`.

#. **Data-driven specification** - when prior beliefs are weak and we
   have ample model evaluations, then prior values for :math:`\beta` are
   typically not required and we can replace adjusted values for
   :math:`\beta` with empirical estimates, :math:`\hat{\beta}`, obtained by
   fitting the linear regression :math:`f(x)=h(x)^T\beta`. Our uncertainty
   statements about :math:`\beta` can then be deduced from the "estimation
   error" associated with :math:`\hat{\beta}`.

Priors for :math:`\sigma^2`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current version of the Toolkit requires a point value for the
variance about the emulator mean, :math:`\sigma^2`. This corresponds
directly to making a specification about :math:`\text{Var}[w(x)]`. As with
the model coefficients above, there are two possible approaches to
making such a quantification. An expert could make the specification by
directly quantifying the magnitude of :math:`\sigma^2`. Alternatively, an
expert assessment of the expected prior adequacy of the mean function at
representing the variation in the simulator outputs can be combined with
information on the variation of the simulator output, which allows for
the deduction of a value of :math:`\sigma^2`. In the case of a data-driven
assessment, the estimate for the residual variance :math:`\hat{\sigma}^2`
can be used.

In subsequent versions of the toolkit, Bayes linear methods will be
developed for :ref:`learning<DefBLVarianceLearning>` about
:math:`\sigma^2` in the emulation process. This will require making prior
specifications about the squared emulator residuals.

Priors for :math:`\delta`
~~~~~~~~~~~~~~~~~~~~~~~~~~

Specification of correlation function hyperparameters is a more
challenging task. Direct elicitation can be difficult as the
hyperparameter :math:`\delta` is hard to conceptualise - the alternatives
page on prior distributions for GP hyperparameters
(:ref:`AltGPPriors<AltGPPriors>`) provides some discussion on this
topic, with particular application to the Gaussian correlation function.
Alternatively, when given a large collection of simulator runs then
:math:`\delta` can be crudely estimated using methods such as
:ref:`variogram<ProcVariogram>` fitting on the empirical residuals.

Assessing and updating uncertainties about :math:`\delta` raises both
conceptual and technical problems as methods which would be optimal for
assessing such parameters given realisations drawn from a corresponding
stochastic process may prove to be highly non-robust when applied to
functional computer output which is only represented very approximately
by such a process. Methods for approaching this problem will appear in a
subsequent version of the toolkit.
