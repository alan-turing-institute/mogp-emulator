.. _ProcApproxDeltaPosterior:

Procedure: Multivariate lognormal approximation for correlation hyperparameters
===============================================================================

Description and Background
--------------------------

The posterior distribution :math::ref:`\pi^*(\delta)` of the
`hyperparameter<DefHyperparameter>` vector :math:`\delta` is given
in the procedure page for building a Gaussian process emulator for the
core problem (:ref:`ProcBuildCoreGP<ProcBuildCoreGP>`) in the case of
the :ref:`core problem<DiscCore>` (with linear mean function and weak
prior information). The ideal way to compute the emulator in this case
is to generate a sample of values of :math:`\delta` from this distribution,
but that is itself a complex computational task. We present here a
simpler method based on a lognormal approximation to :math:`\pi^*(\delta)`.

Inputs
------

-  An emulator as defined in :ref:`ProcBuildCoreGP<ProcBuildCoreGP>`,
   using a linear mean and a weak prior.
-  The mode of the posterior :math:`\strut \\hat{\delta}` as defined in the
   discussion page on finding the posterior mode of correlation lengths
   (:ref:`DiscPostModeDelta<DiscPostModeDelta>`).
-  The :math:`\strut p \\times p` Hessian matrix :math:`\displaystyle
   \\frac{\partial ^2 g(\tau)}{\partial \\tau^2}` with (k,l)-th entry
   :math:`\displaystyle \\frac{\partial ^2 g(\tau)}{\partial \\tau_l
   \\partial \\tau_k}:ref:`, as defined in
   `DiscPostModeDelta<DiscPostModeDelta>`.

Outputs
-------

-  A set of :math:`\strut s` samples for the correlation lengths :math:`\strut
   \\delta`, denoted as :math:`\tilde{\delta}`.
-  A posterior mean :math:`\strut \\tilde{m}^*(\cdot)` and a covariance
   function :math:`\strut \\tilde{u}^*(\cdot,\cdot)`, conditioned on the
   samples :math:`\strut \\tilde{\delta}`

Procedure
---------

-  Define :math:`\displaystyle V = -\left(\frac{\partial ^2
   g(\tau)}{\partial \\tau^2}\right)^{-1}`. Draw :math:`\strut s` samples
   from the p-variate normal distribution :math:`\strut {\cal N}
   (\hat{\tau},V)`, call these samples :math:`\strut \\tilde{\tau}`.

-  Calculate the samples :math:`\strut \\tilde{\delta}` as :math:`\strut
   \\tilde{\delta} = \\exp(\tilde{\tau}/2)`.

-  Given the set of :math:`\strut s` samples :math:`\tilde{\delta}`, the
   posterior mean and variance :math:`\strut \\tilde{m}^*(\cdot)`,
   :math:`\strut \\tilde{u}^*(\cdot,\cdot)` can be calculated with the same
   formulae given in the procedure page for sampling the posterior
   distribution of the correlation lengths
   (:ref:`ProcMCMCDeltaCoreGP<ProcMCMCDeltaCoreGP>`), or in more
   detail in the procedure page for predicting the simulator's outputs
   using a GP emulator (:ref:`ProcPredictGP<ProcPredictGP>`).

Additional Comments
-------------------

Most standard statistical computing packages have facilities for taking
random samples from a multivariate normal distribution.

When an input is not particularly active, the posterior distribution of
the correlation lengths :math:`\strut \\pi^*_{\delta}(\delta)` can be very
flat with respect to that input and obtain its maximum for a large value
of :math:`\strut \\delta`. This can cause the respective entry of the
matrix :math:`\strut V` to be very large, and the samples :math:`\strut
\\tilde{\delta}` that correspond to this input to have both
unrealistically large and small values. An inspection of the samples
that are returned by the above procedure is recommended, especially in
high dimensional input problems, where less active inputs are likely to
exist. If the sampled correlation lengths that correspond to one or more
inputs are found to have very large (e.g. >50) and very small (e.g.
<0.5) values at the same time, a potential remedy could be to fix the
values of these samples to the respective entries of the :math:`\strut
\\hat{\delta}` vector.

References
----------

This method is introduced in the following report.

-  Nagy B., Loeppky J.L. and Welch W.J. (2007). Fast Bayesian Inference
   for Gaussian Process Models. Technical Report 230, Department of
   Statistics, University of British Columbia.

Note however that the authors indicate also that the method works well
when the correlation function has the Gaussian form but may not work so
well in the case of the exponential power form (see the alternatives
page on emulator prior correlation function
(:ref:`AltCorrelationFunction<AltCorrelationFunction>`):

-  Nagy B., Loeppky J.L. and Welch W.J. (2007). Correlation
   parameterization in random function models to improve normal
   approximation of the likelihood or posterior. Technical Report 229,
   Department of Statistics, University of British Columbia.
