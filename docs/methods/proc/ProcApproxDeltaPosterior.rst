.. _ProcApproxDeltaPosterior:

Procedure: Multivariate lognormal approximation for correlation hyperparameters
===============================================================================

Description and Background
--------------------------

The posterior distribution :math:`\pi^*(\delta)` of the
:ref:`hyperparameter<DefHyperparameter>` vector :math:`\delta` is given
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
-  The mode of the posterior :math:`\hat{\delta}` as defined in the
   discussion page on finding the posterior mode of correlation lengths
   (:ref:`DiscPostModeDelta<DiscPostModeDelta>`).
-  The :math:`p \times p` Hessian matrix :math:`\displaystyle
   \frac{\partial ^2 g(\tau)}{\partial \tau^2}` with (k,l)-th entry
   :math:`\displaystyle \frac{\partial ^2 g(\tau)}{\partial \tau_l
   \partial \tau_k}`, as defined in
   :ref:`DiscPostModeDelta<DiscPostModeDelta>`.

Outputs
-------

-  A set of :math:`s` samples for the correlation lengths :math:`\delta`,
   denoted as :math:`\tilde{\delta}`.
-  A posterior mean :math:`\tilde{m}^*(\cdot)` and a covariance
   function :math:`\tilde{u}^*(\cdot,\cdot)`, conditioned on the
   samples :math:`\tilde{\delta}`

Procedure
---------

-  Define :math:`\displaystyle V = -\left(\frac{\partial ^2
   g(\tau)}{\partial \tau^2}\right)^{-1}`. Draw :math:`s` samples
   from the :math:`p`-variate normal distribution :math:`{\cal N}
   (\hat{\tau},V)`, call these samples :math:`\tilde{\tau}`.

-  Calculate the samples :math:`\tilde{\delta}` as
   :math:`\tilde{\delta} = \exp(\tilde{\tau}/2)`.

-  Given the set of :math:`s` samples :math:`\tilde{\delta}`, the
   posterior mean and variance :math:`\tilde{m}^*(\cdot)`,
   :math:`\tilde{u}^*(\cdot,\cdot)` can be calculated with the same
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
the correlation lengths :math:`\pi^*_{\delta}(\delta)` can be very
flat with respect to that input and obtain its maximum for a large value
of :math:`\delta`. This can cause the respective entry of the
matrix :math:`V` to be very large, and the samples :math:`\tilde{\delta}`
that correspond to this input to have both
unrealistically large and small values. An inspection of the samples
that are returned by the above procedure is recommended, especially in
high dimensional input problems, where less active inputs are likely to
exist. If the sampled correlation lengths that correspond to one or more
inputs are found to have very large (e.g. >50) and very small (e.g.
:math:`<0.5`) values at the same time, a potential remedy could be to fix the
values of these samples to the respective entries of the :math:`\hat{\delta}` vector.

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
