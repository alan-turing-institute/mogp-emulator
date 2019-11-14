.. _ProcApproximateUpdateDynamicMeanandVariance:

Procedure: Use simulation to recursively update the dynamic emulator mean and variance in the approximation method
==================================================================================================================

Description and Background
--------------------------

This page is concerned with task of :ref:`emulating<DefEmulator>` a
:ref:`dynamic simulator<DefDynamic>`, as set out in the variant
thread on dynamic emulation
(:ref:`ThreadVariantDynamic<ThreadVariantDynamic>`).

The approximation procedure for iterating the single step emulator
(:ref:`ProcApproximateIterateSingleStepEmulator<ProcApproximateIterateSingleStepEmulator>`)
recursively defines

:math:` \\mu_{t+1}= \\mathrm{E}[ m^*(w_t,a_{t+1},\phi)|f(D)] \`,

:math:` V_{t+1}= \\mathrm{E}[
v^*\{(w_t,a_{t+1},\phi),(w_t,a_{t+1},\phi)\}|f(D)] +
\\mathrm{Var}[m^*(w_t,a_{t+1},\phi)|f(D)] \`,

where the expectations and variances are taken with respect to :math:` w_{t}
\:ref:`, with :math:` w_{t} \\sim N_r(\mu_{t},V_{t}) \`. If the `single
step<DefSingleStepFunction>` emulator has a linear mean and a
separable Gaussian covariance function, then :math:` \\mu_{t+1} \` and :math:`
V_{t+1} \` can be computed explicitly, as described in the procedure
page for recursively updating the dynamic emulator mean and variance
(:ref:`ProcUpdateDynamicMeanAndVariance<ProcUpdateDynamicMeanAndVariance>`).
Otherwise, simulation can be used, which we describe here.

Inputs
------

-  :math:` \\mu_{t} \` and :math:` V_{t} \`
-  The single step emulator, conditioned on training inputs :math:`D \` and
   outputs :math:`f(D)`, and hyperparameters :math:`\theta`, with posterior
   mean and covariance functions :math:`m^*(.)` and :math:`v^*(.,.) \`
   respectively.

Outputs
-------

-  Estimates of :math:` \\mu_{t+1} \` and :math:` V_{t+1} \`

Procedure
---------

We describe a Monte Carlo procedure using :math:`N` Monte Carlo iterations.
For discussion of the choice of :math::ref:`N`, see the discussion page on Monte
Carlo estimation (`DiscMonteCarlo<DiscMonteCarlo>`).

#. For :math:`i=1,\ldots,N \`, sample :math:`w_t^{(i)}` from :math:`N(\mu_t,V_t)`
#. Estimate :math:`\mu_{t+1} \` by
   :math:`\hat{\mu}_{t+1}=\frac{1}{N}\sum_{i=1}^N
   m^*(w_t^{(i)},a_{t+1},\phi)`
#. Estimate :math:`V_{t+1} \` by :math:`\hat{V}_{t+1}=\frac{1}{N}\sum_{i=1}^N
   v^*\{(w_t^{(i)},a_{t+1},\phi),(w_t^{(i)},a_{t+1},\phi)\}
   +\frac{1}{N-1}\sum_{i=1}^N\left(m^*(w_t^{(i)},a_{t+1},\phi)-\hat{\mu}_{t+1}\right)^2
   \`
