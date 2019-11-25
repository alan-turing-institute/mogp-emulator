.. _ProcApproximateIterateSingleStepEmulator:

Procedure: Iterate the single step emulator using an approximation approach
===========================================================================

Description and Background
--------------------------

This page is concerned with task of :ref:`emulating<DefEmulator>` a
:ref:`dynamic simulator<DefDynamic>`, as set out in the variant
thread for dynamic emulation
(:ref:`ThreadVariantDynamic<ThreadVariantDynamic>`).

We have an emulator for the :ref:`single step
function<DefSingleStepFunction>` :math:`w_t=f(w_{t-1},a_t,\phi)`,
and wish to predict the full time series :math:`w_1,\ldots,w_T` for
a specified initial :ref:`state variable<DefStateVector>` :math:`w_0`,
time series of :ref:`forcing variables<DefForcingInput>`
:math:`a_1,\ldots,a_T` and simulator parameters :math:`\phi`. It is not
possible to analytically derive a distribution for :math:`w_1,\ldots,w_T`
if :math:`f(\cdot)` is modelled as a :ref:`Gaussian Process<DefGP>`,
so here we use an approximation based on the normal distribution to
estimate the marginal distribution of each of :math:`w_1,\ldots,w_T`.

Inputs
------

-  An emulator for the single step function :math:`w_t=f(w_{t-1},a_t,\phi)`,
   formulated as a GP or :ref:`t-process<DefTProcess>`
   conditional on hyperparameters, plus a set of hyperparameter values
   :math:`\theta^{(1)},\ldots,\theta^{(s)}`.
-  An initial value for the state variable :math:`w_0`.
-  The values of the forcing variables :math:`a_1,\ldots,a_T`.
-  The values of the simulator parameters :math:`\phi`.

Outputs
-------

-  Approximate marginal distributions for each of :math:`w_1,\ldots,w_T`.
   The distribution of each :math:`w_t` is approximated by a normal
   distribution with a specified mean and variance.

Procedure
---------

For a single choice of emulator hyperparameters :math:`\theta`, we
approximate the marginal distribution of :math:`w_t` by the normal
distribution :math:`N_r(\mu_t,V_t)`

We have

.. math::
   \mu_1 &=& m^*(w_0,a_1,\phi), \\
   V_1 &=& v^*\{(w_0,a_1,\phi),(w_0,a_1,\phi)\}.

The mean and variance are defined recursively:

.. math::
   \mu_{t+1} &=& \textrm{E}[m^*(w_t,a_{t+1},\phi)|f(D),\theta], \\
   V_{t+1} &=& \textrm{E}[v^*\{(w_t,a_{t+1},\phi),(w_t,a_{t+1},\phi)\}|f(D),\theta] +
   \textrm{Var}[m^*(w_t,a_{t+1},\phi)|f(D),\theta],

where the expectations and variances are taken with respect to :math:`w_{t}`,
where :math:`w_{t} \sim N_r(\mu_{t},V_{t})`

Explicit formulae for :math:`\mu_{t+1}` and :math:`V_{t+1}` can be
derived in the case of a linear mean and a separable Gaussian covariance
function. The procedure for calculating :math:`\mu_{t+1}` and
:math:`V_{t+1}` is described in
:ref:`ProcUpdateDynamicMeanAndVariance<ProcUpdateDynamicMeanAndVariance>`.
Otherwise, we can use simulation to estimate :math:`\mu_{t+1}` and
:math:`V_{t+1}`. A simulation procedure is given in
:ref:`ProcApproximateUpdateDynamicMeanandVariance<ProcApproximateUpdateDynamicMeanandVariance>`.

Integrating out the emulator hyperparameters
--------------------------------------------

Assuming we have :math:`s>1`, we can integrate out the emulator
hyperparameters to obtain the unconditional mean and variance of
:math:`w_{t}` using Monte Carlo estimation. In the following procedure, we
define :math:`N` to be the number of Monte Carlo iterations, and for
notational convenience, we suppose that :math:`N\le s`. For discussion of
the choice of :math:`N`, including the case :math:`N>s`, see the discussion
page on Monte Carlo estimation, sample sizes and emulator hyperparameter
sets (:ref:`DiscMonteCarlo<DiscMonteCarlo>`).

1. For :math:`i=1,2,\ldots,N` fix the hyperparameters at the value
   :math:`\theta^{(i)}`, and calculate the corresponding mean and variance
   of :math:`w_t`, which we denote by :math:`\mu_t^{(i)}` and :math:`V_t^{(i)}`.

2. Estimate :math:`\textrm{E}[w_t|f(D)]` by

   .. math::
      \hat{E}_t=\frac{1}{N}\sum_{i=1}^N \mu_t^{(i)}.

3. Estimate :math:`\textrm{Var}[w_t|f(D)]` by

   .. math::
      \frac{1}{N}\sum_{i=1}^N V_t^{(i)}+ \frac{1}{N-1}\sum_{i=1}^N
      \left(\mu_t^{(i)}-\hat{E}_t\right)^2.
