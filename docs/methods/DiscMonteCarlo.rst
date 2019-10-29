.. _DiscMonteCarlo:

Discussion: Monte Carlo estimation, sample sizes and emulator hyperparameter sets
=================================================================================

Various procedures in this toolkit use Monte Carlo sampling to estimate
quantities of interest. For example, suppose we are interested in some
nonlinear function :math::ref:`g(.)` of a `simulator<DefSimulator>`
output :math:`f(x) \`. As :math:`g\{f(x)\} \` is uncertain, we might choose to
estimate it by its mean :math:`\textrm{E}[g\{f(x)\}] \`, and estimate the
mean itself using Monte Carlo.

We obtain a random sample :math:`f^{(1)}(x),\ldots,f^{(N)}(x) \` from the
distribution of :math::ref:`f(x) \`, as described by the
`emulator<DefEmulator>` for :math:`f(.)`, evaluate :math:`t_1=g\{
f^{(1)}(x)\},\ldots,t_N=g\{f^{(N)}(x) \\} \`, and estimate
:math:`\textrm{E}[g\{f(x)\}] \` by

:math:`\hat{\textrm{E}}[g\{f(x)\}] =\frac{1}{N}\sum_{i=1}^N t_i. \`

How large should :math:`N` be?

We can calculate an approximate confidence interval to assess the
accuracy of our estimate. For example, an approximate 95% confidence
interval is given by

:math:`\left(\hat{\textrm{E}}[g\{f(x)\}]
-1.96\sqrt{\frac{\hat{\sigma}^2}{N}}, \\hat{\textrm{E}}[g\{f(x)\}]
+1.96\sqrt{\frac{\hat{\sigma}^2}{N}}\right),`

where :math:`\hat{\sigma}^2=
\\frac{1}{N-1}\sum_{i=1}^N\left(t_i-\hat{\textrm{E}}[g\{f(x)\}]
\\right)^2 \`.

Hence, we can get an indication of the effect of increasing :math:`N` on
the width of the confidence interval (if we ignore errors in the
estimate :math:`\hat{\sigma}^2`), and assess whether it is necessary to use
a larger Monte Carlo sample size.

Emulator hyperparameter sets
----------------------------

Now suppose we wish to allow for emulator hyperparameter uncertainty
when estimating :math:`g\{f(x)\} \`.

Throughout the toolkit, we represent a fully Bayesian emulator in two
parts as described in the discussion page on forms of GP-based emulators
(:ref:`DiscGPBasedEmulator<DiscGPBasedEmulator>`). The first part is
the posterior (i.e. conditioned on the :ref:`training
inputs<DefTrainingSample>` :math:`D \` and training outputs :math:`f(D)
\:ref:`) distribution of :math:`f(\cdot)` conditional on hyperparameters
:math:`\theta`, which may be a `Gaussian process<DefGP>` or a
:ref:`t-process<DefTProcess>`. The second part is the posterior
distribution of :math:`\theta`, represented by a set
:math:`\{\theta^{(1)},\ldots,\theta^{(s)}\} \` of emulator hyperparameter
values. The above procedure is then applied as follows.

Given an independent sample of hyperparameters
:math:`\theta^{(1)},\ldots,\theta^{(N)} \`, we sample :math:`f^{(i)}(x) \`
from the posterior distribution of :math:`f(x)` conditional on :math:`\theta`
(the first part of the emulator) setting :math:`\theta=\theta^{(i)}`, and
proceed as before. By sampling one :math:`f^{(i)}(x) \` for each
:math:`\theta^{(i)}`, we guarantee independence between :math:`t_1,\ldots,t_N`
which is required for the confidence interval formula given above. (If
MCMC has been used to obtain :math:`\theta^{(1)},\ldots,\theta^{(N)} \`
then the hyperparameter samples may not be independent. In this case, we
suggest checking the sample :math:`t_1,\ldots,t_N` for autocorrelation
before calculating the confidence interval).

We have seen above how to assess the sample size :math:`N` required for
adequate Monte Carlo computation, and for this purpose the size :math:`s`
of the set of emulator hyperparameter values may need increasing.
Usually, the computational cost of obtaining additional hyperparameter
values will be small relative to the cost of the Monte Carlo procedure
itself.

If, for some reason, it is not possible or practical to obtain
additional hyperparameter values, then we recommend cycling through the
sample :math:`\{\theta^{(1)},\ldots,\theta^{(s)}\} \` as we generate each
of :math:`t_1,\ldots,t_N`. Again, we can check the sample
:math:`t_1,\ldots,t_N` for autocorrelation before calculating the
confidence interval. If possible, it is worth exploring whether the
distribution of :math:`g\{f(x)\} \` (or whatever quantity is of interest)
is robust to the choice of :math:`\theta \`, for example, by estimating
:math:`\textrm{E}[g\{f(x)\}] \` for different fixed choices of :math:`\theta
\` from the sample :math:`\{\theta^{(1)},\ldots,\theta^{(s)}\} \`.

In some cases, we may have a single estimate of :math:`\theta`, perhaps
obtained by maximum likelihood. Monte Carlo (and alternative) methods
still allow us to usefully consider simulator uncertainty about
:math:`g\{f(x)\} \`, although we are now not allowing for hyperparameter
uncertainty. Again, if possible we recommend testing for robustness to
different choices of :math:`\theta \`.
