.. _ProcMCMCDeltaCoreGP:

Procedure: Sampling the posterior distribution of the correlation lengths
=========================================================================

Description and Background
--------------------------

This procedure shows how to draw samples from the posterior distribution
of the correlation lengths :math:`\strut \\pi^*_{\delta}(\delta)` and how
to use the drawn samples to make predictions using the emulator. This
procedure complements the procedure for building a Gaussian process
emulator for the core problem
(:ref:`ProcBuildCoreGP<ProcBuildCoreGP>`), and represents the formal
way of accounting for the uncertainty about the true value of the
correlation lengths :math:`\strut \\delta`.

Inputs
------

-  An emulator as defined in :ref:`ProcBuildCoreGP<ProcBuildCoreGP>`,
   using a linear mean and a weak prior.

Outputs
-------

-  A set of :math:`\strut s` samples for the correlation lengths :math:`\strut
   \\delta`, denoted as :math:`\strut \\tilde{\delta}`.
-  A posterior mean :math:`\strut \\tilde{m}^*(\cdot)` and covariance
   function :math:`\strut \\tilde{u}^*(\cdot,\cdot)`, conditioned on the
   drawn samples :math:`\strut \\tilde{\delta}`.

Procedure
---------

A method for drawing samples from the posterior distribution of
:math:`\strut \\delta` is via the Metropolis-Hastings algorithm. Setting up
this algorithm requires an initial estimate of the correlation lengths,
and a proposal distribution. An initial estimate can be the value that
maximises the posterior distribution, as this is defined in the
discussion page for finding the posterior mode of correlation lengths
(:ref:`DiscPostModeDelta<DiscPostModeDelta>`). We call this estimate
:math:`\strut \\hat{\delta}`.

According to reference [1], a proposal distribution can be

:math:`\delta^{(i)} \\sim {\cal N}(\delta^{(i-1)},V)`

where :math:`\strut \\delta^{(i-1)}` is the sample drawn at the (i-1)-th
step of the algorithm, and

:math:`\displaystyle V = -\frac{2.4}{\sqrt{p}}\left( \\frac{\partial^2
\\pi^*_{\delta}(\hat{\delta})}{\partial \\hat{\delta}}\right)^{-1}`

The Hessian matrix :math:`\strut \\frac{\partial^2
\\pi^*_{\delta}(\hat{\delta})}{\partial \\hat{\delta}}` is the same as
:math:`\strut \\frac{\partial^2 g(\tau)}{\partial \\tau_l \\partial
\\tau_k}:ref:`, which was defined in
`DiscPostModeDelta<DiscPostModeDelta>`, after substituting the
derivatives :math:`\strut \\partial A / \\partial \\tau` with the
derivatives :math:`\strut \\partial A / \\partial \\delta`. The latter are
given by

:math:`\displaystyle \\left(\frac{\partial A} {\partial
\\delta_k}\right)_{i,j} =
A(i,j)\left[\frac{2(x_{k,i}-x_{k,j})^2}{\delta_k^3}\right] \`

:math:`\displaystyle \\left(\frac{\partial^2 A} {\partial^2
\\delta_k}\right)_{i,j} = A(i,j) \\frac{(x_{k,i}-x_{k,j})^2}{\delta_k^4}
\\left[ \\frac{4(x_{k,i}-x_{k,j})^2}{\delta_k^2} - 6 \\right]`

and finally

:math:`\displaystyle \\left(\frac{\partial^2 A} {\partial \\delta_l\partial
\\delta_k}\right)_{i,j} = A(i,j)
\\left[\frac{2(x_{l,i}-x_{l,j})^2}{\delta_l^3}\right]
\\left[\frac{2(x_{k,i}-x_{k,j})^2}{\delta_k^3}\right]`

Having defined the initial estimate of :math:`\strut \\delta` and the
proposal distribution :math:` \\strut {\cal N}(\delta^{(i-1)},V)` we can
set up the following Metropolis Hastings algorithm

#. Set :math:`\strut \\delta^{(1)}` equal to :math:`\strut \\hat{\delta}`
#. Add to :math:` \\strut \\delta^{(i)}` a normal variate drawn from
   :math:`\strut {\cal N}(0,V)` and call the result :math:`\strut \\delta'`
#. Calculate the ratio :math:`\displaystyle \\alpha =
   \\frac{\pi^*_{\delta}(\delta')}{\pi^*_{\delta}(\delta^{(i)})}`
#. Draw :math:`\strut w` from a uniform distribution in [0,1]
#. if :math:`w <\alpha` set :math:`\delta^{(i+1)}` equal to :math:`\delta'`, else
   set it equal to :math:`\delta^{(i)}`
#. Repeat steps 2-5 :math:`\strut s` times

Finally, if we define as :math:`\strut m^{*(i)}(x)` and :math:`\strut
u^{*(i)}(x,x')` the posterior mean and variance of the emulator using
sample :math:`\strut \\delta^{(i)}`, a total estimate of these two
quantities, taking into account all the :math:`\strut s` samples of
:math:`\strut \\delta \` drawn, is given by

:math:`\displaystyle \\tilde{m}^*(x) = \\frac{1}{s}\sum_{i=1}^s
m^{*(i)}(x)`

and

:math:`\displaystyle \\tilde{u}^*(x,x') = \\frac{1}{s}\sum_{i=1}^s
u^{*(i)}(x,x') + \\frac{1}{s}\sum_{i=1}^s \\left[m^{*(i)}(x) -
\\tilde{m}^*(x)\right] \\left[m^{*(i)}(x') - \\tilde{m}^*(x')\right]`

The procedure for predicting the simulator outputs using more than one
hyperparameter sets is described in greater detail in page
(:ref:`ProcPredictGP<ProcPredictGP>`).

References
----------

#. Gilks, W.R., Richardson, S. & Spiegelhalter, D.J. (1996). Markov
   Chain Monte Carlo in Practice. Chapman & Hall.
