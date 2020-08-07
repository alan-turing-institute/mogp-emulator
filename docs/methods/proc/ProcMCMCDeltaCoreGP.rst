.. _ProcMCMCDeltaCoreGP:

Procedure: Sampling the posterior distribution of the correlation lengths
=========================================================================

Description and Background
--------------------------

This procedure shows how to draw samples from the posterior distribution
of the correlation lengths :math:`\pi^*_{\delta}(\delta)` and how
to use the drawn samples to make predictions using the emulator. This
procedure complements the procedure for building a Gaussian process
emulator for the core problem
(:ref:`ProcBuildCoreGP<ProcBuildCoreGP>`), and represents the formal
way of accounting for the uncertainty about the true value of the
correlation lengths :math:`\delta`.

Inputs
------

-  An emulator as defined in :ref:`ProcBuildCoreGP<ProcBuildCoreGP>`,
   using a linear mean and a weak prior.

Outputs
-------

-  A set of :math:`s` samples for the correlation lengths :math:`\delta`,
   denoted as :math:`\tilde{\delta}`.
-  A posterior mean :math:`\tilde{m}^*(\cdot)` and covariance
   function :math:`\tilde{u}^*(\cdot,\cdot)`, conditioned on the
   drawn samples :math:`\tilde{\delta}`.

Procedure
---------

A method for drawing samples from the posterior distribution of
:math:`\delta` is via the Metropolis-Hastings algorithm. Setting up
this algorithm requires an initial estimate of the correlation lengths,
and a proposal distribution. An initial estimate can be the value that
maximises the posterior distribution, as this is defined in the
discussion page for finding the posterior mode of correlation lengths
(:ref:`DiscPostModeDelta<DiscPostModeDelta>`). We call this estimate
:math:`\hat{\delta}`.

According to reference [1], a proposal distribution can be

.. math::
   \delta^{(i)} \sim {\cal N}(\delta^{(i-1)},V)

where :math:`\delta^{(i-1)}` is the sample drawn at the :math:`(i-1)`-th
step of the algorithm, and

.. math::
   \displaystyle V = -\frac{2.4}{\sqrt{p}}\left( \frac{\partial^2
   \pi^*_{\delta}(\hat{\delta})}{\partial \hat{\delta}}\right)^{-1}

The Hessian matrix :math:`\frac{\partial^2
\pi^*_{\delta}(\hat{\delta})}{\partial \hat{\delta}}` is the same as
:math:`\frac{\partial^2 g(\tau)}{\partial \tau_l \partial
\tau_k}`, which was defined in
:ref:`DiscPostModeDelta<DiscPostModeDelta>`, after substituting the
derivatives :math:`\partial A / \partial \tau` with the
derivatives :math:`\partial A / \partial \delta`. The latter are
given by

.. math::
   \displaystyle \left(\frac{\partial A} {\partial
   \delta_k}\right)_{i,j} &=
   A(i,j)\left[\frac{2(x_{k,i}-x_{k,j})^2}{\delta_k^3}\right] \\
   \displaystyle \left(\frac{\partial^2 A} {\partial^2
   \delta_k}\right)_{i,j} &= A(i,j) \frac{(x_{k,i}-x_{k,j})^2}{\delta_k^4}
   \left[ \frac{4(x_{k,i}-x_{k,j})^2}{\delta_k^2} - 6 \right]

and finally

.. math::
   \displaystyle \left(\frac{\partial^2 A} {\partial \delta_l\partial
   \delta_k}\right)_{i,j} = A(i,j)
   \left[\frac{2(x_{l,i}-x_{l,j})^2}{\delta_l^3}\right]
   \left[\frac{2(x_{k,i}-x_{k,j})^2}{\delta_k^3}\right]

Having defined the initial estimate of :math:`\delta` and the
proposal distribution :math:`{\cal N}(\delta^{(i-1)},V)` we can
set up the following Metropolis Hastings algorithm

#. Set :math:`\delta^{(1)}` equal to :math:`\hat{\delta}`
#. Add to :math:`\delta^{(i)}` a normal variate drawn from
   :math:`{\cal N}(0,V)` and call the result :math:`\delta'`
#. Calculate the ratio :math:`\displaystyle \alpha =
   \frac{\pi^*_{\delta}(\delta')}{\pi^*_{\delta}(\delta^{(i)})}`
#. Draw :math:`w` from a uniform distribution in [0,1]
#. if :math:`w <\alpha` set :math:`\delta^{(i+1)}` equal to :math:`\delta'`, else
   set it equal to :math:`\delta^{(i)}`
#. Repeat steps 2-5 :math:`s` times

Finally, if we define as :math:`m^{*(i)}(x)` and
:math:`u^{*(i)}(x,x')` the posterior mean and variance of the emulator using
sample :math:`\delta^{(i)}`, a total estimate of these two
quantities, taking into account all the :math:`s` samples of
:math:`\delta` drawn, is given by

.. math::
   \displaystyle \tilde{m}^*(x) = \frac{1}{s}\sum_{i=1}^s
   m^{*(i)}(x)

and

.. math::
   \displaystyle \tilde{u}^*(x,x') = \frac{1}{s}\sum_{i=1}^s
   u^{*(i)}(x,x') + \frac{1}{s}\sum_{i=1}^s \left[m^{*(i)}(x) -
   \tilde{m}^*(x)\right] \left[m^{*(i)}(x') - \tilde{m}^*(x')\right]

The procedure for predicting the simulator outputs using more than one
hyperparameter sets is described in greater detail in page
(:ref:`ProcPredictGP<ProcPredictGP>`).

References
----------

1. Gilks, W.R., Richardson, S. & Spiegelhalter, D.J. (1996). Markov
   Chain Monte Carlo in Practice. Chapman & Hall.
