.. _ProcBuildCoreGP:

Procedure: Build Gaussian process emulator for the core problem
===============================================================

Description and Background
--------------------------

The preparation for building a :ref:`Gaussian process<DefGP>` (GP)
:ref:`emulator<DefEmulator>` for the :ref:`core problem<DiscCore>`
involves defining the prior mean and covariance functions, identifying
prior distributions for :ref:`hyperparameters<DefHyperparameter>`,
creating a :ref:`design<DefDesign>` for the :ref:`training
sample<DefTrainingSample>`, then running the
:ref:`simulator<DefSimulator>` at the input configurations specified
in the design. All of this is described in the thread for the analysis
of the core model using Gaussian process methods
(:ref:`ThreadCoreGP<ThreadCoreGP>`). The procedure here is for taking
those various ingredients and creating the GP emulator.

Inputs
------

-  GP prior mean function :math:`m(\cdot)` depending on hyperparameters
   :math:`\beta`
-  GP prior correlation function :math:`c(\cdot,\cdot)` depending on
   hyperparameters :math:`\delta`
-  Prior distribution :math:`\pi(\cdot,\cdot,\cdot)` for
   :math:`\beta,\sigma^2` and :math:`\delta`, where :math:`\sigma^2` is the
   process variance hyperparameter
-  Design :math:`D` comprising points :math:`\{x_1,x_2,\ldots,x_n\}` in
   the input space
-  Output vector :math:`f(D)=(f(x_1),f(x_2),\ldots,f(x_n))^T`, where
   :math:`f(x_j)` is the simulator output from input point :math:`x_j`

Outputs
-------

A GP-based emulator in one of the forms presented in the discussion page
on GP emulator forms
(:ref:`DiscGPBasedEmulator<DiscGPBasedEmulator>`).

In the case of general prior mean and correlation functions and general
prior distribution:

-  A GP posterior conditional distribution with mean function
   :math:`m^*(\cdot)` and covariance function :math:`v^*(\cdot,\cdot)`
   conditional on :math:`\theta=\{\beta,\sigma^2,\delta\}`
-  A posterior representation for :math:`\theta`

In the case of linear mean function, general correlation function, weak
prior information on :math:`\beta,\sigma^2` and general prior distribution
for :math:`\delta`:

-  A :ref:`t process<DefTProcess>` posterior conditional distribution
   with mean function :math:`m^*(\cdot)`, covariance function
   :math:`v^*(\cdot,\cdot)` and degrees of freedom :math:`b^*`
   conditional on :math:`\delta`
-  A posterior representation for :math:`\delta`

As explained in :ref:`DiscGPBasedEmulator<DiscGPBasedEmulator>`, the
"posterior representation" for the hyperparameters is formally the
posterior distribution for those hyperparameters, but for computational
purposes this distribution is represented by a sample of hyperparameter
values. In either case, the outputs define the emulator and allow all
necessary computations for tasks such as prediction of the simulator
output, :ref:`uncertainty analysis<DefUncertaintyAnalysis>` or
:ref:`sensitivity analysis<DefSensitivityAnalysis>`.

Procedure
---------

General case
~~~~~~~~~~~~

Define the following arrays, according to the conventions set out in the
notation page (:ref:`MetaNotation<MetaNotation>`).

:math:`e=f(D)-m(D)`, an :math:`n\times 1` vector;

:math:`A=c(D,D)`, an :math:`n\times n` matrix;

:math:`t(x)=c(D,x)`, an :math:`n\times 1` vector function of :math:`x`.

Then, conditional on :math:`\theta` and the training sample, the simulator
output :math:`f(x)` is a GP with posterior mean function

.. math::
   m^*(x) = m(x) + t(x)^{\rm T} A^{-1} e

and posterior covariance function

.. math::
   v^*(x,x^\prime) = \sigma^2\{c(x,x^\prime) - t(x)^{\rm T} A^{-1}
   t(x^\prime) \}.

This is the first part of the emulator as discussed in
:ref:`DiscGPBasedEmulator<DiscGPBasedEmulator>`. The emulator is
completed by a second part formally comprising the posterior
distribution of :math:`\theta`, which has density given by

.. math::
   \pi^*(\beta,\sigma^2,\delta) \propto \pi(\beta,\sigma^2,\delta)
   \times (\sigma^2)^{-n/2}|A|^{-1/2} \times \exp\{-e^{\rm
   T}A^{-1}e/(2\sigma^2)\},

where the symbol :math:`\propto` denotes proportionality as usual in
Bayesian statistics. In order to compute the emulator predictions and
other tasks, the posterior representation of :math:`\theta` includes a
sample from this posterior distribution. The standard method for
obtaining this is Markov chain Monte Carlo (MCMC). For this general
case, the form of the posterior distribution depends very much on the
forms of prior mean and correlation functions and the prior
distribution, so no general advice can be given. The References section
below lists some useful texts on MCMC.

Linear mean and weak prior case
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose now that the mean function has the linear form :math:`m(x) =
h(x)^{\rm T}\beta`, where :math:`h(\cdot)` is a vector of :math:`q` known
:ref:`basis functions<DefBasisFunctions>` of the inputs and
:math:`\beta` is a :math:`q\times 1` column vector of hyperparameters. Suppose
also that the prior distribution has the form
:math:`\pi(\beta,\sigma^2,\delta) \propto \sigma^{-2}\pi_\delta(\delta)`,
i.e. that we have weak prior information on :math:`\beta` and :math:`\sigma^2`
and an arbitrary prior distribution :math:`\pi_\delta(\cdot)` for
:math:`\delta`.

Define :math:`A` and :math:`t(\cdot)` as in the previous case. In
addition, define the :math:`n\times q` matrix

.. math::
   H = [h(x_1),h(x_2),\ldots,h(x_n)]^{\rm T},

or in a more compact notation as :math:`H = h(D^{\rm T})^{\rm T}`, the vector

.. math::
   \widehat{\beta}=\left( H^{\rm T} A^{-1} H\right)^{-1}H^{\rm T} A^{-1}
   f(D),

and the scalar

.. math::
   \widehat\sigma^2 = (n-q-2)^{-1}f(D)^{\rm T}\left\{A^{-1} - A^{-1}
   H\left( H^{\rm T} A^{-1} H\right)^{-1}H^{\rm T}A^{-1}\right\} f(D),

which can also be written as

.. math::
   \widehat\sigma^2 = (n-q-2)^{-1}(f(D)-H\hat{\beta})^{\rm T} A^{-1}
   (f(D)-H\hat{\beta}).

Then, conditional on :math:`\delta` and the training sample, the simulator
output :math:`f(x)` is a t process with :math:`b^*=n-q` degrees of freedom,
posterior mean function

.. math::
   m^*(x) = h(x)^{\rm T}\widehat\beta + t(x)^{\rm T} A^{-1}
   (f(D)-H\widehat\beta)

and posterior covariance function

.. math::
   v^*(x,x^\prime) = \widehat\sigma^2\{c(x,x^\prime) - t(x)^{\rm T}
   A^{-1} t(x^\prime) + \left( h(x)^{\rm T} - t(x)^{\rm T} A^{-1}H
   \right) \left( H^{\rm T} A^{-1} H\right)^{-1} \left( h(x^\prime)^{\rm
   T} - t(x^\prime)^{\rm T} A^{-1}H \right)^{\rm T} \}.

This is the first part of the emulator as discussed in
:ref:`DiscGPBasedEmulator<DiscGPBasedEmulator>`. The emulator is
formally completed by a second part comprising the posterior
distribution of :math:`\delta`, which has density given by

.. math::
   \pi_\delta^*(\delta) \propto \pi_\delta(\delta) \times
   (\widehat\sigma^2)^{-(n-q)/2}|A|^{-1/2}| H^{\rm T} A^{-1}
   H|^{-1/2}.

In order to derive the sample representation of this posterior
distribution for the second part of the emulator, three approaches can
be considered.

#. A common approximation is simply to fix :math:`\delta` at a single value
   estimated from the posterior distribution. The usual choice is the
   posterior mode, which can be found as the value of :math:`\delta` for
   which :math:`\pi^*(\delta)` is maximised. The discussion page on finding
   the posterior mode of delta
   (:ref:`DiscPostModeDelta<DiscPostModeDelta>`), presents some
   details of this procedure. See also the alternatives page on
   estimators of correlation hyperparameters
   (:ref:`AltEstimateDelta<AltEstimateDelta>`) for a discussion of
   alternative estimators.
#. Another approach is to formally account for the uncertainty about the
   true value of :math:`\delta`, by sampling the posterior
   distribution of the correlation lengths and performing a Monte Carlo
   integration. This is described in the procedure page
   :ref:`ProcMCMCDeltaCoreGP<ProcMCMCDeltaCoreGP>`. A reference on
   MCMC algorithms can be found below.
#. An intermediate approach first approximates the posterior
   distribution by a multivariate lognormal distribution and then uses a
   sample from this distribution. See also the procedure on multivariate
   lognormal approximation for correlation hyperparameters
   (:ref:`ProcApproxDeltaPosterior<ProcApproxDeltaPosterior>`).

Each of these approaches results in a set of values (or just a single
value in the case of the first approach) of :math:`\delta`, which allow the
emulator predictions and other required inferences to be computed.

Although it represents an approximation that ignores the uncertainty in
:math:`\delta`, approach 1 has been widely used. It has often been
suggested that, although uncertainty in these correlation
hyperparameters can be substantial, taking proper account of that
uncertainty through approach 2 does not lead to appreciable differences
in the resulting emulator. On the other hand, although this may be true
if a good single estimate for :math:`\delta` is used, this is not
necessarily easy to find, and the posterior mode may sometimes be a poor
choice. Approach 3 has not been used much, but can be recommended when
there is concern about using just a single :math:`\delta` estimate. It is
simpler than the full MCMC approach 2, but should capture the
uncertainty in :math:`\delta` well.

Approaches 1 and 2 are both used in the
`GEM-SA <http://tonyohagan.co.uk/academic/GEM/>`_ software
(:ref:`disclaimer<MetaSoftwareDisclaimer>`).

Additional Comments
-------------------

Several computational issues can arise in implementing this procedure.
These are discussed in :ref:`DiscBuildCoreGP<DiscBuildCoreGP>`.

References
----------

Here are two leading textbooks on MCMC:

-  Gilks, W.R., Richardson, S. & Spiegelhalter, D.J. (1996). Markov
   Chain Monte Carlo in Practice. Chapman & Hall.
-  Gamerman, D. and Lopes, H. F. (2006). Markov Chain Monte Carlo:
   Stochastic Simulation for Bayesian Inference. CRC Press.

Although MCMC for the distribution of :math:`\delta` has been reported in a
number of articles, they have not given any details for how to do this,
assuming instead that the reader is familiar with MCMC techniques.
