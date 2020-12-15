.. _ProcBuildMultiOutputGP:

Procedure: Build multivariate Gaussian process emulator for the core problem
============================================================================

Description and Background
--------------------------

The preparation for building a :ref:`multivariate Gaussian
process<DefMultivariateGP>` (GP) :ref:`emulator<DefEmulator>`
for the :ref:`core problem<DiscCore>` involves defining the prior
mean and covariance functions, identifying prior distributions for
:ref:`hyperparameters<DefHyperparameter>`, creating a
:ref:`design<DefDesign>` for the :ref:`training
sample<DefTrainingSample>`, then running the
:ref:`simulator<DefSimulator>` at the input configurations specified
in the design. All of this is described in the variant thread for
analysis of a simulator with multiple outputs using Gaussian Process
methods
(:ref:`ThreadVariantMultipleOutputs<ThreadVariantMultipleOutputs>`).
The procedure here is for taking those various ingredients and creating
the GP emulator.

In the case of :math:`r` outputs, the simulator is a :math:`1\times r` row
vector function :math:`f(\cdot)`,and so its mean function :math:`m(\cdot)` is
also a :math:`1\times r` row vector function, while its covariance function
:math:`v(\cdot,\cdot)` is a :math:`r\times r` matrix function. The :math:`i, j`\th
element of :math:`v(\cdot,\cdot)` expresses the covariance between
:math:`f_i(\cdot)` and :math:`f_j(\cdot)`.

Inputs
------

-  GP prior mean function :math:`m(\cdot)` depending on hyperparameters
   :math:`\beta`
-  GP prior input-space covariance function :math:`v(\cdot,\cdot)`
   depending on hyperparameters :math:`\omega`
-  Prior distribution :math:`\pi(\cdot,\cdot)` for :math:`\beta` and
   :math:`\omega`.
-  Design :math:`D` comprising points :math:`\{x_1,x_2,\ldots,x_n\}` in the
   input space.
-  :math:`{n\times r}` output matrix :math:`f(D)`.

Outputs
-------

A GP-based emulator in one of the forms presented in the discussion page
:ref:`DiscGPBasedEmulator<DiscGPBasedEmulator>`.

In the case of general prior mean and correlation functions and general
prior distribution

-  A multivariate GP posterior conditional distribution with mean
   function :math:`m^{*}(\cdot)` and covariance function
   :math:`v^{*}(\cdot,\cdot)` conditional on :math:`\theta=\{\beta,\omega\}`.
-  A posterior representation for :math:`\theta`.

In the case of linear mean function, general covariance function, weak
prior information on :math:`\beta` and general prior distribution for
:math:`\omega` we have:

-  A :ref:`multivariate Gaussian process<DefMultivariateGP>`
   posterior conditional distribution with mean function
   :math:`{m^{*}(\cdot)}` and covariance function :math:`v^{*}(\cdot,\cdot)`
   conditional on :math:`\omega`.
-  A posterior representation for :math:`\omega`.

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

We define the following arrays (following the conventions set out in the
Toolkit's notation page (:ref:`MetaNotation<MetaNotation>`)).

-  :math:`e=f(D)-m(D)`, an :math:`n\times r` matrix;

-  :math:`V=v(D,D)`, the :math:`rn\times rn` covariance matrix composed of
   :math:`n\times n` blocks :math:`\{V_{ij}:i,j=1,...,r\}`, where the
   :math:`k,\ell`\th entry of :math:`V_{ij}` is the covariance between
   :math:`f_i(x_k)` and :math:`f_j(x_\ell)`;

-  :math:`\strut u(x)=v(D,x)`, the :math:`rn\times r` matrix function of
   :math:`x` composed of :math:`n\times 1` blocks
   :math:`\{u_{ij}(x):i,j=1,...,r\}`, where the :math:`k`th entry of
   :math:`u_{ij}(x)` is the covariance between :math:`f_i(x_k)` and
   :math:`f_j(x)`.

Then, conditional on :math:`\theta` and the training sample, the simulator
output vector :math:`f(x)` is a multivariate GP with posterior mean
function

.. math::
   \strut m^*(x) = m(x) + \mathrm{vec}(e)^{\rm T}V^{-1}u(x)

and posterior covariance function

.. math::
   \strut v^*(x,x^\prime) = v(x,x^\prime) - u(x)^{\rm T} V^{-1}
   u(x^\prime).

This is the first part of the emulator as discussed in
:ref:`DiscGPBasedEmulator<DiscGPBasedEmulator>`. The emulator is
completed by a second part formally comprising the posterior
distribution of :math:`\theta`, which has density given by

.. math::
   \pi^*(\beta,\omega) \propto \pi(\beta,\omega) \times
   |V|^{-1/2} \exp\left\{-\frac{1}{2}\mathrm{vec}(e)^{\rm
   T}V^{-1}\mathrm{vec}(e)\right\}

where the symbol :math:`\propto` denotes proportionality as usual in
Bayesian statistics. In order to compute the emulator predictions and
other tasks, the posterior representation of :math:`\theta` includes a
sample from this posterior distribution. The standard method for doing
this is Markov chain Monte Carlo (MCMC). For this general case, the form
of the posterior distribution depends very much on the forms of prior
mean and correlation functions and the prior distribution, so no general
advice can be given. The References section below lists some useful
texts on MCMC.

Linear mean and weak prior case
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose now that the mean function has the linear form :math:`m(x) =
h(x)^{\rm T}\beta`, where :math:`h(\cdot)` is a vector of :math:`q` known
:ref:`basis functions<DefBasisFunctions>` of the inputs and
:math:`\beta` is a :math:`q\times r` matrix of hyperparameters. Suppose also
that the prior distribution has the form :math:`\pi(\beta,\omega) \propto
\pi_\omega(\omega)`, i.e. that we have weak prior information on
:math:`{\beta}` and an arbitrary prior distribution :math:`\pi_\omega(\cdot)`
for :math:`\omega`.

Define :math:`V` and :math:`u(x)` as in the previous case. In addition, define
the :math:`n \times q` matrix

.. math::
   H = h(D)^{\rm T},

the :math:`q\times r` matrix :math:`\widehat{\beta}=` such that

.. math::
   \mathrm{vec}(\widehat{\beta})=\left( (I_k\otimes H^{\rm T})
   V^{-1} (I_k\otimes H)\right)^{-1}(I_k\otimes H^{\rm T}) V^{-1}
   \mathrm{vec}(f(D)),

and the :math:`r\times qr` matrix

.. math::
   R(x) = I_k\otimes h(x)^{\rm T} - u(x)^{\rm T}
   V^{-1}(I_k\otimes H).

Then, conditional on :math:`\omega` and the training sample, the simulator
output vector :math:`f(x)` is a :ref:`multivariate
GP<DefMultivariateGP>` with posterior mean function

.. math::
   m^*(x) = h(x)^T\widehat\beta + u(x)^{\rm T} V^{-1}
   \mathrm{vec}(f(D)-H\widehat\beta)

and posterior covariance function

.. math::
   v^{*}(x,x^{\prime}) = v(x,x^\prime) - u(x)^{\rm T} V^{-1}
   u(x^\prime) + R(x) \left( (I_k\otimes H^{\rm T}) V^{-1} (I_k\otimes
   H)\right)^{-1} R(x^{\prime})^{\rm T}.

This is the first part of the emulator as discussed in
:ref:`DiscGPBasedEmulator<DiscGPBasedEmulator>`. The emulator is
formally completed by a second part comprising the posterior
distribution of :math:`\omega`, which has density given by

.. math::
   \strut \\pi_\omega^{*}(\omega) \propto \pi_\omega(\omega) \times
   |V|^{-1/2}\| (I_k\otimes H^{\rm T}) V^{-1} (I_k\otimes H)|^{-1/2}
   \exp\left\{-\frac{1}{2}\mathrm{vec}(f(D)-H\widehat\beta)^{\rm
   T}V^{-1}\mathrm{vec}(f(D)-H\widehat\beta)\right\}.

In order to compute the emulator predictions and other tasks, the
posterior representation of :math:`\theta` includes a sample from this
posterior distribution. The standard method for doing this is Markov
chain Monte Carlo (MCMC). For this general case, the form of the
posterior distribution depends very much on the forms of prior mean and
correlation functions and the prior distribution, so no general advice
can be given. The References section below lists some useful texts on
MCMC.

Choice of covariance function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The procedures above are for a general multivariate covariance function
:math:`v(\cdot,\cdot)`. As such, the emulators are conditional on the choice of
covariance function :math:`v(\cdot,\cdot)` and its associated hyperparameters
:math:`\omega`. In order to use the emulator, a structure for :math:`v(\cdot,\cdot)`
must be chosen that ensures the covariance matrix :math:`v(D,D)` is
positive semi-definite for any design :math:`D`. The options for this
structure are found in the alternatives page
:ref:`AltMultivariateCovarianceStructures<AltMultivariateCovarianceStructures>`.

The simplest option is the :ref:`separable<DefSeparable>` structure.
In many cases the separable structure is adequate, and leads to several
simplifications in the above mathematics. The result is an easily built
and workable multi-output emulator. The aforementioned mathematical
simplifications, and the procedure for completing the separable
multi-output emulator, are in the procedure page
:ref:`ProcBuildMultiOutputGPSep<ProcBuildMultiOutputGPSep>`.

More complex, nonseparable structures are available and can provide
greater flexibility than the separable structure, but at the cost of
producing emulators that are harder to build. Options for nonseparable
covariance functions are discussed in
:ref:`AltMultivariateCovarianceStructures<AltMultivariateCovarianceStructures>`.

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
