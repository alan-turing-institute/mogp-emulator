.. _ProcBuildMultiOutputGPSep:

Procedure: Complete the multivariate Gaussian process emulator with a separable covariance function
===================================================================================================

Description and Background
--------------------------

The first steps in building a multivariate Gaussian process emulator for
a simulator with :math::ref:`r` outputs are described in
`ProcBuildMultiOutputGP<ProcBuildMultiOutputGP>`. We assume here
that a linear mean function :math:`m(x) = h(x)^{\rm T}\beta` with a weak
prior :math:`\pi(\beta) \\propto 1` is used in that procedure, so the
result is a multivariate Gaussian process emulator that is conditional
on the choice of covariance function :math:`v(.,.)` and its associated
hyperparameters :math:`\omega`. It is necessary to choose one of the
structures for :math::ref:`v(.,.)` discussed in
`AltMultivariateCovarianceStructures<AltMultivariateCovarianceStructures>`.
The most simple option is a :ref:`separable<DefSeparable>` structure
which leads to a simpler multivariate emulator than that described in
:ref:`ProcBuildMultiOutputGP<ProcBuildMultiOutputGP>`. The procedure
here is for creating that simplified multivariate Gaussian process
emulator with a :ref:`separable<DefSeparable>` covariance function.

A separable covariance has the form

:math:`v(.,.) = \\Sigma c(\cdot,\cdot)\, , \`

where :math:`\Sigma` is a :math:` r \\times r` covariance matrix between
outputs and :math:`c(.,.)` is a correlation function between input points.
The hyperparameters for the separable covariance are
:math:`\omega=(\Sigma,\delta)`, where :math:`\delta` are the hyperparameters
for :math::ref:`c(.,.)`. The choice of prior for :math:`\Sigma` is discussed in
`AltMultivariateGPPriors<AltMultivariateGPPriors>`, but here we
assume here that :math:`\Sigma` has the weak prior :math:`\pi_\Sigma(\Sigma)
\\propto \|\Sigma|^{-\frac{r+1}{2}}`.

As discussed in
:ref:`AltMultivariateCovarianceStructures<AltMultivariateCovarianceStructures>`,
the assumption of separability imposes a restriction that all the
outputs have the same correlation function :math:`c(.,.)` across the input
space. We shall see in the following procedure that this leads to a
simple emulation methodology. The drawback is that the emulator may not
perform well if the outputs represent several different types of
physical quantity, since it assumes that all outputs have the same
smoothness properties. If that assumption is too restrictive, then a
nonseparable covariance function may be required. Some options for
nonseparable covariance function are described in
:ref:`AltMultivariateCovarianceStructures<AltMultivariateCovarianceStructures>`.

Inputs
------

-  Multivariate GP emulator with a linear mean function that is
   conditional on the choice of covariance function :math:`v(.,.)` and its
   associated hyperparameters :math:`\omega`, which is constructed
   according to the procedure in
   :ref:`ProcBuildMultiOutputGP<ProcBuildMultiOutputGP>`.

-  A GP prior input-space correlation function :math:`c(\cdot,\cdot)`
   depending on hyperparameters :math:`\delta`

-  A prior distribution :math:`\pi(\cdot)` for :math:`\delta`.

Outputs
-------

-  A GP-based emulator with a :ref:`multivariate
   t-process<DefMultivariateTProcess>` posterior conditional
   distribution with mean function :math:`{m^{*}(\cdot)} \`, covariance
   function :math:`v^{*}(\cdot,\cdot)` and degrees of freedom :math:`b^*`
   conditional on :math:`\delta`.

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

In addition to the notation defined in
:ref:`ProcBuildMultiOutputGP<ProcBuildMultiOutputGP>`, we define the
following arrays (following the conventions set out in the Toolkit's
notation page :ref:`MetaNotation<MetaNotation>`).

-  :math:`A=c(D,D)`, the :math:`n\times n` matrix of input-space correlations
   between all pairs of design points in :math:`D`;

-  :math:`t(x)=c(D,x)`, an :math:`n\times 1` vector function of :math:`x`.

-  :math:` R(x) = h(x)^{\rm T} - t(x)^{\rm T} A^{-1}H \`

A consequence of the separable structure for :math:`v(.,.)` is that the
:math:`rn\times rn` covariance matrix :math:`V=v(D,D)` has the Kronecker
product representation :math:`V=\Sigma \\otimes A`, and the :math:`rn\times r`
matrix function :math:`\strut u(x)=v(D,x)` has the Kronecker product
representation :math:`u(x)=\Sigma \\otimes t(x)`. As a result the
:math:`n\times r` matrix :math:`\widehat{\beta}` has the simpler form

:math:`\widehat{\beta}=\left( H^{\rm T} A^{-1} H\right)^{-1}H^{\rm T} A^{-1}
f(D)\, . \`

Then, conditional on :math:`\delta` and the training sample, the simulator
output vector :math::ref:`f(x)` is a `multivariate
t-process<DefMultivariateTProcess>` with :math:`b^*=n-q` degrees of
freedom, posterior mean function

:math:`m^*(x) = h(x)^T\widehat\beta + t(x)^{\rm T} A^{-1}
(f(D)-H\widehat\beta)`

and posterior covariance function

:math:`v^{*}(x,x^{\prime}) = \\widehat\Sigma\,\left\{c(x,x^{\prime})\, -\,
t(x)^{\rm T} A^{-1} t(x^{\prime})\, +\, R(x) \\left( H^{\rm T} A^{-1}
H\right)^{-1} R(x^{\prime})^{\rm T} \\right\}\, , \`

where

:math:` \\widehat\Sigma = (n-q)^{-1} (f(D)-H\widehat\beta)^{\rm T} A^{-1}
(f(D)-H\widehat\beta)\\ = (n-q)^{-1} f(D)^{\rm T}\left\{A^{-1} - A^{-1}
H\left( H^{\rm T} A^{-1} H\right)^{-1}H^{\rm T}A^{-1}\right\} f(D) \\, .
\`

This is the first part of the emulator as discussed in
:ref:`DiscGPBasedEmulator<DiscGPBasedEmulator>`. The emulator is
formally completed by a second part comprising the posterior
distribution of :math:`\delta`, which has density given by

:math:` \\pi_\delta^{*}(\delta) \\propto \\pi_\delta(\delta) \\times
\|\widehat\Sigma|^{-(n-q)/2}|A|^{-r/2}\| H^{\rm T} A^{-1} H|^{-r/2}\,.
\`

In order to compute the emulator predictions and other tasks, three
approaches can be considered.

#. Exact computations require a sample from the posterior distribution
   of :math:`\delta`. This can be obtained by MCMC; a suitable reference
   can be found below.
#. A common approximation is simply to fix :math:`\delta` at a single value
   estimated from the posterior distribution. The usual choice is the
   posterior mode, which can be found as the value of :math:`\delta` for
   which :math:`\pi^{*}_{\delta}(\delta)` is maximised. See the page on
   alternative estimators of correlation hyperparameters
   (:ref:`AltEstimateDelta<AltEstimateDelta>`).
#. An intermediate approach first approximates the posterior
   distribution by a multivariate lognormal distribution and then uses a
   sample from this distribution, as described in the procedure page
   :ref:`ProcApproxDeltaPosterior<ProcApproxDeltaPosterior>`.

Each of these approaches results in a set of values (or just a single
value in the case of the second approach) of :math:`\delta`, which allow
the emulator predictions and other required inferences to be computed.

Although it represents an approximation that ignores the uncertainty in
:math:`\delta`, approach 2 has been widely used. It has often been
suggested that, although uncertainty in these correlation
hyperparameters can be substantial, taking proper account of that
uncertainty through approach 1 does not lead to appreciable differences
in the resulting emulator. On the other hand, although this may be true
if a good single estimate for :math:`\delta` is used, this is not
necessarily easy to find, and the posterior mode may sometimes be a poor
choice. Approach 3 has not been used much, but can be recommended when
there is concern about using just a single :math:`\delta` estimate. It is
simpler than the full MCMC approach 1, but should capture the
uncertainty in :math:`\delta` well.

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

Details of the linear mean weak prior case can be found in:

Conti, S. and O'Hagan, A. (2009). Bayesian emulation of complex
multi-output and dynamic computer models. Journal of Statistical
Planning and Inference. `doi:
10.1016/j.jspi.2009.08.006 <http://dx.doi.org/10.1016/j.jspi.2009.08.006>`__

The multi-output emulator with the linear mean form is a special case of
the outer product emulator. The following reference gives formulae which
exploit separable structures in both the mean and covariance functions
to achieve computational efficiency that allows very large (output
dimension) simulators to be emulated.

J.C. Rougier (2008), Efficient Emulators for Multivariate Deterministic
Functions, Journal of Computational and Graphical Statistics, 17(4),
827-843.
`doi:10.1198/106186008X384032 <http://pubs.amstat.org/doi/abs/10.1198/106186008X384032>`__.
