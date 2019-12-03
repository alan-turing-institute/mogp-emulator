.. _ProcBuildWithDerivsGP:

Procedure: Build Gaussian process emulator with derivative information
======================================================================

Description and Background
--------------------------

The preparation for building a :ref:`Gaussian process<DefGP>` (GP)
:ref:`emulator<DefEmulator>` with derivative information involves
defining the prior mean and covariance functions, identifying prior
distributions for :ref:`hyperparameters<DefHyperparameter>`, creating
a :ref:`design<DefDesign>` for the :ref:`training
sample<DefTrainingSample>`, then running the
:ref:`adjoint<DefAdjoint>`, or :ref:`simulator<DefSimulator>` and
a method to obtain derivatives, at the input configurations specified in
the design. All of this is described in the variant thread on emulators
with derivative information
(:ref:`ThreadVariantWithDerivatives<ThreadVariantWithDerivatives>`).
The procedure here is for taking those various ingredients and creating
the GP emulator.

Additional notation for this page
---------------------------------

Including derivative information requires further notation than is
specified in the Toolkit notation page
(:ref:`MetaNotation<MetaNotation>`). We declare this notation here
but it is only applicable to the derivatives thread variant pages.

-  The tilde symbol (:math:`\tilde{}`) placed over a letter denotes
   derivative information and function output combined.
-  We introduce an extra argument to denote a derivative. We define
   :math:`\tilde{f}(x,d)` to be the derivative of :math:`f(x)` with respect to
   input :math:`d` and so :math:`d \in\{0,1,\ldots,p\}`. When
   :math:`d=0` we have :math:`\tilde{f}(x,0)=f(x)`. For simplicity,
   when :math:`d=0` we adopt the shorter notation so we use
   :math:`f(x)` rather than :math:`\tilde{f}(x,0)`.
-  An input is denoted by a superscript on :math:`x`, while a
   subscript on :math:`x` refers to the point in the input space.
   For example, :math:`x_i^{(k)}` refers to input :math:`k` at
   point :math:`i`.

Inputs
------

-  GP prior mean function :math:`m(\cdot)`, differentiable and depending on
   hyperparameters :math:`\beta`
-  GP prior correlation function :math:`c(\cdot,\cdot)`, twice
   differentiable and depending on hyperparameters :math:`\delta`
-  Prior distribution :math:`\pi(\cdot,\cdot,\cdot)` for
   :math:`\beta,\sigma^2` and :math:`\delta` where :math:`\sigma^2`
   is the process variance hyperparameter
-  We require a design. In the core thread
   :ref:`ThreadCoreGP<ThreadCoreGP>` the design, :math:`D`, is an
   ordered set of points :math:`D=\{x_1,x_2,\ldots,x_n\}`, where each
   :math:`x` is a location in the input space. Here, we need a
   design which in addition to specifying the location of the inputs,
   also determines at which points we require function output and at
   which points we require first derivatives. We arrange this
   information in the design :math:`\tilde{D} = \{(x_k,d_k)\}`, where :math:`k
   = \{1,\ldots,\tilde{n}\}` and :math:`d_k \in \{0,1,\ldots,p\}`. We
   have :math:`x_k` which refers to the location in the design and
   :math:`d_k` determines whether at point :math:`x_k` we
   require function output or a first derivative w.r.t one of the
   inputs. Each :math:`x_k` is not distinct as we may have a
   derivative and the function output at point :math:`x_k` or we may
   require a derivative w.r.t several inputs at point :math:`x_k`.
-  Output vector is :math:`\tilde{f}(\tilde{D})` of length :math:`\tilde{n}`.

Outputs
-------

A GP-based emulator in one of the forms discussed in
:ref:`DiscGPBasedEmulator<DiscGPBasedEmulator>`.

In the case of general prior mean and correlation functions and general
prior distribution:

-  A GP posterior conditional distribution with mean function
   :math:`m^*(\cdot)` and covariance function :math:`v^*(\cdot,\cdot)`
   conditional on :math:`\theta=\{\beta,\sigma^2,\delta\}`. If we want to
   emulate the derivatives rather than the function output see
   ProcBuildGPEmulateDerivs.
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

We define the following arrays (following the conventions set out in
:ref:`MetaNotation<MetaNotation>` where possible).

:math:`\tilde{e}=\tilde{f}(\tilde{D})-\tilde{m}(\tilde{D})`, an
:math:`\tilde{n}\times 1` vector, where :math:`\tilde{m}(x,0)=m(x)`, and
:math:`\tilde{m}(x,d) = \frac{\partial}{\partial x^{(d)}}m(x)` if
:math:`d\ne0`.

:math:`\tilde{A}=\tilde{c}(\tilde{D},\tilde{D}),` an :math:`\tilde{n}\times
\tilde{n}` matrix, where the exact form of :math:`\tilde{c}(\cdot,\cdot)` depends
on where derivatives are included. The general expression for this is:
:math:`\tilde{c}(.,.) = {\rm
Corr}\{\tilde{f}(x_i,d_i),\tilde{f}(x_j,d_j)\}` and we can break it
down into three cases:

-  Case 1 is for when :math:`\strut d_i=d_j=0` and as such represents the
   covariance between 2 points. This is the same as in
   :ref:`ThreadCoreGP<ThreadCoreGP>` and is given by:

   .. math::
      {\rm Corr}\{\tilde{f}(x_i,0),\tilde{f}(x_j,0)\} = c(x_i,x_j).

-  Case 2 is for when :math:`d_i\ne 0` and :math:`d_j=0` and as such
   represents the covariance between a derivative and a point. This is
   obtained by differentiating :math:`c(\cdot,\cdot)` w.r.t input :math:`d_i`:

   .. math::
      {\rm Corr}\{\tilde{f}(x_i,d_i),\tilde{f}(x_j,0)\} =
      \frac{\partial c(x_i,x_j)}{\partial x_i^{(d_i)}}, {\rm for}\; d_i\ne
      0.

-  Case 3 is for when :math:`d_i\ne 0` and :math:`d_j\ne 0` and as such
   represents the covariance between two derivatives. This is obtained
   by differentiating :math:`c(\cdot,\cdot)` twice: once w.r.t input
   :math:`d_i` and once w.r.t input :math:`d_j`:

   .. math::
      {\rm Corr}\{\tilde{f}(x_i,d_j),\tilde{f}(x_j,d_j)\} = \frac{\partial^2
      c(x_i,x_j)}{\partial x_i^{(d_i)} \partial x_j^{(d_j)}}, {\rm for}\;
      d_i,d_j\ne0\.

   -  Case 3a. If :math:`d_i,d_j\ne 0` and :math:`d_i=d_j` we have a special
      version of Case 3 which gives:

      .. math::
         {\rm Corr}\{\tilde{f}(x_i,d_i),\tilde{f}(x_j,d_i)\} = \frac{\partial^2
         c(x_i,x_j)}{\partial x_i^{(d_i)},x_j^{(d_i)}}, {\rm for}\;
         d_i\ne0.

:math:`\tilde{t}(x)=\tilde{c}\{\tilde{D},(x,0)\}`, an :math:`\tilde{n}\times
1` vector function of :math:`x`. We have :math:`d=0` as here we
want to emulate function output. To emulate derivatives, :math:`d\ne0` and
this is covered in the generic thread on emulating derivatives
(:ref:`ThreadGenericEmulateDerivatives<ThreadGenericEmulateDerivatives>`).

Then, conditional on :math:`\theta` and the training sample, the
output vector :math:`\tilde{f}(x,0)=f(x)` is a multivariate GP with
posterior mean function

.. math::
   m^*(x) = m(x) + \tilde{t}(x)^{\rm T} \tilde{A}^{-1} \tilde{e}

and posterior covariance function

.. math::
   v^*(x_i,x_j) = \sigma^2 \{c(x_i,x_j)-\tilde{t}(x_i)^{\rm T}
   \tilde{A}^{-1} \tilde{t}(x_j) \}.

This is the first part of the emulator as discussed in
:ref:`DiscGPBasedEmulator<DiscGPBasedEmulator>`. The emulator is
completed by a second part formally comprising the posterior
distribution of :math:`\theta`, which has density given by

.. math::
   \pi^*(\beta,\sigma^2,\delta) \propto \pi(\beta,\sigma^2,\delta)
   \times (\sigma^2)^{-\tilde{n}/2}|\tilde{A}|^{-1/2} \times
   \exp\{-\tilde{e}^{\rm T}\tilde{A}^{-1}\tilde{e}/(2\sigma^2)\}.

For the output vector :math:`\tilde{f}(x,d)` with :math:`d\ne0` see the
procedure page on building an emulator of derivatives
(:ref:`ProcBuildEmulateDerivsGP<ProcBuildEmulateDerivsGP>`).

Linear mean and weak prior case
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose now that the mean function has the linear form :math:`m(x) =
h(x)^{\rm T}\beta`, where :math:`h(\cdot)` is a vector of :math:`q` known
:ref:`basis functions<DefBasisFunctions>` of the inputs and
:math:`\beta` is a :math:`q\times 1` column vector of hyperparameters. When
:math:`d\ne0` we therefore have :math:`\tilde{m}(x,d) = \tilde{h}(x,d)^{\rm
T}\beta = \frac{\partial}{\partial x^{(d)}}h(x)^{\rm T}\beta`. Suppose
also that the prior distribution has the form
:math:`\pi(\beta,\sigma^2,\delta) \propto \sigma^{-2}\pi_\delta(\delta)`,
i.e. that we have weak prior information on :math:`\beta` and
:math:`\sigma^2` and an arbitrary prior distribution
:math:`\pi_\delta(\cdot)` for :math:`\delta`.

Define :math:`\tilde{A}` and :math:`\tilde{t}(x)` as in the previous
case. In addition, define the :math:`\tilde{n} \times q` matrix

.. math::
   \tilde{H}=[\tilde{h}(x_1,d_1),\ldots,\tilde{h}(x_{\tilde{n}},d_{\tilde{n}})]^{\rm
   T},

the vector

.. math::
   \widehat{\beta}=\left( \tilde{H}^{\rm T} \tilde{A}^{-1}
   \tilde{H}\right)^{-1}\tilde{H}^{\rm T} \tilde{A}^{-1}
   \tilde{f}(\tilde{D})

and the scalar

.. math::
   \widehat\sigma^2 = (\tilde{n}-q-2)^{-1}\tilde{f}(\tilde{D})^{\rm
   T}\left\{\tilde{A}^{-1} - \tilde{A}^{-1} \tilde{H}\left(
   \tilde{H}^{\rm T} \tilde{A}^{-1} \tilde{H}\right)^{-1}\tilde{H}^{\rm
   T}\tilde{A}^{-1}\right\} \tilde{f}(\tilde{D}).

Then, conditional on :math:`\delta` and the training sample, the
output vector :math:`\tilde{f}(x,0)=f(x)` is a t process with
:math:`b^*=\tilde{n}-q` degrees of freedom, posterior mean function

.. math::
   m^*(x) = h(x)^{\rm T}\widehat\beta + \tilde{t}(x)^{\rm T}
   \tilde{A}^{-1} (\tilde{f}(\tilde{D})-\tilde{H}\widehat\beta)

and posterior covariance function

.. math::
   v^*(x_i,x_j) = \widehat\sigma^2\{c(x_i,x_j) -
   \tilde{t}(x_i)^{\rm T} \tilde{A}^{-1} \tilde{t}(x_j) + \left(
   h(x_i)^{\rm T} - \tilde{t}(x_i)^{\rm T} \tilde{A}^{-1}\tilde{H}
   \right) \left( \tilde{H}^{\rm T} \\tilde{A}^{-1}
   \tilde{H}\right)^{-1} \left( h(x_j)^{\rm T} - \tilde{t}(x_j)^{\rm T}
   \tilde{A}^{-1}\tilde{H} \right)^{\rm T} \}.

This is the first part of the emulator as discussed in
:ref:`DiscGPBasedEmulator<DiscGPBasedEmulator>`. The emulator is
formally completed by a second part comprising the posterior
distribution of :math:`\delta`, which has density given by

.. math::
   \pi_\delta^*(\delta) \propto \pi_\delta(\delta) \times
   (\widehat\sigma^2)^{-(\tilde{n}-q)/2}|\tilde{A}|^{-1/2}|
   \tilde{H}^{\rm T} \tilde{A}^{-1} \tilde{H}|^{-1/2}.

In order to derive the sample representation of this posterior
distribution for the second part of the emulator, three approaches can
be considered.

#. Exact computations require a sample from the posterior distribution
   of :math:`\delta`. This can be obtained by MCMC; a suitable
   reference can be found below.
#. A common approximation is simply to fix :math:`\delta` at a
   single value estimated from the posterior distribution. The usual
   choice is the posterior mode, which can be found as the value of
   :math:`\delta` for which :math:`\pi^*(\delta)` is maximised. See
   the alternatives page on estimators of correlation hyperparameters
   (:ref:`AltEstimateDelta<AltEstimateDelta>`).
#. An intermediate approach first approximates the posterior
   distribution by a multivariate lognormal distribution and then uses a
   sample from this distribution, as described in the procedure page
   :ref:`ProcApproxDeltaPosterior<ProcApproxDeltaPosterior>`.

Each of these approaches results in a set of values (or just a single
value in the case of the second approach) of :math:`\delta`, which
allow the emulator predictions and other required inferences to be
computed.

Although it represents an approximation that ignores the uncertainty in
:math:`\delta`, approach 2 has been widely used. It has often been
suggested that, although uncertainty in these correlation
hyperparameters can be substantial, taking proper account of that
uncertainty through approach 1 does not lead to appreciable differences
in the resulting emulator. On the other hand, although this may be true
if a good single estimate for :math:`\delta` is used, this is not
necessarily easy to find, and the posterior mode may sometimes be a poor
choice. Approach 3 has not been used much, but can be recommended when
there is concern about using just a single :math:`\delta` estimate.
It is simpler than the full MCMC approach 1, but should capture the
uncertainty in :math:`\delta` well.

References
----------

Morris, M. D., Mitchell, T. J. and Ylvisaker, D. (1993). Bayesian design
and analysis of computer experiments: Use of derivatives in surface
prediction. *Technometrics*, 35, 243-255.
