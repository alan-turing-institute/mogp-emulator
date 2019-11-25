.. _ProcUABL:

Procedure: Uncertainty Analysis for a Bayes linear emulator
===========================================================

Description and Background
--------------------------

One of the tasks that is required by users of
:ref:`simulators<DefSimulator>` is :ref:`uncertainty
analysis<DefUncertaintyAnalysis>` (UA), which studies the
uncertainty in model outputs that is induced by uncertainty in the
inputs. Although relatively simple in concept, UA is both important and
demanding. It is important because it is the primary way to quantify the
uncertainty in predictions made by simulators. It is demanding because
in principle it requires us to evaulate the output at all possible
values of the uncertain inputs. The :ref:`MUCM<DefMUCM>` approach of
first building an :ref:`emulator<DefEmulator>` is a powerful way of
making UA feasible for complex and computer-intensive simulators.

This procedure considers the evaluation of the expectaion and variance
of the computer model when the input at which it is evaluated is
uncertain. The expressions for these quantities when the input :math:`x`
takes the unknown value :math:`x_0` are:

.. math::
   \mu = \text{E}[f(x_0)] = \text{E}^*[ \mu(x_0) ]

and

.. math::
   \Sigma = \text{Var}[f(x_0)] = \text{Var}^*[ \mu(x_0) ] +
   \text{E}^*[ \Sigma(x_0) ]

where expectations and variances marked with a \* are over the unknown
input :math:`x_0`, and where we use the shorthand expressions
:math:`\mu(x)=\text{E}_F[f(x)]` and :math:`\Sigma(x)=\text{Var}_F[f(x)]`.

Inputs
------

-  An emulator
-  The prior emulator beliefs for the emulator in the form of
   :math:`\text{Var}[\beta]`, :math:`\text{Var}[w(x)]`
-  The training sample design :math:`X`
-  Second-order beliefs about :math:`x_0`

Outputs
-------

-  The expectation and variance of :math:`f(x)` when :math:`x` is the unknown
   :math:`x_0`

Procedure
---------

Definitions
~~~~~~~~~~~

Define the following quantities used in the procedure of predicting
simulator output at a known input
(:ref:`ProcBLPredict<ProcBLPredict>`):

-  :math:`\hat{\beta}=\text{E}_F[\beta]`
-  :math:`B=\text{Var}[\beta]`
-  :math:`\sigma^2=\text{Var}[w(x)]`
-  :math:`V=\text{Var}[f(X)]`
-  :math:`e=f(X)-\textrm{E}[f(X)]`
-  :math:`B_F =\text{Var}_F[\beta]=B-BHV^{-1}H^TB^T`

where :math:`f(x)`, :math:`\beta` and :math:`w(x)` are as defined in the thread for
Bayes linear emulation of the core model
(:ref:`ThreadCoreBL<ThreadCoreBL>`).

Using these definitions, we can write the general adjusted emulator
expectation and variance at a **known** input :math:`x` (as given in
:ref:`ProcBLPredict<ProcBLPredict>`) in the form:

.. math::
   \mu(x) &=& \hat{\beta}^T h(x) + c(x) V^{-1} e \\
   \Sigma(x) &=& h(x)^T B_F h(x) + \sigma^2 - c(x)^T V^{-1} c(x) -
   h(x)^T BHV^{-1} c(x) - c(x)^T V^{-1} H^T B h(x)

where the vector :math:`h(x)` and the matrix :math:`H` are as defined in
:ref:`ProcBLPredict<ProcBLPredict>`, :math:`c(x)` is the :math:`n\times 1`
vector such that :math:`c(x)^T=\text{Cov}[w(x),w(X)]`, and :math:`B_F` is the
adjusted variance of the :math:`\beta`.

To obtain the expectation, :math:`\mu`, and variance, :math:`\Sigma`, for the
simulator at the unknown input :math:`x_0` we take expectations and
variances of these quantities over :math:`x` as described below.

Calculating :math:`\mu`
~~~~~~~~~~~~~~~~~~~~~~~~

To calculate :math:`\mu`, the expectation of the simulator at the unknown
input :math:`x_0`, we calculate the following expectation:

.. math::
   \mu=\text{E}[\mu(x_0)]=\hat{\beta}^T\text{E}[h_0]+\text{E}[c_0]^TV^{-1}d,

where we define :math:`h_0=h(x_0)` and :math:`c_0^T=\text{Cov}[w(x_0),w(X)]`.

Specification of beliefs for :math:`h_0` and :math:`c_o` is discussed at the
end of this page.

Calculating :math:`\Sigma`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:`\Sigma` is defined to be the sum of two components
:math:`\text{Var}[\mu(x_0)]` and :math:`\text{E}^*[ \Sigma(x_0) ]`. Using
:math:`h_0` and :math:`c_0` as defined above, we can write these expressions
as:

.. math::
   \textrm{Var}[\mu(x_0)] &=& \hat{\beta}^T\textrm{Var}[h_0]
   \hat{\beta}+e^TV^{-1}\textrm{Var}[c_0]^TV^{-1}e +
   2\hat{\beta}^T\textrm{Cov}[h_0,c_0] V^{-1}e \\
   \text{E}[\Sigma(x_0)] &=& \sigma^2 + \text{E}[h_0]^TB_F\text{E}[h_0] -
   \text{E}[c_0]^TV^{-1}\text{E}[c_0] - 2 \text{E}[h_0]^TB H
   V^{-1}\text{E}[c_0] \\
   & & + \text{tr}\left\{ \text{Var}[h_0]B_F - \text{Var}[c_0]V^{-1}
   -2\text{Cov}[h_0,c_0]V^{-1}H^TB\right\}

Beliefs about :math:`g_0` and :math:`c_0`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can see from the expressions given above, that in order to calculate
:math:`\mu` and :math:`\sigma`, we require statements on the expectations,
variances, and covariances for the collection :math:`(h_0,c_0)`. In the
Bayes linear framework, it will be straightforward to obtain
expectations, variances, and covariances for :math:`x_0` however since
:math:`h_0` and :math:`c_0` are complex functions of :math:`x_0` it can be
difficult to use our beliefs about :math:`x_0` to directly obtain beliefs
about :math:`h_0` or :math:`c_0`.

In general, we rely on the following strategies:

-  Monomial :math:`h(\cdot)` -- When the trend basis functions take the
   form of simple monomials in :math:`x_0`, then the expectation, and
   (co)variance for :math:`h_0` can be expressed in terms of higher-order
   moments of :math:`x_0` and so can be found directly. These higher order
   moments could be specified directly, or found via lower order moments
   using appropriate assumptions. In some cases, where our basis
   functions :math:`h(\cdot)` are not monomials but more complex functions,
   e.g. :math:`\text{sin}(x)`, these more complex functions may have a
   particular physical interpretation or relevance to the model under
   study. In these cases, it can be effective to consider the
   transformed inputs themselves and thus :math:`h(\cdot)` becomes a
   monomial in the transformed space.
-  Exploit probability distributions -- We construct a range of
   probability distributions for :math:`x_0` which are consistent with our
   second-order beliefs and our general sources of knowledge about
   likely values of :math:`x_0`. We then compute the appropriate integrals
   over our prior for :math:`x_0` to obtain the corresponding second-order
   moments either analytically or via simulation. When the correlation
   function is Gaussian, then we can obtain results analytically for
   certain choices of prior distribution of :math:`x_0` -- the procedure
   page on uncertainty analysis using a GP emulator
   (:ref:`ProcUAGP<ProcUAGP>`) addresses this material in detail.
