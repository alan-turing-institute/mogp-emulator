.. _ProcBLVarianceLearning:

Procedure: Bayes linear method for learning about the emulator residual variance
================================================================================

Description and Background
--------------------------

This page assumes that the reader is familiar with the concepts of
:ref:`exchangeability<DefExchangeability>`, :ref:`second-order
exchangeability<DefSecondOrderExch>`, and the general
methodology of Bayes linear belief adjustment for collections of
exchangeable quantities as discussed in
:ref:`DiscAdjustExchBeliefs<DiscAdjustExchBeliefs>`.

In the study of computer models, learning about the mean behaviour of
the simulator can be a challenging task though there are many tools
available for such an analysis. The problem of learning about the
variance of a simulator, or the variance parameters of its emulator is
more difficult still. One approach is to apply the Bayes linear methods
for adjusting exchangeable beliefs. For example, suppose our emulator
has the form

.. math::
   f(x)=\sum_j\beta_jh_j(x)+e(x),

then if our trend component is known and the residuals can be treated as
second-order exchangeable then we can directly apply the methods of
:ref:`DiscAdjustExchBeliefs<DiscAdjustExchBeliefs>` to learn about
:math:`\text{Var}[e(x)]`, the variance of the residual process. The
constraint on this procedure is that we require the emulator residuals
to be (approximately) second-order exchangeable. In the context of
computer models, we can reasonably make this judgement in two
situations:

#. **The design points** :math:`D=(x_1,\dots,x_n)^T` **are such that they are
   sufficiently well-separated that they can be treated as
   uncorrelated.** This may occur due to simply having a small number of
   points, or having a large number of points in a high-dimensional
   input space. In either case, the relative sparsity of model
   evaluations means that we can express the covariance matrix of the
   emulator residuals as :math:`\text{Var}[e(D)]=\sigma^2 I_n` where
   :math:`I_n` is the :math:`n\times n` identity matrix and only :math:`\sigma^2`
   is uncertain. In this case, the residuals :math:`e(x)` have the same
   prior mean (0), the same prior variance (:math:`\sigma^2`) and every pair
   of residuals has the same covariance (0) therefore the residuals can
   be treated as second-order exchangeable.
#. **The form of the correlation function** :math:`c(x,x')` **and its
   parameters** :math:`\delta` **are known.** In this case, for any number or
   arrangement of design points we can express
   :math:`\text{Var}[e(D)]=\sigma^2 R` where :math:`R` is a known matrix of
   correlations, and :math:`\sigma^2` is the unknown variance of the
   residual process. Since the correlation between every pair of
   residuals is not constant, we do not immediately have second-order
   exchangeability. However, since the form of :math:`R` is known we can
   perform a linear transformation of the residuals which will preserve
   the constant mean and variance but will result in a correlation
   matrix of the form :math:`R=I_n` and so resulting in a collection of
   transformed second-order exchangeable residuals which can be used to
   learn about :math:`\sigma^2`.

The general process for the adjustment is identical for either
situation, however in the second case we must perform a preliminary
transformation step.

Inputs
------

The following inputs are required:

-  Output vector :math:`f(D)=(f(x_1), \dots, f(x_n))`, where :math:`f(x_i)`
   is the scalar simulator output corresponding to input vector :math:`x_i`
   in :math:`D`;
-  The form of the emulator trend basis functions :math:`h_j(\cdot)`
-  Design matrix :math:`D=(x_1,\dots,x_n)`.
-  Specification of prior beliefs for :math:`\omega_e=\sigma^2` and the
   fourth-order quantities :math:`\omega_\mathcal{M}` and
   :math:`\omega_\mathcal{R}` as defined below and in
   :ref:`DiscAdjustExchBeliefs<DiscAdjustExchBeliefs>`.

We also make the following requirements:

-  **Either**: The design points in :math:`D` are sufficiently
   well-separated that the corresponding residuals can be considered to
   be uncorrelated
-  **Or**: The form of the correlation function, :math:`c(x,x')`, and the
   values of its hyperparameters, :math:`\delta`, (and hence the
   correlations between every pair of residuals) are known.

Outputs
-------

-  Adjusted expectation and variance for the variance of the residual
   process

Procedure
---------

Overview
~~~~~~~~

Our goal is to learn about the variance of the residual process,
:math:`e(x)`, in the emulator of a given computer model. Typically, we
assume that our emulator has a mean function which takes linear form in
some appropriate basis functions of the inputs, and so we express our
emulator as

.. math::
   f(x)=\beta^Th(x)+e(x),

where :math:`\beta` is a vector of emulator trend coefficients, :math:`h(x)`
is a vector of the :ref:`basis functions<DefBasisFunctions>`
evaluated at input :math:`x`, and :math:`e(x)` is a stochastic residual
process. We consider that the coefficients
:math:`\beta=(\beta_1,\dots,\beta_q)` are unknown, and then work with the
derived residual quantities :math:`e_i=e(x_i)`. We assume that we consider
the residual process to be weakly stationary with mean zero a priori,
which gives

.. math::
   e_i &= f(x_i) - \beta_1 h_1(x_i) - \dots - \beta_q h_q(x_i) \\
   \text{E}[e_i] &= 0 \\
   \text{Var}[e_i] &= \sigma^2 = \omega_e

where we introduce :math:`\omega_e=\sigma^2` as the variance of :math:`e(x)`
for notational convenience and to mirror the notation of
:ref:`DiscAdjustExchBeliefs<DiscAdjustExchBeliefs>`.

Orthogonalisation
~~~~~~~~~~~~~~~~~

In the case where the emulator residuals are not uncorrelated, but can
be expressed in the form :math:`\text{Var}[e]=\sigma^2 R`, where :math:`R` is
a known :math:`n\times n` correlation matrix, we are required to make a
transformation in order to de-correlate the residuals in order to obtain
a collection of second-order exchangeable random quantities. To do this,
we adopt the standard approach in regression with correlated errors --
namely generalised least squares.

Let :math:`Q` be any matrix satisfying :math:`QQ^T=R`, and we can then
transform the emulator :math:`f(D)=X\beta+e` to the form

.. math::
   f'(D)=X'\beta + e',

where :math:`f'(D)=Q^{-1}f(D)`, :math:`X'=Q^{-1}Z`, and :math:`e'=Q^{-1}e`. An
example of a suitable matrix :math:`Q` would be if we find the
eigen-decomposition of :math:`R` such that :math:`R=A\Lambda A^T` then
:math:`Q^{-1}=\Lambda^{-\frac{1}{2}}A^T` would provide a suitable
transformation matrix. Under this transformation, we have that

.. math::
   \text{E}[e'] &= Q^{-1}\text{E}[e]=0 \\
   \text{Var}[e'] &= Q^{-1}\text{Var}[e]Q^{-T}=\omega_e I_n.

Note that he transformed residuals :math:`e'` have both the same mean and
variance as the un-transformed residuals :math:`e_i`, and in particular
note that :math:`\text{Var}[e_i] = \text{Var}[e'_i]=\sigma^2` which is the
quantity we seek to estimate. Further, the transformed residuals :math:`e'`
are second-order exchangeable as they have a common mean and variance,
and every pair has a common covariance.

Exchangeability Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to revise our beliefs about the population residual variance,
we judge that the residuals :math:`e_i` are second-order exchangeable. When
the residuals are well-separated and uncorrelated, this is immediately
true. In the case of the known correlations, then we make this statement
about the transformed residuals, :math:`e_i'`, and proceed through the
subsequent stages operating with the :math:`e_i'` instead of :math:`e_i`. For
simplicity, from this point on we only discuss :math:`e_i` and assume that
any necessary orthogonalisation has been made.

We begin with the uncorrelated second-order exchangeable sequence of
residuals :math:`e_i`. Suppose further that we judge that the :math:`e_i^2`
are also second-order exchangeable and so we write

.. math::
   v_i=e_i^2=\mathcal{M}(v)+\mathcal{R}_i(v)

where :math:`\text{E}[\mathcal{M}(v)]=\omega_e=\sigma^2`,
:math:`\text{Var}[\mathcal{M}(v)]=\omega_\mathcal{M}`, and that the
:math:`\mathcal{R}_i(v)` are SOE, uncorrelated and have zero mean and
variance :math:`\text{Var}[\mathcal{R}_i(v)]=\omega_\mathcal{R}`. We also
make the fourth-order uncorrelated assumptions mentioned in
:ref:`DiscAdjustExchBeliefs<DiscAdjustExchBeliefs>`.

In order to adjust our beliefs about the population residual variance,
we use the residual mean square :math:`\hat{\sigma}^2`,

.. math::
   \hat{\sigma}^2=\frac{1}{n-q}\hat{e}^T\hat{e},

where :math:`\hat{e}=f(D)-X\hat{\beta}=(I_n-H)f(D)`, where :math:`H` is the
idempotent matrix :math:`H=X(X^T X)^{-1}X^T`, :math:`X` is the model matrix
with :math:`i`-th row equal to :math:`(h_1(x_i),\dots,h_q(x_i))`, and
:math:`\hat{\beta}` are the least-squares estimates for :math:`\beta` given by
:math:`\hat{\beta}=(X^TX)^{-1}X^Tf(D)`. We could update our beliefs by
other quantities, though :math:`s^2` has a relatively simple representation
improving the tractability of subsequent calculations.

We can now express :math:`\hat{\sigma}^2` as

.. math::
   \hat{\sigma}^2 =\mathcal{M}(v)+T,

and

.. math::
   T=\frac{1}{n-q}\left[\sum_k (1-h_{kk})\mathcal{R}_k(v)-2\sum_{k <
   j} h_{kj} e_k e_j\right]

and it follows that we have the follow belief statements

.. math::
   \text{E}[\hat{\sigma}^2] &= \omega_e=\sigma^2, \\
   \text{Var}[\hat{\sigma}^2] &= \omega_\mathcal{M} + \omega_t, \\
   \text{Cov}[\hat{\sigma}^2,\mathcal{M}(v)] &= \omega_\mathcal{M}, \\
   \omega_T  &= \frac{1}{(n-q)^2}\left[ \omega_\mathcal{R} \sum_k
   (1-h_{kk})^2 -2(\omega_\mathcal{M}+\omega_e^2)\sum_k h_{kk}^2
   +2q(\omega_\mathcal{M}+\omega_e^2)\right],

which complete our belief specification for :math:`\hat{\sigma}^2` and
:math:`\mathcal{M}(v)`.

Variance Adjustment
~~~~~~~~~~~~~~~~~~~

Given the beliefs derived as above and the residual mean square
:math:`\hat{\sigma}^2` as calculated from the emulator runs and emulator
trend, we obtain the following expression for the adjusted mean and
variance for :math:`\mathcal{M}(v)`, the population residual variance:

.. math::
   \text{E}_{\hat{\sigma}^2}[\mathcal{M}(v)] &=
   \frac{\omega_\mathcal{M}\hat{\sigma}^2+\omega_T\omega_e}{\omega_\mathcal{M}+\omega_T} \\
   \text{Var}_{\hat{\sigma}^2}[\mathcal{M}(v)] &=
   \frac{\omega_\mathcal{M}\omega_t}{\omega_\mathcal{M}+\omega_t}

Comments and Discussion
-----------------------

When approaching problems based on exchangeable observations, we are
often also interested in learning about the population mean in addition
to the population variance. In terms of computer models, the primary
goal is to learn about the mean behaviour of our emulator residuals
rather than the emulator variance. To combine these approaches, we carry
out the analysis in two stages. For the first stage, we carry out
variance assessement as described above which gives us a revised
estimate for our residual variance, :math:`\sigma^2`. In the second stage,
we perform the standard Bayes linear analysis for the mean vector. This
involves following the standard methods of learning about the emulator
residual means as described in
:ref:`ProcBuildCoreBL<ProcBuildCoreBL>`, having replaced our prior
value for the residual variance with the adjusted estimate obtained from
the methods above. This procedure is called a **two-stage Bayes linear
analysis**, and is a simpler alternative to jointly learning about both
mean and variance which ignores uncertainty in the variance when
updating the mean vector.

References
----------

-  Goldstein, M. and Wooff, D. A. (2007), Bayes Linear Statistics:
   Theory and Methods, Wiley.
