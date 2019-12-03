.. _DiscVarianceBasedSATheory:

Discussion: Theory of variance-based sensitivity analysis
=========================================================

Description and background
--------------------------

The methods of variance-based :ref:`sensitivity
analysis<DefSensitivityAnalysis>` are set out in the discussion
page :ref:`DiscVarianceBasedSA<DiscVarianceBasedSA>`. Some technical
results and details are considered here.

Notation and terminology may all be found in
:ref:`DiscVarianceBasedSA<DiscVarianceBasedSA>`, although we often
repeat definitions here for clarity.

Discussion
----------

Equivalent interpretations of :math:`V_J`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In :ref:`DiscVarianceBasedSA<DiscVarianceBasedSA>`, the sensitivity
variance :math:`V_J` for inputs :math:`x_J` is defined as
:math:`\mathrm{Var}[M_J(X_J)]`, where :math:`M_J(x_J)` is the mean effect of
:math:`x_J`, defined as :math:`\mathrm{E}[f(X)\,|\,x_j]`. Now the following is
a general result in Statistics: let :math:`Y` and :math:`Z` be
two random variables, then

.. math::
   \mathrm{Var}[Y] = \mathrm{E}[\mathrm{Var}[Y\,|\,Z]] +
   \mathrm{Var}[\mathrm{E}[Y\,|\,Z]].

In words, the variance of :math:`Y` can be expressed as a sum of two
parts, the first is the expectation of the conditional variance of
:math:`Y` given :math:`Z`, and the second is the variance of the
conditional expectation of :math:`Y` given :math:`Z`.

We use this formula by equating :math:`Y` to :math:`f(X)` and :math:`Z`
to :math:`X_J`. From the above definitions, the second term on the
right hand side of the formula is seen to be :math:`V_J`. The other terms
easily reduce to other measures that are defined in
:ref:`DiscVarianceBasedSA<DiscVarianceBasedSA>`. The left hand side
is :math:`\mathrm{Var}[f(X)]` , which is the overall output variance
:math:`V`. The first term on the right hand side is the expectation
of :math:`\mathrm{Var}[f(X)\,|\,X_J]`, which is :math:`w(X_J)`. We therefore
have the result

.. math::
   V = W_J + V_J.

This provides the second interpretation of :math:`V_J` in
:ref:`DiscVarianceBasedSA<DiscVarianceBasedSA>`. It is the amount by
which uncertainty is expected to be reduced if we were to learn
:math:`x_J`, i.e. :math:`V-W_J`.

The variance partition theorem for independent inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A very useful result in variance-based SA theory is equation (4) in
:ref:`DiscVarianceBasedSA<DiscVarianceBasedSA>`, and concerns
interaction variances when the inputs :math:`x_j` (for :math:`j=1,2,\ldots,p`)
in :math:`x` are all mutually independent. In
:ref:`DiscVarianceBasedSA<DiscVarianceBasedSA>`, equation (3) can be
rewritten as

.. math::
   M_J(x_J) = \sum_{J'\subseteq J}I_{J'}(x_{J'})\,.\qquad (A)

In this formula, the sum is over all subsets :math:`J'` of
:math:`J`, including the empty set and :math:`J` itself. The
term :math:`I_{J'}(x_{J'})` is in general called the interaction effect of
:math:`x_{J'}`, except that when :math:`J'` contains just a single
element :math:`J'=\{j\}` then it is referred to as the main effect of
:math:`x_j`, and when :math:`J'` is the empty set we define it to be
the uncertainty mean :math:`M=\mathrm{E}[f(X)]`.

**Example**: Suppose that :math:`J=\{1,2,3\}`, then there are eight
possible subsets :math:`J'` and equation (A) becomes

.. math::
   \begin{array}{rl}M_{\{1,2,3\}}(x_{\{1,2,3\}}) = &
   I_{\{1,2,3\}}(x_{\{1,2,3\}})\\ & \quad+ I_{\{1,2\}}(x_{\{1,2\}}) +
   I_{\{1,3\}}(x_{\{1,3\}}) + I_{\{2,3\}}(x_{\{2,3\}}) \\ & \quad +
   I_{\{1\}}(x_{\{1\}}) + I_{\{2\}}(x_{\{2\}}) + I_{\{3\}}(x_{\{3\}})
   + M. \end{array}

The first term in this example is the three-input interaction; on the
next line are the three two-input interactions; on the third line are
the three main effects and finally the overall mean.

Now consider :math:`V_J`, which is the variance of :math:`M_J(X_J)`. Taking
the variance of the right hand side of (A), a standard result in
Statistics is that the variance of a sum of random variables is the sum
of the variances of those random variables plus the sum of the
covariances between all pairs. Now it can be shown that when the inputs
are independent all those covariances are zero. Since the variances of
the interaction terms in (A) are the interaction variances :math:`V^I_J`
this yields equation (4) in
:ref:`DiscVarianceBasedSA<DiscVarianceBasedSA>`.

In order to prove that result, it remains to show that all covariances
are zero when inputs are independent. First note that because inputs are
independent the covariance between :math:`I_J(X_J)` and :math:`I_{J'}(X_{J'})`
will be zero if :math:`J` and :math:`J'` have no elements in
common. To illustrate the proof when there are elements in common,
consider the correlation between the main effect

.. math::
   I_{\{j\}}(X_{\{j\}}) = M_{\{j\}}(X_{\{j\}}) - M

and the two-input interaction

.. math::
   I_{\{j,j'\}}(X_{\{j,j'\}}) = M_{\{j,j'\}}(X_{\{j,j'\}}) -
   M_{\{j\}}(X_{\{j\}}) - M_{\{j'\}}(X_{\{j'\}}) + M.

First notice that :math:`\mathrm{E}[I_{\{j\}}(X_{\{j\}})] = M-M=0` and
:math:`\mathrm{E}[I_{\{j,j'\}}(X_{\{j,j'\}})]=M-M-M+M=0`.

In fact, the expectation of every interaction term :math:`I_J(X_J)` is zero
when :math:`J` is not the empty set. However, this expectation is
taken with respect to all of :math:`X_J`; it is also true that if we take
the expectation with respect to just one :math:`X_j`, for any :math:`j`
in :math:`J` the result is also zero *provided* that the inputs are
independent. For instance, in the case of
:math:`I_{\{j,j'\}}(X_{\{j,j'\}})` consider the expectation with respect to
:math:`X_j` only. If :math:`X_j` and :math:`X_{j'}` are independent, we obtain
:math:`M_{\{j'\}}(X_{\{j'\}}) - M - M_{\{j'\}}(X_{\{j'\}}) + M=0`.

Now we can use these facts to find the covariance. First, because the
expectations are zero the covariance equals the expectation of the
product :math:`I_{\{j\}}(X_{\{j\}})\times I_{\{j,j'\}}(X_{\{j,j'\}})`.
This is the expectation with respect to both :math:`X_j` and :math:`X_{j'}`.
Because of independence, we can take the expectation in two stages,
first with respect to :math:`X_{j'}` then with respect to :math:`X_j`.
However, the first step produces zero because the expectation of
:math:`I_{\{j,j'\}}(X_{\{j,j'\}})` with respect to either of the inputs is
zero. Hence the covariance is zero.

The same argument works for any covariance. The covariance will be the
expectation of the product, and that will be zero when we first take
expectation with respect to any input that is in one but not both of
:math:`X_J` and :math:`X_{J'}`.

Non-independent inputs
~~~~~~~~~~~~~~~~~~~~~~

When the inputs are not independent, the above theorem does not hold. As
a result, some convenient simplifications also fail. For simplicity,
suppose we have just two inputs, :math:`X_1` and :math:`X_2`. With
independence, we have

.. math::
   V = V_{\{1,2\}} = V_{\{1\}} + V_{\{2\}} + I_{\{1,2\}}

If independence does not hold, then :math:`V_{\{1\}}`, :math:`V_{\{2\}}` and
:math:`I_{\{1,2\}}` are still defined as before and are positive
quantities, but their sum will not generally be :math:`V`.
Furthermore, in the case of independence, the total sensitivity index
:math:`T_{\{1\}}` of :math:`X_1` will be larger than its sensitivity index
:math:`S_{\{1\}}`, but this is also not necessarily true when inputs are
not independent.

Consider the case where the simulator has the form :math:`f(x_1,x_2) = x_1 +
2x_2`, and suppose that :math:`X_1` and :math:`X_2` both have zero means and
unit variances but have correlation :math:`\rho`. Now we find after some
simple algebra that :math:`V = 2(1+\rho)`, :math:`V_{\{1\}} = V_{\{2\}} =
(1+\rho)^2` and :math:`I_{\{1,2\}} = 2\rho^2(1+\rho)`. The last of these
is perhaps surprising. When the simulator is strictly a sum of two
terms, one a function of :math:`x_1` and the other a function of :math:`x_2`,
we would think of this as a case where there is no interaction, and
indeed if the inputs were independent then :math:`I_{\{1,2\}}` would be
zero, but this is not true when they are correlated.

For this example we have sensitivity indices

.. math::
   S_{\{1\}} = S_{\{2\}} = (1+\rho)/2\,, \qquad T_{\{1\}} = T_{\{2\}} =
   (1-\rho)/2,

and clearly :math:`T_{\{1\}} < S_{\{1\}}` in this instance if the
correlation is positive. 'Total sensitivity' can be a misleading term.

When the inputs are not independent it is not generally helpful to look
at interactions, and the sensitivity and total sensitivity indices must
be interpreted carefully.

Sequential decomposition
~~~~~~~~~~~~~~~~~~~~~~~~

The following way of decomposing the total uncertainty variance
:math:`V` applies whether the inputs are independent or not, and is
sometimes convenient computationally.

For exposition, suppose that we have just four uncertain inputs, :math:`X =
(X_1, X_2, X_3, X_4)`. Also, to clarify the formulae, we add subscripts
to the expectation and variance operators to denote which distribution
they are applied to.

.. math::
   \begin{array}{rl} V = {\rm Var}_X[y(X)] =& {\rm Var}_{X_1}[{\rm
   E}_{X_2,X_3,X_4|X_1}[y(X)]] + {\rm E}_{X_1}[{\rm
   Var}_{X_2,X_3,X_4|X_1}[y(X)]]\\\ =& {\rm Var}_{X_1}[{\rm
   E}_{X_2,X_3,X_4|X_1}[y(X)]] + {\rm E}_{X_1}[{\rm Var}_{X_2|X_1}[{\rm
   E}_{X_3,X_4|X_1,X_2}[y(X)]]] \\ & \qquad + {\rm E}_{X_1,X_2}[{\rm
   Var}_{X_3,X_4|X_1,X_2}[y(X)]] \\ =& {\rm Var}_{X_1}[{\rm
   E}_{X_2,X_3,X_4|X_1}[y(X)]] + {\rm E}_{X_1}[{\rm Var}_{X_2|X_1}[{\rm
   E}_{X_3,X_4|X_1,X_2}[y(X)]]] \\ & \qquad + {\rm E}_{X_1,X_2}[{\rm
   Var}_{X_3|X_1,X_2}[{\rm E}_{X_4|X_1,X_2,X_3}[y(X)]]] + {\rm
   E}_{X_1,X_2,X_3}[{\rm Var}_{X_4|X_1,X_2,X_3}[y(X)]] \end{array}

The formula can obviously be continued for more inputs, and we can
replace individual inputs with groups of inputs.

If inputs are independent, then the conditioning in the subscripts can
be removed. In this case, note that the first term in the decomposition
is :math:`V_{\{1\}}`, the second term, however, is
:math:`V_{\{2\}}+I_{\{1,2\}}`. Each successive term includes interactions
with the inputs coming earlier in the sequence. Finally, the last term
is :math:`T_{\{4\}}`.

If inputs are not independent, then this is a useful decomposition that
does not rely on separately identifying interactions (which we have seen
are not helpful in this case). However, it is also clear that the
decomposition depends on the order of the inputs, and we can obviously
define sequential decompositions in other sequences. This is analgous to
the analysis of variance in conventional statistics theory when factors
are not orthogonal.

Regression variances
~~~~~~~~~~~~~~~~~~~~

In the page discussing various forms of SA
(:ref:`DiscWhyProbabilisticSA<DiscWhyProbabilisticSA>`), one of those
forms is regression SA. In this approach, a linear regression function
is fitted to the simulator output :math:`f(x)` and sensitivity measures are
calculated from the resulting fitted regression coefficients.

Consider such a regression fit in which the regressor variables are
denoted by a vector :math:`g(x)` of functions of the inputs. The fitted
approximation to the simulator is :math:`\hat f(x) = \hat\gamma^{\rm T}
g(x)`, where :math:`\hat\gamma` is the corresponding vector of fitted
coefficients. This plays the same role as the mean effect :math:`M(x)`, and
from the perspective of variance-based SA, we can define a corresponding
sensitivity variance

.. math::
   V_g = {\rm Var}(\hat\gamma^{\rm T} g(X)).

This will always be less than the overall variance :math:`V` but
will get closer to it the more accurately the fitted regression is able
to approximate the true mean effect. Similarly, if :math:`g(x)` is a
function only of :math:`X_J`, then :math:`V_g < V_J` and the difference
between the two is smaller the more closely the regression function can
approximate :math:`M_J(x)`.

This provides one nice use for regression variances. Let :math:`g(x)^{\rm T}
= (1,x_j)`, so that the regression fit is a simple straight line in
:math:`x_j`. We call the resulting regression variance the linear
sensitivity variance of :math:`x_j`, and the difference between this and
:math:`V_{\{j\}}` is an indicator of how nonlinear the mean effect of
:math:`x_j` is.

Additional comments
-------------------

Formal definitions of regression variances and theory for computing them
from emulators is given in

Oakley, J. E. and O'Hagan, A. (2004). Probabilistic sensitivity analysis
of complex models: a Bayesian approach. *Journal of the Royal Statistical
Society B* 66, 751-769.
