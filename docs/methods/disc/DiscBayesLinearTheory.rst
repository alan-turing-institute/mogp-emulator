.. _DiscBayesLinearTheory:

Discussion: Theoretical aspects of Bayes linear
===============================================

Overview
--------

The :ref:`Bayes linear<DefBayesLinear>` approach is similar in spirit
to conventional :ref:`Bayes<DefBayesian>` analysis, but derives from
a simpler system for prior specification and analysis, and so offers a
practical methodology for analysing partially specified beliefs for
larger problems. The approach uses expectation rather than probability
as the primitive for quantifying uncertainty; see De Finetti (1974,
1975).

For a discussion on the differences between Bayes linear and full Bayes
approaches to emulation see
:ref:`AltGPorBLEmulator<AltGPorBLEmulator>`.

Notation
--------

Given the vector :math:`X` of random quantities, then we write
:math:`\textrm{E}[X]` as the expectation vector for :math:`X`, and
:math:`\textrm{Var}[X]` as the variance-covariance matrix for the elements
of :math:`X`. Given observations :math:`D`, we modify our prior expectations
and variances to obtain :ref:`adjusted<DefBLAdjust>` expectations and
variances for :math:`X` indicated by a subscript, giving
:math:`\textrm{E}_D[X]`, and :math:`\textrm{Var}_D[X]`.

Aspects of the relationship between the adjusted expectation
:math:`\textrm{E}_D[X]` and the conditional expectation :math:`E[X|D]` are
discussed later in this page.

Foundations of Bayes linear methods
-----------------------------------

Let :math:`C=(B,D)` be a vector of random quantities of interest. In the
Bayes linear approach, we make direct prior specifications for that
collection of means, variances and covariances which we are both willing
and able to assess. Namely, we specify :math:`\textrm{E}[C_i]`,
:math:`\textrm{Var}[C_i]`, :math:`\textrm{Cov}[C_i,C_j]` for all elements
:math:`C_i`, :math:`C_j` in the vector :math:`C`, :math:`i\neq j`. Suppose we
observe the values of the subset :math:`D` of :math:`C`. Then, following the
Bayesian paradigm, we modify our beliefs about the quantities :math:`B`
given the observed values of :math:`D`

Following Bayes linear methods, our modified beliefs are expressed by
the *adjusted* expectations, variances and covariance for :math:`B` given
:math:`D`. The adjusted expectation for element :math:`B_i` given :math:`D`,
written :math:`\textrm{E}_D[B_i]`, is the linear combination :math:`a_0 +
\textbf{a}^T D` minimising :math:`\textrm{E}[B_i - a_0 - \textbf{a}^T
D)^2]` over choices of :math:`\{a_0, \textbf{a}\}`. The adjusted
expectation vector is evaluated as

.. math::
   \textrm{E}_D[B] = \textrm{E}[B] + \textrm{Cov}[B,D]
   \textrm{Var}[D]^{-1} (D-\textrm{E}[D])

If the variance matrix :math:`\textrm{Var}[D]` is not invertible, then we
use an appropriate generalised inverse.

Similarly, the *adjusted variance matrix* for :math:`B` given :math:`D` is

.. math::
   \textrm{Var}_D[B] = \textrm{Var}[B] -
   \textrm{Cov}[B,D]\textrm{Var}[D]^{-1}\textrm{Cov}[D,B]

Stone (1963), and Hartigan (1969) are among the first to discuss the
role of such assessments in partial Bayes analysis. A detailed account
of Bayes linear methodology is given in Goldstein and Wooff (2007),
emphasising the interpretive and diagnostic cycle of subjectivist belief
analysis. The basic approach to statistical modelling within this
formalism is through second-order exchangeability.

Interpretations of adjusted expectation and variance
----------------------------------------------------

Viewing this approach from a full Bayesian perspective, the adjusted
expectation offers a tractable approximation to the conditional
expectation, and the adjusted variance provides a strict upper bound for
the expected posterior variance, over all possible prior specifications
which are consistent with the given second-order moment structure. In
certain special cases these approximations are exact, in particular if
the joint probability distribution of :math:`B`, :math:`D` is multivariate
normal then the adjusted and conditional expectations and variances are
identical.

In the special case where the vector :math:`D` is comprised of indicator
functions for the elements of a partition, ie each :math:`D_i` takes value
one or zero and precisely one element :math:`D_i` will equal one, then the
adjusted expectation is numerically equivalent to conditional
expectation. Consequently, adjusted expectation can be viewed as a
generalisation of de Finetti's approach to conditional expectation based
on 'called-off' quadratic penalties, where we now lift the restriction
that we may only condition on the indicator functions for a partition.

Geometrically, we may view each individual random quantity as a vector,
and construct the natural inner product space based on covariance. In
this construction, the adjusted expectation of a random quantity :math:`Y`,
by a further collection of random quantities :math:`D`, is the orthogonal
projection of :math:`Y` into the linear subspace spanned by the elements of
:math:`D` and the adjusted variance is the squared distance between :math:`Y`
and that subspace. This formalism extends naturally to handle infinite
collections of expectation statements, for example those associated with
a standard Bayesian analysis.

A more fundamental interpretation of the Bayes linear approach derives
from the temporal sure preference principle, which says, informally,
that if it is necessary that you will prefer a certain small random
penalty :math:`A` to :math:`C` at some given future time, then you should not
now have a strict preference for penalty :math:`C` over :math:`A`. A
consequence of this principle is that you must judge now that your
actual posterior expectation, :math:`\textrm{E}_T[B]`, at time :math:`T` when
you have observed :math:`D`, satisfies the relation :math:`\textrm{E}_T[B]=
\textrm{E}_D[B] + R`, where :math:`R` has, a priori, zero expectation and
is uncorrelated with :math:`D`. If :math:`D` represents a partition, then
:math:`E_D[B]` is equal to the conditional expectation given :math:`D`, and
:math:`R` has conditional expectation zero for each member of the
partition. In this view, the correspondence between actual belief
revisions and formal analysis based on partial prior specifications is
entirely derived through stochastic relationships of this type.

References
----------

-  De Finetti, B. (1974), Theory of Probability, vol. 1, Wiley.
-  De Finetti, B. (1975), Theory of Probability, vol. 2, Wiley.
-  Goldstein, M. and Wooff, D. A. (2007), Bayes Linear Statistics:
   Theory and Methods, Wiley.
-  Hartigan, J. A. (1969), “Linear Bayes methods,” *Journal of the Royal
   Statistical Society*, Series B, 31, 446–454.
-  Stone, M. (1963), “Robustness of non-ideal decision procedures,”
   *Journal of the American Statistical Association*, 58, 480–486.
