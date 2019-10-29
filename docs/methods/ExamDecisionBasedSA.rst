.. _ExamDecisionBasedSA:

Example: Illustrations of decision-based sensitivity analysis
=============================================================

Description and background
--------------------------

:ref:`Decision-based<DefDecisionBasedSA>` sensitivity analysis (SA)
is a form of probabilistic SA; see the page on varieties of SA
(:ref:`DiscWhyProbabilisticSA<DiscWhyProbabilisticSA>`). We present
here some examples of the formulation of decision-based sensitivity
analyses. Relevant notation and definitions are given in the
decision-based SA page
(:ref:`DiscDecisionBasedSA<DiscDecisionBasedSA>`), but we repeat some
key definitions here for convenience.

In probabilistic SA, we regard the :ref:`simulator<DefSimulator>`
inputs :math:`\strut X` as uncertain with probability distribution
:math:`\omega(x)`. In decision-based SA, we further require a decision set
:math:`\strut\cal{D}` whose elements are the possible decisions, and a loss
function :math:`L(d,x)` expressing the consequences of taking decision
:math:`d\in\cal{D}` when the true inputs are :math:`\strut X=x`. The optimal
decision is the one having smallest expected loss

:math:`\bar{L}(d) = {\rm E}[L(d,X)]\,,`

where the expectation is taken with respect to the distribution of
:math:`\strut X`. Therefore if the optimal decision is :math:`\strut M`,

:math:`\bar{L}(M) = \\min_d\{\bar{L}(d)\}\,.`

If we were able to learn the true value of a subset :math:`x_J` of the
inputs, then the optimal decision would minimise

:math:`\bar{L}_J(d,x_J) = {\rm E}[L(d,X)\,|\,X_J = x_J]\,.`

and would depend on :math:`x_J`. If we denote this decision by
:math:`M_J(x_J)`, then the decision-based sensitivity measure for :math:`X_J`
is the expected value of information

:math:`V_J = \\bar{L}(M) - {\rm E}[\bar{L}_J(M_J(X_J),X_J)]\,.`.

Our examples illustrate some decision problems that can be formulated
through :math:`\strut\cal{D}` and :math:`L(d,x)`, and explore the nature of
:math:`V_J` in each case.

Example 1: Estimation with quadratic loss
-----------------------------------------

First suppose that the decision problem is to estimate the true
simulator output :math:`f(x)`. Hence the decision set :math:`\strut\cal{D}`
consists of all possible values of the simulator output :math:`f(x)`. Our
loss function should be of the form that there is no loss if the
estimate (decision) exactly equals the true :math:`f(X)`, and should be an
increasing function of the estimation error :math:`f(x)-d`. One such
function is the quadratic loss function

:math:`L(d,x) = \\{f(x) - d\}^2\,.`

If we now apply the above theory, the optimal estimate :math:`\strut d`
will minimise

:math:`\begin{array}{rl} \\bar{L}(d) &= {\rm E}[f(X)^2 -2d\,f(X) + d^2] =
{\rm E}[f(X)^2] - 2d\,{\rm E}[f(X)] + d^2\\\ &= {\rm Var}[f(X)] +
\\{{\rm E}[f(X)] - d\}^2 \\,, \\end{array}`

after simple manipulation using the fact that, for any random variable
:math:`\strut Z`, :math:`{\rm Var}[Z] = {\rm E}[Z^2] - \\{{\rm E}[Z]\}^2`. We
can now see that this is minimised by setting :math:`d = {\rm E}[f(X)]`. We
can also see that the expected loss of this optimal decision is
:math:`\bar{L}[M]={\rm Var}[f(X)]`. In this decision problem, then, we find
that :math::ref:`\strut M` has the same form as defined for `variance-based
SA<DefVarianceBasedSA>` in the page
:ref:`DiscVarianceBasedSA<DiscVarianceBasedSA>`. By an identical
argument we find that if we learn the value of :math:`x_J` the optimal
decision becomes :math::ref:`M_J(x_J) = {\rm E}[f(X)\,|\,x_J]`, which is also as
defined in `DiscVarianceBasedSA<DiscVarianceBasedSA>`. Finally,
the sensitivity measure becomes

:math:`V_J = {\rm Var}[f(X)] - {\rm E}[{\rm Var}[f(X)\,|\,X_J]]\,,`

which is once again the sensitivity measure defined in
:ref:`DiscVarianceBasedSA<DiscVarianceBasedSA>`.

This example demonstrates that we can view variance-based SA as a
special case of decision-based SA, in which the decision is to estimate
:math:`f(x)` and the loss function is quadratic loss (or squared error).

Before finishing this example it should be noted first that
:ref:`DiscVarianceBasedSA<DiscVarianceBasedSA>` presents a rationale
for variance-based SA that is persuasive when there is no underlying
decision problem. But second, if there genuinely is a decision to
estimate :math:`f(x)` the loss function should be chosen appropriately to
the context of that decision. It might be quadratic loss, but depending
on what units we choose to measure that loss in we might have a
multiplier :math:`\strut c` such that :math:`L(d,x) = c\{f(x)-d\}^2`. The
optimal decisions would be unchanged but the sensitivity measure (the
value of information) would now be multiplied by :math:`\strut c`.

The same applies to any decision problem. Making a linear transformation
of the loss function does not change the optimal decision, but the value
of information is multiplied by the scale factor of the transformation.

Example 2: Optimisation
-----------------------

A common decision problem in the context of using a simulator is
identify the values of certain inputs in order to maximise or minimise
the output. We will suppose here that the objective is to minimise
:math:`f(x)`, so that the output itself serves as the loss function. This
is straightforward if all of the inputs are under our control, but the
problem becomes more interesting when only some of the inputs can be
controlled to optimise the output, while the remainder are uncertain. We
therefore write the simulator as :math:`f(d,y)`, where :math:`\strut d`
denotes the control inputs and :math:`\strut y` the remaining inputs. The
latter are uncertain with distribution :math:`\omega(y)`.

So the decision problem is characterised by a decision set
:math:`\strut\cal{D}` comprising all possible values of the control inputs
:math:`\strut d`, together with a loss function

:math:`L(d,x) = f(x) = f(d,y)\,.`

To illustrate the calculations in this case, let the simulator have the
form

:math:`f(d,y) = y_1\{y_2 + (y_1-d)^2\}\,,`

where :math:`\strut d` is a scalar and :math:`y=(y_1,y_2)`. With :math:`\strut y`
uncertain, we compute the expected loss. Simple algebra gives

:math:`\bar{L}(d) = {\rm E}[Y_1Y_2 + Y_1^3 - 2dY_1^2 +d^2Y_1] = {\rm
E}[Y_1](d-M)^2 + \\bar{L}[M]\,,`

where

:math:`M = {\rm E}[Y_1^2]/{\rm E}[Y_1]`

is the optimal decision and

:math:`\bar{L}[M] = {\rm E}[Y_1Y_2] +{\rm E}[Y_1^3] - \\{{\rm
E}[Y_1^2]\}^2/{\rm E}[Y_1]`

is the minimal expected loss, i.e. the expected value of the simulator
output at :math:`\strut d=M`. If now we were to learn the value of
:math:`y_1`, then

:math:`\bar{L}_1(y_1) = {\rm E}[L(d,X)\,|\,y_1] = y_1\{{\rm E}[Y_2\,|\,y_1]
+ (y_1-d)^2\}\,,`

and the optimal decision would be :math:`d=y_1` with minimal (expected)
loss :math:`y_1{\rm E}[Y_2|y_1]`. Since the expectation of this with
respect to the uncertainty in :math:`y_1` is just :math:`{\rm E}[Y_1Y_2]`, we
then find that the sensitivity measure for :math:`y_1`, i.e. the expected
value of learning the true value of :math:`y_1`, is

:math:`V_{\{1\}} = {\rm E}[Y_1^3] - \\{{\rm E}[Y_1^2]\}^2/{\rm E}[Y_1]\,.`

It is straightforward to see that the value of learning both :math:`y_1`
and :math:`y_2` is the same, :math:`V_{\{1,2\}} = V_{\{1\}}`, because the
optimal decision is still :math:`d=y_1`. So if we could learn :math:`y_1`
there would be no additional value in learning :math:`y_2`.

It remains to consider the sensitivity measure for learning :math:`y_2`. We
now find that the optimal decision is

:math:`M_2(y_2) = {\rm E}[Y_1^2\,|\,y_2]/{\rm E}[Y_1\,|\,y_2]`

and the expected value of learning the true value of :math:`y_2` is

:math:`V_{\{2\}} = {\rm E}\left [\{{\rm E}[Y_1^2\,|\,Y_2]\}^2/{\rm
E}[Y_1\,|\,Y_2]\right ] - \\{{\rm E}[Y_1^2]\}^2/{\rm E}[Y_1]\,.`

If the two uncertain inputs are independent, then we find that this
reduces to zero. Otherwise, learning the value of :math:`y_2` gives us some
information about :math:`y_1`, which in turn has value.

**Numerical illustration**. Suppose that :math:`Y_2` takes the value 0 or 1
with equal probabilities and that the distribution of :math:`Y_1` given
:math:`Y_2` is :math:`{\rm Ga}(2,5+y_2)`, with moments :math:`{\rm
E}[Y_1|y_2]=(5+y_2)/2`, :math:`{\rm E}[Y_1^2|y_2]=(5+y_2)(6+y_2)/4` and
:math:`{\rm E}[Y_1^3|y_2]=(5+y_2)(6+y_2)(7+y_2)/8`. Then

:math:`{\rm E}[Y_1] = 0.5(5/2 + 6/2) = 2.75\,.`

We similarly find that :math:`{\rm E}[Y_1^2]=9` and :math:`{\rm
E}[Y_1^3]=34.125` and hence that :math:`V_{\{1,2\}} = V_{\{1\}}=4.67`. In
contrast we find :math:`V_{\{2\}}=0.17`, a small value that reflects the
limited way that learning about :math:`y_2` provides information about
:math:`y_1`.

Before ending this example we consider an additional complication. As
explained in the page describing uses of SA in the toolkit
(:ref:`DiscToolkitSensitivityAnalysis<DiscToolkitSensitivityAnalysis>`),
in the context of complex, computer-intensive simulators we will have
additional :ref:`code uncertainty<DefCodeUncertainty>` arising from
building an :ref:`emulator<DefEmulator>` of the simulator. All of the
quantities required in the decision-based SA calculation are now subject
to code uncertainty. Two approaches to incorporating code uncertainty
are discussed in
:ref:`DiscToolkitSensitivityAnalysis<DiscToolkitSensitivityAnalysis>`.

In the approach characterised as decision under code uncertainty, we
optimise the posterior expected loss :math:`\bar{L}^*(d) = {\rm
E}^*[\bar{L}(d)]:ref:`, where :math:`{\rm E}^*[\cdot]` denotes a posterior
expectation (in the case of a fully `Bayesian<DefBayesian>`
emulator, or the adjusted mean in the :ref:`Bayes
linear<DefBayesLinear>` case). In this example,
:math:`\bar{L}^*(d)` is the expectation of the posterior mean of
:math:`f(d,Y)`. We simply replace the emulator by its posterior mean and
carry out all the computations above.

However, it is often more appropriate to consider the second approach
that is characterised as code uncertainty about the decision. Code
uncertainty now induces a (posterior) distribution for :math:`\strut M`
whose mean is not generally just the result of minimising
:math:`\bar{L}^*(d)`. In general, the analysis for the optimisation problem
becomes much more complex in this approach.
