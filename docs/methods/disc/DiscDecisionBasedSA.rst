.. _DiscDecisionBasedSA:

Discussion: Decision-based Sensitivity Analysis
===============================================

Description and background
--------------------------

The basic ideas of :ref:`sensitivity
analysis<DefSensitivityAnalysis>` (SA) are presented in the
topic thread on sensitivity analysis
(:ref:`ThreadTopicSensitivityAnalysis<ThreadTopicSensitivityAnalysis>`).
We concentrate in the :ref:`MUCM<DefMUCM>` toolkit on probabilistic
SA, but the reasons for this choice and alternative approaches are
considered in the discussion page
:ref:`DiscWhyProbabilisticSA<DiscWhyProbabilisticSA>`. In
probabilistic SA, we assign a probability density function
:math:`\omega(x)` to represent uncertainty about the inputs.

Decision-based sensitivity analysis is a form of probabilistic
sensitivity analysis in which the primary concern is for how uncertainty
in inputs impacts on a decision. We suppose that the output of a
:ref:`simulator<DefSimulator>` is an input to this decision, and
because of uncertainty about the inputs there is uncertainty about the
output, and hence uncertainty as to which is the best decision.
Uncertainty about an individual input is important if by removing that
uncertainty we can expect to make much better decisions. We develop here
the principal measures of influence and sensitivity for individual
inputs and groups of inputs.

Notation
~~~~~~~~

The following notation and terminology is introduced in
:ref:`ThreadTopicSensitivityAnalysis<ThreadTopicSensitivityAnalysis>`.

In accordance with the standard :ref:`toolkit
notation<MetaNotation>`, we denote the simulator by :math:`f` and
its inputs by :math:`\strut x`. The focus of SA is the relationship between
:math:`\strut x` and the simulator output(s) :math:`f(x)`. Since SA also
typically tries to isolate the influences of individual inputs, or
groups of inputs, on the output(s), we let :math:`x_j` be the j-th element
of :math:`\strut x` and will refer to this as the j-th input, for
:math:`j=1,2,\ldots,p`, where as usual :math:`p` is the number of inputs. If
:math:`\strut J` is a subset of the indices :math:`\{1,2,\ldots,p\}`, then
:math:`x_J` will denote the corresponding subset of inputs. For instance,
if :math:`J=\{2,6\}` then :math:`x_J=x_{\{2,6\}}` comprises inputs 2 and 6,
while :math:`x_j` is the special case of :math:`x_J` when :math:`J=\{j\}`.
Finally, :math:`x_{-j}` will denote the whole of the inputs :math:`\strut x`
*except* :math:`x_j`, and similarly :math:`x_{-J}` will be the set of all
inputs except those in :math:`x_J`.

Formally, :math:`\omega(x)` is a joint probability density function for all
the inputs. The marginal density function for input :math:`x_j` is denoted
by :math:`\omega_j(x_j)`, while for the group of inputs :math:`x_J` the
density function is :math:`\omega_J(x_J)`. The conditional distribution for
:math:`x_{-J}` given :math:`x_J` is :math:`\omega_{-J|J}(x_{-J}\,|\,x_J)`. Note,
however, that it is common for the distribution :math:`\strut\omega` to be
such that the various inputs are statistically independent. In this case
the conditional distribution :math:`\omega_{-J|J}` does not depend on
:math:`x_J` and is identical to the marginal distribution :math:`\omega_{-J}`.

Note that by assigning probability distributions to the inputs we
formally treat those inputs as random variables. Notationally, it is
conventional in statistics to denote random variables by capital
letters, and this distinction is useful also in probabilistic SA. Thus,
the symbol :math:`\strut X` denotes the set of inputs when regarded as
random (i.e. uncertain), while :math:`\strut x` continues to denote a
particular set of input values. Similarly, :math:`X_J` represents those
random inputs with subscripts in the set :math:`\strut J`, while :math:`x_J`
denotes an actual value for those inputs.

Discussion
----------

Decision under uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~

| Decisions are hardest to make when their consequences are uncertain.
  The problem of decision-making in the face of uncertainty is addressed
  by statistical decision theory. Interest in
  :ref:`simulator<DefSimulator>` output uncertainty is often driven
  by the need to make decisions, where the simulator output :math:`f(x)` is
  a factor in that decision.
| In addition to the joint probability density function :math:`\omega(x)`
  which represents uncertainty about the inputs, we need two more
  components for a formal decision analysis.

#. *Decision set*. The set of available decisions is denoted by
   :math:`\strut\cal D`. We will denote an individual decision in
   :math:`\strut\cal D` by :math:`\strut d`.
#. *Loss function*. The loss function :math:`L(d,x)` expresses the
   consequences of taking decision :math:`\strut d` when the true inputs
   are :math:`\strut x`.

In order to understand the loss function, first note that we have shown
it as a function of :math:`\strut x`, but it only depends on the inputs
indirectly. The decision consequences actually depend on the output(s)
:math:`f(x)`, and it is uncertainty in the output that matters in the
decision. But the output is directly produced by the input, and so we
can write the loss function as a function of :math:`\strut x`. However, in
order to evaluate :math:`L(d,x)` at any :math:`\strut x` we will need to run
the simulator first to find :math:`f(x)`.

| The interpretation of the loss function is that it represents, on a
  suitable scale, a penalty for making a poor decision. To make the best
  decision we need to find the :math:`\strut d` that minimises the loss,
  but this depends on :math:`\strut x`. It is in this sense that
  uncertainty about (the simulator output and hence about) the inputs
  :math:`\strut x` makes the decision difficult. Uncertainty about
  :math:`\strut x` leads to uncertainty about the best decision. It is this
  decision uncertainty that is the focus of decision-based SA.
| Note finally that for convenience of exposition we refer to a loss
  function and a single simulator output. In decision analysis the loss
  function is often replaced by a utility function, such that higher
  utility is preferred and the best decision is the one which maximises
  utility. However, we can just interpret utility as negative loss.
  Also, :math:`f(x)` can be a vector of outputs (all of which may influence
  the decision) - all of the development presented here follows through
  unchanged in this case.

Value of perfect information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We do not know :math:`\strut x`, so it is a random variable :math:`\strut X`,
but we still have to take a decision. The optimal decision is the one
that minimises the *expected* loss

:math:`\bar L(d)=\mathrm{E}[L(d,X)] .`

The use of expectation is important here, because it relates to our
earlier statement that the loss function represents a penalty "on a
suitable scale". The way that loss is defined must be such that expected
loss is what matters. That is, a certain loss of 1 should be regarded as
equivalent to an uncertain loss which is 0 or 2 on the flip of a coin.
The expectation of the uncertain loss is :math:`0.5\times 0 + 0.5\times
2=1`. The formulation of loss functions (or utility functions) is an
important matter that is fundamental to decision theory - see references
at the end of this page.

If we denote this optimal decision by :math:`\strut M`, then :math:`\bar L(M) =
\\min_d \\bar L(d)`.

Suppose we were able to discover the true value of :math:`\strut x`. We let
:math:`M_\Omega(x)` be the best decision given the value of :math:`\strut x`.
(In later sections we will use the notation :math:`M_J(x_J)` to denote the
best decision given a subset of inputs :math:`x_J`). For each value of
:math:`\strut x` we have :math:`L(M_\Omega(x),x)=\min_d L(d,x)`

The impact of uncertainty about :math:`\strut X` can be measured by the
amount by which expected loss would be reduced if we could learn its
true value. Given that :math:`\strut X` is actually unknown, we compare
:math:`\bar L(M)` with the *expectation* of the uncertain
:math:`L(M_\Omega(X),X)`. The difference

:math:`V = \\bar L(M) - \\mathrm{E}[L(M_\Omega(X),X)]`

is known as the expected value of perfect information (EVPI).

Value here is measured in the very real currency of expected loss saved.
It is possible to show that :math:`\strut V` is always positive.

Value of imperfect information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In decision-based SA, we are interested particularly in the impact of
uncertainty in individual inputs or a group of inputs. Accordingly,
suppose that we were to learn the true value of input :math:`x_j`. Then our
optimal decision would be :math:`M_j(x_j)`, which minimises the expected
loss conditional on :math:`x_j`, i.e.

:math:`\bar L_j(d,x_j)=\mathrm{E}[L(d,X)\,|\,x_j]\,.`

As in the case of perfect information, we do not actually know the value
of :math:`x_j`, so the value of this information is the difference

:math:`V_j = \\bar L(M) - \\mathrm{E}[\bar L_j(M_j(X_j),X_j)]\,.`

More generally, if we were to learn the value of a group of inputs
:math:`x_J`, then the optimal decision would become :math:`M_J(x_J)`,
minimising

:math:`\bar L_J(d,x_J) = \\mathrm{E}[L(d,X)\,|\,x_J]\,,`

and the value of this information is

:math:`V_J = \\bar L(M) - \\mathrm{E}[\bar L_J(M_J(X_J),X_j)]\,.`

In each case we have perfect information about :math:`x_j` or :math:`x_J` but
no additional direct information about :math:`x_{-j}` or :math:`x_{-J}`. This
is a kind of imperfect information that is sometimes called "partial
perfect information". Naturally, the value of such information, :math:`V_j`
or :math:`V_J`, will be less than the EVPI.

Another kind of imperfect information is when we can get some additional
data :math:`\strut S`. For instance an input to the simulator might be the
mean effect of some medical treatment, and we can get data :math:`\strut S`
on a sample of patients. We can then calculate an expected value of
sample information (EVSI); see references below.

Additional comments
-------------------

Some examples to illustrate the concepts of decisions and decision-based
SA are presented in the example page
:ref:`ExamDecisionBasedSA<ExamDecisionBasedSA>`.

Relationship with variance-based SA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The basic decision-based SA measures are :math:`M_J(x_J)`, which
characterises the effect of a group :math:`x_J` of inputs in terms of how
the optimal decision changes when we fix :math:`x_J`, and the value of
information :math:`V_J`, which quantifies the expected loss reduction from
learning the value of :math:`x_J`. It is no coincidence that the same
symbols are used for the principal SA measures in variance-based SA,
given in the discussion page
:ref:`DiscVarianceBasedSA<DiscVarianceBasedSA>`. One of the examples
in :ref:`ExamDecisionBasedSA<ExamDecisionBasedSA>` shows that
variance-based SA can be obtained as a special case of decision-based
SA. In that example, the decision-based measures :math:`M_J(x_J)` and
:math:`V_J` reduce to the measures with the same symbols in variance-based
SA.

But although we can consider variance-based SA as arising from a special
kind of decision problem, its measures are also very natural ways to
think of sensitivity when there is no explicit decision problem. Thus
:math:`M_J(x_J)` is the mean effect of varying :math:`x_J`, while :math:`V_J` can
be interpreted both as the variance of :math:`M_J(X_J)` and as the expected
reduction of overall uncertainty from learning about :math:`x_J`. In
decision-based SA, :math:`M_J(x_J)` and :math:`V_J` are defined in ways that
are most appropriate to the decision context but they do not have such
intuitive interpretations. In particular, we note the following.

-  :math:`V_J` is not in general the variance of :math:`M_J(X_J)`. Although
   that variance might be interesting in the more general decision
   context, the definition of :math:`V_J` in terms of reduction in expected
   loss is more appropriate.
-  In :ref:`DiscVarianceBasedSA<DiscVarianceBasedSA>`, the mean
   effect :math:`M_J(x_J)` can be expressed in terms of main effects and
   interactions, but these are not generally useful in the wider
   decision context. For instance, in some decision problems the set
   :math:`\cal D` of possible decisions is discrete, and it is then not
   even meaningful to take averages or differences.
-  Even if :math:`\omega(x)` is such that inputs are statistically
   independent, there is no analogue in decision-based SA of the
   decomposition of overall uncertainty into main effect and interaction
   variances.

References
~~~~~~~~~~

The following are some standard texts on statistical decision analysis,
where more details about loss/utility functions, expected utility and
value of information can be found.

Smith, J.Q. *Decision Analysis: A Bayesian Approach*. Chapman and Hall.
1988.

Clemen, R. *Making Hard Decisions: An Introduction to Decision
Analysis*, 2nd edition. Belmont CA: Duxbury Press, 1996.

An example of decision based SA using emulators is given in the
following reference.

Oakley, J. E. (2009). Decision-theoretic sensitivity analysis for
complex computer models. *Technometrics* 51, 121-129.
