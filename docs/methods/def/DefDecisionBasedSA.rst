.. _DefDecisionBasedSA:

Definition of Term: Decision-based sensitivity analysis
=======================================================

In decision-based :ref:`sensitivity analysis<DefSensitivityAnalysis>`
we consider the effect on a decision which will be based on the output
:math:`f(X)` of a :ref:`simulator<DefSimulator>` as we vary the inputs
:math:`X`, when the variation of those inputs is described by a (joint)
probability distribution. This probability distribution can be
interpreted as describing uncertainty about the best or true values for
the inputs.

We measure the sensitivity to an individual input :math:`X_i` by the extent
to which we would be able to make a better decision if we could remove
the uncertainty in that input.

A decision problem is characterised by a set of possible decisions and a
utility function that gives a value :math:`U(d,f(x))` if we take decision
:math:`d` and the true value for the input vector is :math:`x`. The optimal
decision, given the uncertainty in :math:`X`, is the one which maximises
the expected utility. Let :math:`U^*` be the resulting maximised expected
utility based on the current uncertainty in the inputs.

If we were to remove the uncertainty in the i-th input by learning that
its true value is :math:`X_i = x_i`, then we might make a different
decision. We would now take the expected utility with respect to the
*conditional* distribution of :math:`X` given that :math:`X_i = x_i`, and then
maximise this with respect to the decision :math:`d`. Let :math:`U^*_i(x_i)`
be the resulting maximised expected utility. This of course depends on
the true value :math:`x_i` of :math:`X_i`, which we do not know. The
decision-based sensitivity measure for the i-th input is then the value
of learning the true value of :math:`X_i` in terms of improved expected
utility, i.e. :math:`V_i = \text{E}[U^*_i(X_i)] - U^*`, where the
expectation in the first term is with respect to the marginal
distribution of :math:`X_i`.

We can similarly define the sensitivity measure for two or more inputs
as being the value of learning the true values of all of these.

:ref:`Variance-based<DefVarianceBasedSA>` sensitivity analysis is a
special case of decision-based analysis, when the decision is simply to
estimate the true output :math:`f(X)` and the utility function is negative
squared error. In practice, though, variance-based sensitivity analysis
provides natural measures of sensitivity when there is no specific
decision problem.
