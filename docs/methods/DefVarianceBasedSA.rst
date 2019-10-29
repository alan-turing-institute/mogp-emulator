.. _DefVarianceBasedSA:

Definition of Term: Variance-based sensitivity analysis
=======================================================

In variance-based :ref:`sensitivity analysis<DefSensitivityAnalysis>`
we consider the effect on the output :math::ref:`f(X)` of a
`simulator<DefSimulator>` as we vary the inputs :math:`X`, when the
variation of those inputs is described by a (joint) probability
distribution. This probability distribution can be interpreted as
describing uncertainty about the best or true values for the inputs.

We measure the sensitivity to an individual input :math:`X_i` by the amount
of the variance in the output that is attributable to that input. So we
begin by considering the variance :math:`\textrm{Var}[f(X)]`, which
represents total uncertainty about the output that is induced by
uncertainty about the inputs. In the multi output case this is the
variance matrix, which has diagonal elements equivalent to the
sensitivity of each output dimension, and cross covariances
corresponding to joint sensitivities. This variance also arises as an
overall measure of uncertainty in :ref:`uncertainty
analysis<DefUncertaintyAnalysis>`.

If we were to remove the uncertainty in :math:`X_i`, then we would expect
the uncertainty in :math:`f(X)` to reduce. The amount of this expected
reduction is :math:` V_i = \\textrm{Var}[f(X)] -
\\textrm{E}[\textrm{Var}[f(X)\,|\,X_i]] \`. Notice that the second term
in this expression involves first finding the conditional variance of
:math:`f(X)` given that :math:`X_i` takes some value :math:`x_i`, which is the
uncertainty we would have about the simulator output if we were certain
that :math:`X_i` had that value, so this is the reduced uncertainty in the
output. But we are currently uncertain about :math:`X_i`, so we take an
expectation of that conditional variance with respect to our current
uncertainty in :math:`X_i`.

Thus, :math:`V_i` is defined as the variance-based sensitivity variance for
the i-th input. Referring to the definition of the main effect
:math::ref:`I_i(x_i)` in the `sensitivity analysis
definition<DefSensitivityAnalysis>`, it can be shown that
:math:`V_i` is the variance of this main effect. Again for the multi output
case this will be a variance matrix, with the main effect being a
vector.

We can similarly define the variance-based interaction variance
:math:`V_{\{i,j\}}` of inputs :math:`X_i` and :math:`X_j` as the variance of the
interaction effect :math:`I_{\{i,j\}}(x_i,x_j)`. When the probability
distribution on the inputs is such that they are all independent, then
it can be shown that the main effects and interactions (including the
higher order interactions that involve three or more inputs) sum to the
total variance :math:`\textrm{Var}[f(X)]`.

The sensitivity variance is often expressed as a proportion of the
overall variance :math:`V= \\textrm{Var}[f(X)]`. The ratio :math:`S_i=V_i/V`
is referred to as the sensitivity index for the i-th input, and
sensitivity indices for interactions are similarly defined.

Another index of sensitivity for the i-th input that is sometimes used
is the total sensitivity index :math:` T_i =
\\textrm{E}[\textrm{Var}[f(X)\,|\,X_{-i}]]/V \`, where :math:`X_{-i}`
means all the inputs *except* :math:`X_i`. This is the expected proportion
of uncertainty in the model output that would be left if we removed the
uncertainty in all the inputs except the i-th. In the case of
independent inputs, it can be shown that :math:`T_i` is the sum of the main
effect index :math:`S_i` and the interaction indices for all the
interactions involving :math:`X_i` and one or more other inputs.
