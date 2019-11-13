.. _DefSensitivityAnalysis:

Definition of Term: Sensitivity analysis
========================================

In general, sensitivity analysis is the process of studying how
sensitive the output (here this might be a single variable or a vector
of outputs) of a :ref:`simulator<DefSimulator>` is to changes in each
of its inputs (and sometimes to combinations of those inputs). There are
several different approaches to sensitivity analysis that have been
suggested. We can first characterise the methods as local or global.

-  *Local sensitivity analysis* consists of seeing how much the output
   changes when inputs are perturbed by minuscule amounts.

-  *Global sensitivity analysis* considers how the output changes when
   we vary the inputs by larger amounts, reflecting in some sense the
   range of values of interest for those inputs.

In both approaches, it is usual to specify a *base* set of input values,
and to see how the output changes when inputs are varied from from their
base values. Essentially, local sensitivity analysis studies the
(partial) derivatives of the output with respect to the inputs,
evaluated at the base values. Global sensitivity analysis is more
complex, because there are a number of different ways to measure the
effect on the output, and a number of different ways to define how the
inputs are perturbed.

Local sensitivity analysis is very limited because it only looks at the
influence of the inputs in a tiny neighbourhood around the base values,
and the derivatives are themselves highly sensitive to the scale of
measurement of the inputs. For instance, if we decide to measure an
input in kilometres instead of metres, then its derivative will be 1000
times larger. Global sensitivity analyses are also influenced by how far
we consider varying the inputs. We need a well-defined reason for the
choice of ranges, otherwise the sensitivity measures are again arbitrary
in the same way as local sensitivity measures respond to an arbitrary
scale of measurement.

In the :ref:`MUCM<DefMUCM>` toolkit we generally consider only
probabilistic sensitivity analysis, in which the amounts of perturbation
are defined by a (joint) probability distribution over the inputs. The
usual interpretation of this distribution is as measuring the
uncertainty that we have about what the best or "true" values of those
inputs should be, in which case the distribution is well defined.

With respect to such a distribution we can define main effects and
interactions as follows. Let the model output for input vector :math:`X` be
:math:`f(X)`. Let the i-th input be :math:`X_i`, and as usual the symbol
:math:`\textrm{E}[\cdot]` denotes expectation with respect to the
probability distribution defined for :math:`X`.

-  The main effect of an input :math:`X_i` is the function
   :math:`I_i(x_i)=\textrm{E}[f(X)\,|\,X_i=x_i] - \textrm{E}[f(X)]`. So we
   first take the expected value of the output, averaged over the
   distribution of all the other inputs conditional on the value of
   :math:`X_i` being :math:`x_i`, then we subtract the overall expected value
   of the output, averaged over the distribution of all the inputs.

-  The interaction between inputs :math:`X_i` and :math:`X_j` is the function
   :math:`I_{\{i,j\}}(x_i,x_j)=\textrm{E}[f(X)\,|\,X_i=x_i, X_j=x_j] -
   I_i(x_i) - I_j(x_j) - \textrm{E}[f(X)]`. This represents
   deviation in the joint effect of varying the two inputs after
   subtracting their main effects.

Higher order interactions, involving more than two inputs, are defined
analogously.

The main effects and interactions provide a very detailed analysis of
how the output responds to the inputs, but we often require simpler,
single-figure measures of the sensitivity of the output to individual
inputs or combinations of inputs. There are two main ways to do this -
:ref:`variance based<DefVarianceBasedSA>` and :ref:`decision
based<DefDecisionBasedSA>` sensitivity analysis.
