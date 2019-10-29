.. _DiscWhyProbabilisticSA:

Discussion: Why Probabilistic Sensitivity Analysis?
===================================================

Description and background
--------------------------

The uses of :ref:`sensitivity analysis<DefSensitivityAnalysis>` (SA)
are outlined in the topic thread on SA
(:ref:`ThreadTopicSensitivityAnalysis<ThreadTopicSensitivityAnalysis>`),
where they are related to the methods of probabilistic SA that are
employed in the :ref:`MUCM<DefMUCM>` toolkit. We discuss here
alternative approaches to SA and why probabilistic SA is preferred.

First we review some notation and terminology that is introduced in
:ref:`ThreadTopicSensitivityAnalysis<ThreadTopicSensitivityAnalysis>`.

In accordance with the standard :ref:`toolkit
notation<MetaNotation>`, we denote the
:ref:`simulator<DefSimulator>` by :math:`f` and its inputs by :math:`\strut
x`. The focus of SA is the relationship between :math:`\strut x` and the
simulator output(s) :math:`f(x)`. Since SA also typically tries to isolate
the influences of individual inputs, or groups of inputs, on the
output(s), we let :math:`x_j` be the j-th element of :math:`\strut x` and will
refer to this as the j-th input, for :math:`j=1,2,\ldots,p`, where as usual
:math:`p` is the number of inputs. If :math:`\strut J` is a subset of the
indices :math:`\{1,2,\ldots,p\}`, then :math:`x_J` will denote the
corresponding subset of inputs. For instance, if :math:`J=\{2,6\}` then
:math:`x_J=x_{\{2,6\}}` comprises inputs 2 and 6, while :math:`x_j` is the
special case of :math:`x_J` when :math:`J=\{j\}`. Finally, :math:`x_{-j}` will
denote the whole of the inputs :math:`\strut x` *except* :math:`x_j`, and
similarly :math:`x_{-J}` will be the set of all inputs except those in
:math:`x_J`.

Discussion
----------

Local SA
~~~~~~~~

SA began as a response to concern for the consequences of mis-specifying
the values of inputs to a :ref:`simulator<DefSimulator>`. The
simulator user could provide values :math:`\strut\hat x` that would be
regarded as best estimates for the inputs, and so :math:`f(\hat x)` would
in some sense be an estimate for the corresponding simulator output(s).
However, if the correct inputs differed from :math:`\hat x` then the
correct output would differ from :math:`f(\hat x)`. The output would be
regarded as sensitive to a particular input :math:`x_j` if the output
changed substantially when :math:`x_j` was perturbed slightly. This led to
the idea of measuring sensitivity by differentiation of the function.
The measure of sensitivity to :math:`x_j` was the derivative :math:`\partial
f(x)/\partial x_j`, evaluated at :math:`x=\hat x`.

SA based on derivatives has a number of deficiencies, however. The
differential measures only the impact of an infinitesimal change in
:math:`x_j`, and for this reason this kind of SA is referred to as local
SA. If the response of the output to :math:`x_j` is far from linear, then
perturbing :math:`x_j` more than a tiny amount might have an effect that is
not well represented by the derivative.

More seriously, the derivative is not invariant to the units of
measurement. If, for instance, we choose to measure :math:`x_j` in
kilometres rather than metres, then :math:`\partial f(x)/\partial x_j` will
change by a factor of 1000. The output may appear to be more sensitive
to input :math:`x_j` than to :math:`x_{j^\prime}` because the derivative
evaluated at :math:`\hat x` is larger, but this ordering could easily be
reversed if we changed the scales of measurement.

One-way SA
~~~~~~~~~~

Alternatives to local SA based on derivatives are known as global SA
methods. A simple global method involves perturbing :math:`x_j` from its
nominal value :math:`\hat x_j`, say to a point :math:`x^\prime_j`, with all
other inputs held at their nominal values :math:`\hat x_{-j}`. The
resulting change in output is then regarded as a measure of sensitivity
to :math:`x_j`. This is known as one-way SA, because the inputs are varied
only one at a time from their nominal values.

One-way SA addresses the problems noted for local SA. First, we do not
consider only infinitesimal perturbations, and any nonlinearity in the
response to :math:`x_j` is accounted for in the evaluation of the output at
the new point (where :math:`x_j=x^\prime_j` but :math:`x_{-j}=\hat x_{-j}`).
Second, if we change the units of measurement for :math:`x_j` then this
will be reflected in both :math:`\hat x_j` and :math:`x^\prime_j` and the SA
measure will be unaffected.

However, one-way SA has its own problems. The SA measures depend on how
far we perturb the individual inputs, and the ordering of inputs
produced by their SA measures can change if we change the
:math:`x^\prime_j` values.

Also, one-way SA fails to quantify joint effects of perturbing more than
one input together. For instance, we have measures for :math:`x_1` and
:math:`x_2` but not for :math:`x_{\{1,2\}}`.The effect of perturbing :math:`x_1`
and :math:`x_2` together cannot be inferred from knowing the effects of
perturbing them individually. Statisticians say that two inputs interact
when the effect of perturbing both is not just the sum of the effects of
perturbing them individually. One-way SA is not able to measure
interactions.

Multi-way SA
~~~~~~~~~~~~

| In the wider context of experimental design, statisticians have for
  more than 50 years decried the idea of varying factors one at a time
  for precisely the reason that such an experiment cannot identify
  interactions between the effects of two or more factors. Statistical
  experimental design involves varying the factors together, the
  principal classical designs being various forms of factorial design.
  SA based on varying the inputs together rather than individually is
  called multi-way SA.
| Through techniques analogous to the method of analysis of variance in
  Statistics, multi-way sensitivity analysis can identify interaction
  effects.

Regression SA
~~~~~~~~~~~~~

| Thorough multi-way SA typically demands a large and highly structured
  set of simulator runs, even for quite modest numbers of inputs. In the
  same way as the analysis of variance is a particular case of
  regression analysis in Statistics, an alternative to multi-way SA is
  based on regression. Analysis involves fitting regression models to
  the outputs of available simulator runs.
| If the regression model is a simple linear regression, the fitted
  slope parameters for the various inputs represent measures of
  sensitivity. However, such an approach cannot identify interactions,
  and shares most of the drawbacks of one-way SA. More thorough analysis
  will fit product terms for interactions, and potentially also
  nonlinear terms.

Probabilistic SA
~~~~~~~~~~~~~~~~

| Careful use and interpretation of multi-way or regression SA methods
  can yield quite comprehensive analysis of the relationship between the
  simulator's output and its inputs, for the purposes of understanding
  and/or dimension reduction. However, probabilistic SA was developed
  specifically to address the use of SA in the context of uncertain
  inputs. As remarked above, it was in response to uncertainty about
  inputs that SA evolved, but all of the preceding methods treat the
  uncertainty in the inputs only implicitly. In probabilistic SA the
  input uncertainty is explicit and described in a probability
  distribution :math:`\omega(x)`.
| Probabilistic SA is also a comprehensive approach to SA that can
  address interactions and nonlinearities, and it is preferred in the
  :ref:`MUCM<DefMUCM>` toolkit because it uniquely has the ability to
  characterise the relationship between input uncertainty and output
  uncertainty. It also extends naturally to address SA for
  decision-making.

Additional comments
-------------------

Screening methods
~~~~~~~~~~~~~~~~~

Simulators often have very large numbers of inputs. Carrying out
sophisticated SA methods on such a simulator can be highly demanding
computationally. Only a small number of the inputs will generally have
appreciable impact on the output, particularly when we are interested in
the simulator's behaviour over a relatively small part of the possible
input space. In practice, simple :ref:`screening<DefScreening>`
techniques are widely used initially to eliminate many inputs that have
essentially no effect. Once this kind of drastic dimension reduction has
been carried out, more formal and demanding SA techniques can be used to
confirm and further refine the choice of :ref:`active
inputs<DefActiveInput>`. For a discussion of screening methods
see the topic thread on screening
(:ref:`ThreadTopicScreening<ThreadTopicScreening>`).

Range of variation
~~~~~~~~~~~~~~~~~~

| When we consider sensitivity of the output(s) to variations in an
  input, it can be important to define how (and in particular how far)
  that input might be varied. If we allow larger variations in :math:`x_j`
  then we will often (although not necessarily) see larger impact on
  :math:`f(x)`. In all of the approaches to SA discussed above, this issue
  arises. It is most obvious in one-way SA, where we have to specify the
  particular alternative value(s) for :math:`x_j`, and clearly if we change
  those we may change the sensitivity and influence measures. It is also
  obvious in probabilistic SA, where the probability distribution
  assigned to :math:`\strut x` identifies in detail how the inputs are to
  be considered to vary.
| As we have seen, the question of how (far) we allow :math:`x_j` to vary
  does not obviously seem relevant at all for local SA, but the fact
  that the derivatives are not invariant to scale changes is an analgous
  issue. The case of regression SA is similar, because a coefficient in
  a simple linear regression fit is a gradient in the same way as a
  derivative and, in particular, changing the units of measurement
  affects such a coefficient in the same way. There is another aspect to
  the question for regression SA, though. If we fit a simple linear
  regression when :math:`f(x)` genuinely responds linearly to varying
  :math:`x_j`, then the fitted gradient coefficient will not depend (or
  only minimally) on the range of variation we specify for :math:`x_j`, but
  if the underlying response of :math:`f(x)` is non-linear then the
  coefficient in a linear fit will change if the range of variation
  changes.

In general, measures of sensitivity depend on how we consider perturbing
the inputs. Changing the range of variation will change the SA measures
and can alter the ranking of inputs according to which the outputs are
most sensitive to. Probabilistic SA is no exception, but its emphasis on
the careful specification of a probability distribution :math:`\omega(x)`
over the input space avoids the often arbitrary way in which variations
are defined in other SA approaches.
