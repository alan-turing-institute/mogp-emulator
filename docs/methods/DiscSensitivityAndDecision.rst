.. _DiscSensitivityAndDecision:

Discussion: Sensitivity measures for decision uncertainty
=========================================================

Description and background
--------------------------

The basic ideas of :ref:`sensitivity
analysis<DefSensitivityAnalysis>` (SA) are presented in the
topic thread on sensitivity analysis
(:ref:`ThreadTopicSensitivityAnalysis<ThreadTopicSensitivityAnalysis>`).
We concentrate in the :ref:`MUCM<DefMUCM>` toolkit on probabilistic
SA, but the reasons for this choice and alternative approaches are
considered in the discussion page
:ref:`DiscWhyProbabilisticSA<DiscWhyProbabilisticSA>`.
:ref:`ThreadTopicSensitivityAnalysis<ThreadTopicSensitivityAnalysis>`
also outlines four general uses for SA; we discuss here the SA measures
appropriate to one of those uses - analysing the way that uncertainty
concerning the consequences of a decision is induced by uncertainty in
its inputs.

Examples of decisions that might be based on simulator output are not
difficult to find. Consider a simulator which models the response of the
global climate to atmospheric CO\(_2` levels. The simulator will
predict global warming and rising sea levels based on future carbon
emissions scenarios, and we can imagine a national policy decision
whether to build sea defences to protect a coastal area or city.
Uncertainty in the future carbon emissions and climate response to
increased CO\(_2` mean that the consequences of buiding or not building
sea defences are uncertain. Another decision based on this simulator
might involve setting policy on power station emissions to try to
control the nation's contribution to atmospheric CO\(_2`.

Uncertainty in the inputs is described by a joint probability density
function :math:`\omega(x)`, whose specification will be considered as part
of the discussion below.

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

Discussion
----------

We consider here how the methods of probabilistic SA can be used for
analysing decision uncertainty. All of the SA measures recommended here
are defined and discussed in page
:ref:`DiscDecisionBasedSA<DiscDecisionBasedSA>`.

Specifying :math:`\omega(x)`
~~~~~~~~~~~~~~~~~~~~~~~~~

The role of :math:`\omega(x)` is to represent uncertainty about the
simulator inputs :math:`\strut x`. As such, it should reflect the best
available knowledge about the true, but uncertain, values of the inputs.
Typically, this distribution will be centred on whatever are the best
current estimates of the inputs, with a spread around those estimates
that faithfully describes the degree of uncertainty. However, turning
this intuitive impression into an appropriate probability distribution
is not a simple process.

The basic technique for specifying :math::ref:`\omega(x)` is known as
`elicitation<DefElicitation>`, which is the methodology whereby
expert knowledge/uncertainty is converted into a probability
distribution. Some resources for elicitation may be found in the
"Additional comments" section below.

Elicitation of a probability distribution for several uncertain
quantities is difficult, and it is common to make a simplifying
assumption that the various inputs are independent. Formally, this
assumption implies that if you were to obtain additional information
about some of the inputs (which would generally reduce your uncertainty
about those inputs) then your knowledge/uncertainty about the other
inputs would not change. Independence is quite a strong assumption but
has the benefit that it greatly simplifies the task of elicitation.
Instead of having to think about uncertainty concerning all the various
inputs together, it is enough to think about the uncertainty regarding
each individual input :math:`x_j` separately. We specify thereby the
marginal density function :math:`\omega_j(x_j)` for each input separately,
and the joint density function :math:`\omega(x)` is just the product of
these marginal density functions:

:math:`\omega(x) = \\prod_{j=1}^p \\omega_j(x_j)\,.`

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

| The interpretation of the loss function is that it represents, on a
  suitable scale, a penalty for making a poor decision. To make the best
  decision we need to find the :math:`\strut d` that minimises the loss,
  but this depends on :math:`\strut x`. It is in this sense that
  uncertainty about (the simulator output and hence about) the inputs
  :math:`\strut x` makes the decision difficult. Uncertainty about
  :math:`\strut x` leads to uncertainty about the best decision. It is this
  decision uncertainty that is the focus of decision-based SA.
| There is more detailed discussion of the loss function in
  :ref:`DiscDecisionBasedSA<DiscDecisionBasedSA>`, and examples may
  be found in the example page
  :ref:`ExamDecisionBasedSA<ExamDecisionBasedSA>`.

Sensitivity
~~~~~~~~~~~

Consider the effect of uncertainty in a group of inputs :math:`x_J`; the
case of a single input :math:`x_j` is then included through :math:`J=\{j\}`.
As far as the decision problem is concerned, the effect of :math:`x_J` is
shown in the function :math:`M_J(x_J)`. This is the optimal decision
expressed as a function of :math:`x_J`. The optimal decision is the one
that we would take if we learnt the true value of :math:`x_J` (but
otherwise learnt nothing about :math:`x_{-J}`).

If the optimal decision :math:`M_J(x_J)` were the same for all :math:`x_J`,
then clearly the uncertainty about :math:`x_J` would be irrelevant, so in
some sense the more :math:`M_J(x_J)` varies with :math:`x_J` the more
influential this group of inputs is. However, whilst it is of interest
if the decision changes with :math:`x_J`, the true measure of importance of
this decision uncertainty is whether, by choosing different decisions
for different :math:`x_J`, we expect to make much *better* decisions. That
is, how much would we expect the loss to reduce if we were learn the
value of :math:`x_J`? The appropriate measure is the expected value of
learning :math:`x_J`, which is denoted by :math:`V_J`.

Thus, :math:`V_J` is our primary SA measure.

Prioritising research
~~~~~~~~~~~~~~~~~~~~~

One reason for this kind of SA is to determine whether it would be
useful to carry out some research to reduce uncertainty about one or
more of the inputs. Decision-based SA is the ideal framework for
considering such questions, because we can explicitly value the
research. Such research will not usually be able to identify precisely
the values of one or more inputs, but :math:`V_J` represents an upper bound
on the value of any research aimed at improving our understanding of
:math:`x_J`.

More precise values can be given to research using the idea of the
expected value of sample information (EVSI), which is outlined in
:ref:`DiscDecisionBasedSA<DiscDecisionBasedSA>`.

We can compare the value of research directly with what that research
would cost. This is particularly easy if the loss function is measured
in financial terms, so that :math:`V_J` (or a more precise EVSI) becomes
equivalent to a sum of money. Loss functions for commercial decisions
are often framed in monetary terms, but when loss is on some other scale
the comparison is less straightforward. Nevertheless, quantifying the
effect on decision uncertainty in this way is the best basis for
deciding on the cost-effectiveness of research.

Additional comments
-------------------

The following resources on elicitation will help with the process of
specifying :math:`\omega(x)`. The first is a thorough review of the field
of elicitation, and provides a wealth of general background information
on ideas and methods. The second (SHELF) is a package of documents and
simple software that is designed to help those with less experience of
elicitation to elicit expert knowledge effectively. SHELF is based on
the authors' own experiences and represents current best practice in the
field.

O'Hagan, A., Buck, C. E., Daneshkhah, A., Eiser, J. R., Garthwaite, P.
H., Jenkinson, D. J., Oakley, J. E. and Rakow, T. (2006). Uncertain
Judgements: Eliciting Expert Probabilities. John Wiley and Sons,
Chichester. 328pp. ISBN 0-470-02999-4.

SHELF - the Sheffield Elicitation Framework - can be downloaded from
http://tonyohagan.co.uk/shelf
(:ref:`Disclaimer<MetaSoftwareDisclaimer>`)
