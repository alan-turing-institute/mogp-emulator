.. _DiscSensitivityAndOutputUncertainty:

Discussion: Sensitivity measures for output uncertainty
=======================================================

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
concerning a :ref:`simulator's<DefSimulator>` output is induced by
uncertainty in its inputs.

Uncertainty in the inputs is described by a joint probability density
function :math:`\omega(x)`, whose specification will be considered as part
of the discussion below.

Notation
~~~~~~~~

The following notation and terminology is introduced in
:ref:`ThreadTopicSensitivityAnalysis<ThreadTopicSensitivityAnalysis>`.

In accordance with the standard :ref:`toolkit
notation<MetaNotation>`, we denote the simulator by :math:`f` and
its inputs by :math:`x`. The focus of SA is the relationship between
:math:`x` and the simulator output(s) :math:`f(x)`. Since SA also
typically tries to isolate the influences of individual inputs, or
groups of inputs, on the output(s), we let :math:`x_j` be the j-th element
of :math:`x` and will refer to this as the j-th input, for
:math:`j=1,2,\ldots,p`, where as usual :math:`p` is the number of inputs. If
:math:`J` is a subset of the indices :math:`\{1,2,\ldots,p\}`, then
:math:`x_J` will denote the corresponding subset of inputs. For instance,
if :math:`J=\{2,6\}` then :math:`x_J=x_{\{2,6\}}` comprises inputs 2 and 6,
while :math:`x_j` is the special case of :math:`x_J` when :math:`J=\{j\}`.
Finally, :math:`x_{-j}` will denote the whole of the inputs :math:`x`
*except* :math:`x_j`, and similarly :math:`x_{-J}` will be the set of all
inputs except those in :math:`x_J`.

Discussion
----------

We consider here how the methods of probabilistic SA can be used for
analysing output uncertainty. All of the SA measures recommended here
are defined and discussed in page
:ref:`DiscVarianceBasedSA<DiscVarianceBasedSA>`.

We initially assume that interest focuses on a single simulator output
before briefly considering the case of multiple outputs.

Specifying :math:`\omega(x)`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The role of :math:`\omega(x)` is to represent uncertainty about the
simulator inputs :math:`x`. As such, it should reflect the best
available knowledge about the true, but uncertain, values of the inputs.
Typically, this distribution will be centred on whatever are the best
current estimates of the inputs, with a spread around those estimates
that faithfully describes the degree of uncertainty. However, turning
this intuitive impression into an appropriate probability distribution
is not a simple process.

The basic technique for specifying :math:`\omega(x)` is known as
:ref:`elicitation<DefElicitation>`, which is the methodology whereby
expert knowledge/uncertainty is converted into a probability
distribution. Some resources for elicitation may be found in the
"Additional comments" section below.

Elicitation of a probability distribution for several uncertain
quantities is difficult, and it is common to make a simplifying
assumption that the various inputs are independent. Formally, this
assumption implies that if you were to obtain addtional information
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

.. math::
   \omega(x) = \prod_{j=1}^p \omega_j(x_j).

Independence also simplifies the interpretation of some of the SA
measures described in
:ref:`DiscVarianceBasedSA<DiscVarianceBasedSA>`.

Uncertainty analysis
~~~~~~~~~~~~~~~~~~~~

Uncertainty about :math:`x` induces uncertainty in the simulator
output :math:`f(x)`. The task of measuring and describing that uncertainty
is known as :ref:`uncertainty analysis<DefUncertaintyAnalysis>` (UA).
Formally, we regard the uncertain inputs as a random variable :math:`X`
(conventionally, random variables are denoted by capital letters in
Statistics). Then the output :math:`f(X)` is also a random variable and has
a probability distribution known as the uncertainty distribution. Some
of the most widely used measures of output uncertainty in UA are as
follows.

-  The distribution function :math:`F(c)=Pr(f(X)\le c)`.
-  The uncertainty mean :math:`M=\mathrm{E}[f(X)]`.
-  The uncertainty variance :math:`V=\mathrm{Var}[f(X)]`.
-  The exceedance probability :math:`\bar F(c)=1-F(c)`, that :math:`f(X)`
   exceeds some threshhold :math:`c`.

Sensitivity
~~~~~~~~~~~

The goal of SA, as opposed to UA, is to analyse the output uncertainty
so as to understand which uncertain inputs are most responsible for the
output uncertainty. If uncertainty in a given input :math:`x_j` is
accountable for a large part of the output uncertainty, then the output
is said to be very sensitive to :math:`x_j`. Therefore, SA explores the
relative sensitivities of the inputs, both individually and in groups.

Output uncertainty is primarily summarised by the variance :math:`V`.
As defined in :ref:`DiscVarianceBasedSA<DiscVarianceBasedSA>`,
the proportion of this overall variance that can be attributed to a
group of inputs :math:`x_J` is given by the sensitivity index :math:`S_J` or
by the total sensitivity index :math:`T_J`.

Formally, :math:`S_J` is the expected amount by which uncertainty would be
reduced if we were to learn the true values of the inputs in :math:`x_J`.
For instance, if :math:`S_J=0.25` then learning the true value of :math:`x_J`
would reduce output uncertainty by 25%.

On the other hand, :math:`T_J` is the expected proportion of uncertainty
remaining if we were to learn the true values of all the *other* inputs,
i.e. :math:`x_{-J}`. As explained in
:ref:`DiscVarianceBasedSA<DiscVarianceBasedSA>`, when inputs are
independent :math:`T_J` will be larger than :math:`S_J` by an amount
indicating the magnitude of interactions between inputs in :math:`x_J` and
other inputs outside the group.

If there is an interaction between inputs :math:`x_j` and :math:`x_{j'}` then
(again assuming independence) the sensitivity index :math:`S_{\{j,j'\}}`
for the two inputs together is greater than the sum of their individual
sensitivity indices :math:`S_j` and :math:`S_{j'}`. So interactions can be
important in identifying which groups of inputs have the most influence
on the output.

Prioritising research
~~~~~~~~~~~~~~~~~~~~~

One reason for this kind of SA is to determine whether it would be
useful to carry out some research to reduce uncertainty about one or
more of the inputs. In general, there would be little value in
conducting such research to learn about the true value of an input whose
sensitivity index is very small. An input (or input group) with a high
sensitivity index is more likely to be a priority for research effort.

However, a more careful consideration of research priorities would
involve other factors. First of these would be cost. An input may have a
high sensitivity index but still might not be a priority for research if
the cost of investigating it would be very high. Another factor is the
purpose for which the research is envisaged. The primary objective may
be more complex than simply reducing uncertainty about :math:`f(X)`. The
reason why we are interested in :math:`f(X)` in the first place is likely
to be as an input to some decision problem, and the importance of input
uncertainty is then not simply that it causes output uncertainty but
that it causes decision uncertainty, i.e. uncertainty about the best
decision. This takes into the realm of decision-based SA; see
:ref:`DiscDecisionBasedSA<DiscDecisionBasedSA>`.

Exceedances
~~~~~~~~~~~

Although overall uncertainty, as measured by :math:`V`, is generally
the most important basis for determining sensitivity, interest may
sometimes focus on other aspects of the uncertainty distribution. If
there is a decision problem, for instance, even if we do not wish to
pursue the methods of decision-based SA, the decision context may
suggest some function of the output :math:`f(X)` that is of more interest
than the output itself. Then SA based on the variance of that function
may be more useful. We present a simple example here.

Suppose that interest focuses on whether :math:`f(X)` exceeds some
threshhold :math:`c`. Instead of :math:`f(x)` we consider as our output
:math:`f_c(x)`, which takes the value :math:`f_c(x)=1` if :math:`f(X)>c` and
otherwise :math:`f_c(x)=0`. Now the uncertainty mean :math:`M` is just
the exceedance probability :math:`M=\bar F(c)` and the uncertainty variance
can be shown to be :math:`V=\bar F(c)\{1-\bar F(c)\}`.

The mean effect of inputs :math:`x_J` becomes

.. math::
   M_J(x_J) = Pr(f(X)>c\,|\,x_J)

and the sensitivity variance :math:`V_J` is the variance of this mean
effect with respect to the distribution of :math:`x_J`. The corresponding
sensitivity index :math:`S_J = V_J / V` then measures the extent to which
:math:`x_J` influences the probability of exceedance.

Multiple outputs
~~~~~~~~~~~~~~~~

When the simulator produces multiple outputs, then we may be interested
in uncertainty about all of these outputs. Although
:ref:`DiscVarianceBasedSA<DiscVarianceBasedSA>` describes how then we
can generalise the sensitivity variance :math:`V_J` to a matrix, there is
generally little extra value to be gained from looking at sensitivity of
multiple outputs in this way. It is usually adequate to identify the
inputs that most influence each of the outputs separately. However, this
will typically lead to different groups of inputs being most important
for different inputs, and it is no longer clear which ones are
candidates for research prioritisation. In practice, the solution is
again to think about the decision context and to use the methods of
decision-based SA.

Additional comments
-------------------

Transformation of the output may make for simpler SA. If, for instance,
:math:`f(x)` must be positive but can vary through two or more orders of
magnitude (a factor of 100 or more) then working with its logarithm,
:math:`\log f(x)`, is worth considering. There may be fewer important
inputs and fewer interactions on the logarithmic scale.

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
`http://tonyohagan.co.uk/shelf <http://tonyohagan.co.uk/shelf>`_
(:ref:`Disclaimer<MetaSoftwareDisclaimer>`)
