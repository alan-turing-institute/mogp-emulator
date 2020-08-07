.. _ThreadTopicSensitivityAnalysis:

Thread: Sensitivity analysis
============================

Overview
--------

This is a topic thread, on the topic of :ref:`sensitivity
analysis<DefSensitivityAnalysis>`. Topic threads are designed to
provide additional background information on a topic, and to link to
places where the topic is developed in the core, variant and generic
threads (see the discussion of different kinds of threads in the Toolkit
structure page (:ref:`MetaToolkitStructure<MetaToolkitStructure>`)).

The various forms of sensitivity analysis (SA) are tools for studying
the relationship between a :ref:`simulator<DefSimulator>`'s inputs
and outputs. They are widely used to undestand the behaviour of the
simulator, to identify which inputs have the strongest influence on
outputs and to put a value on learning more about uncertain inputs. It
can be a prelude to simplifying the simulator, or to constructing a
simplified :ref:`emulator<DefEmulator>`, in which the number of
inputs is reduced.

Procedures for carrying out SA using emulators are given in most of the
core, variant and generic threads in the toolkit. This topic thread
places those specific tools and procedures in the wider context of SA
methods and discusses the practical uses of such methods.

Uses of SA
----------

As indicated above, there are several uses for SA. We identify four uses
which are outlined here and discussed further when we present particular
SA methods.

-  *Understanding.* Techniques to show how the output :math:`f(x)` behaves
   as we vary one or more of the inputs :math:`x` are an aid to
   understanding :math:`f`. They can act as a 'face validity' check, in the
   sense that if the simulator is responding in unexpected ways to
   changes in its inputs, or if some inputs appear to have unexpectedly
   strong influences, then perhaps there are errors in the mathematical
   model or in its software implementation.

-  *Dimension reduction.* If SA can show that certain inputs have
   negligible effect on the output, then this offers the prospect of
   simplifying :math:`f` by fixing those inputs. Whilst this does not
   usually simplify the simulator in the sense of making it quicker or
   easier to evaluate :math:`f(x)` at any desired :math:`x`, it reduces
   the dimensionality of the input space. This makes it easier to
   understand and use.

-  *Analysing output uncertainty.* Where there is uncertainty about
   inputs, there is uncertainty about the resulting outputs :math:`f(x)`;
   quantifying the output uncertainty is the role of :ref:`uncertainty
   analysis<DefUncertaintyAnalysis>`, but there is often
   interest in knowing how much of the overall output uncertainty can be
   attributed to uncertainty in particular inputs or groups of inputs.
   In particular, effort to reduce output uncertainty can be expended
   most efficiently if it is focused on those inputs that are
   influencing the output most strongly.

-  *Analysing decision uncertainty.* More generally, uncertainty about
   simulator outputs is most relevant when those outputs are to be used
   in the making of some decision (such as using a climate simulator in
   setting targets for the reduction of carbon dioxide emissions). Again
   it is often useful to quantify how much of the overall decision
   uncertainty is due to uncertainty in particular inputs. The
   prioritising of research effort to reduce uncertainty depends on the
   influence of inputs, their current uncertainty and the nature of the
   decision.

Probabilistic SA
----------------

Although a variety of approaches to SA have been discussed and used by
people who study and use simulators, there is a strong preference in
:ref:`MUCM<DefMUCM>` for a methodology known as probabilistic SA.
Other approaches and the reasons for preferring probabilistic SA are
discussed in page
:ref:`DiscWhyProbabilisticSA<DiscWhyProbabilisticSA>`.

Probabilistic SA requires a probability distribution to be assigned to
the simulator inputs. We first present some notation and will then
discuss the interpretation of the probability distribution and the
relevant measures of sensitivity in probabilistic SA.

Notation
~~~~~~~~

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
if :math:`J=\{2,6\}` then :math:`x_J=x_{\{2,6\}}` comprises inputs 2 and 6.
Finally, :math:`x_{-j}` will denote the whole of the inputs :math:`x`
*except* :math:`x_j`, and similarly :math:`x_{-J}` will be the set of all
inputs except those in :math:`x_J`.

We denote the probability distribution over the inputs :math:`x` by
:math:`\omega`. Formally, :math:`\omega(x)` is the joint probability
density function for all the inputs. The marginal density function for
input :math:`x_j` is denoted by :math:`\omega_j(x_j)`, while for the group of
inputs :math:`x_J` the density function is :math:`\omega_J(x_J)`. The
conditional distribution for :math:`x_{-J}` given :math:`x_J` is
:math:`\omega_{-J|J}(x_{-J}\,|\,x_J)`. Note, however, that it is common for
the distribution :math:`\strut\omega` to be such that the various inputs
are statistically independent. In this case the conditional distribution
:math:`\omega_{-J|J}` does not depend on :math:`x_J` and is identical to the
marginal distribution :math:`\omega_{-J}`.

Random :math:`X`
~~~~~~~~~~~~~~~~~~~~

Note that by assigning probability distributions to the inputs in
probabilistic SA we formally treat those inputs as random variables.
Notationally, it is conventional in statistics to denote random
variables by capital letters, and this distinction is useful also in
probabilistic SA. Thus, the symbol :math:`X` denotes the set of
inputs when regarded as random (i.e. uncertain), while :math:`x`
continues to denote a particular set of input values. Similarly,
:math:`X_J` represents those random inputs with subscripts in the set
:math:`J`, while :math:`x_J` denotes an actual value for those inputs.

It is most natural to think of :math:`X` as a random variable in the
context of the third and fourth uses of SA, as listed above. When there
is genuine uncertainty about the proper values to assign to inputs in
order to obtain the output(s) of interest, then :math:`X` can
indeed be interpreted as random, and :math:`\omega(x)` is then the
probability density function describing the relative probabilities of
different possible values :math:`x` for :math:`X`. SA then
involves trying to understand the role of uncertainty about the various
inputs in the induced uncertainty concerning the outputs :math:`f(X)` or
concerning a decision based on these outputs.

Interpretation of :math:`\omega(x)` as a weight function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

However, in the context of other uses of SA it may be less natural to
think of :math:`X` as random. When our objective is to gain
understanding of the simulator's behaviour or to identify inputs that
are more or less redundant, it is not necessary to regard the inputs as
uncertain. It is, nevertheless, important to think about the range of
input values over which we wish to achieve the desired understanding or
dimension reduction. In this case, :math:`\omega(x)` can simply define that
range by being zero for any :math:`x` outside the range. Within the
range of interest, we may regard :math:`\omega(x)` as a weight function.
Whilst we might normally give equal weight to all points in the range,
for some purposes it may be appropriate to give more weight to some
points than to others.

Whether we regard :math:`\omega(x)` as defining a probability distribution
or simply as a weight function, it allows us to average over the region
of interest to define measures of sensitivity.

Probabilistic SA methods
~~~~~~~~~~~~~~~~~~~~~~~~

We have seen that different uses of SA may suggest not only different
ways of interpreting the :math:`\omega(x)` function but also may demand
different kinds of SA measures. However, there are similarities and
connections between the various measures, particularly between measures
used for understanding, dimension reduction and analysing output
uncertainty. These are discussed together in page
:ref:`DiscVarianceBasedSA<DiscVarianceBasedSA>` (with some technical
details in page
:ref:`DiscVarianceBasedSATheory<DiscVarianceBasedSATheory>`). Ways to
use these variance based SA measures for output uncertainty are
considered in page
:ref:`DiscSensitivityAndOutputUncertainty<DiscSensitivityAndOutputUncertainty>`.
Their usage for understanding and dimension reduction is discussed in
page
:ref:`DiscSensitivityAndSimplification<DiscSensitivityAndSimplification>`.

Measures specific to analysing decision uncertainty are presented in the
discussion page :ref:`DiscDecisionBasedSA<DiscDecisionBasedSA>`,
where the variance based measures are also shown to arise as a special
case, while the discussion page
:ref:`DiscSensitivityAndDecision<DiscSensitivityAndDecision>`
considers the practical use of these measures for decision uncertainty.

SA in the toolkit
-----------------

All SA measures concern the relationship between a simulator's inputs
and outputs. They generally depend on the whole function :math:`f` and
implicitly suppose that we know :math:`f(x)` for all :math:`x`. In
practice, we can only run the simulator at a limited number of input
configurations, and as a result any computation of SA measures must be
subject to computation error. The conventional Monte Carlo approach, for
instance, involves randomly sampling input values and then running the
simulator at each sampled :math:`x`. Its accuracy can be quantified
statistically and reduces as the sample size increases. For large and
complex simulators, Monte Carlo may be infeasible because of the amount
of computation required. One of the motivations for the MUCM approach is
that tasks such as SA can be performed much more efficiently, first
building an emulator using a modest number of simulator runs, and then
computing the SA measures using the emulator. The computation error is
then quantified in terms of :ref:`code
uncertainty<DefCodeUncertainty>`.

SA is one of the tasks that we aim to cover in each of the main threads
(i.e. core threads, variant threads and generic threads). Each of these
threads describes the modelling and building of an emulator for a
particular kind of simulator, and each explains how to use that emulator
to carry out tasks associated with that simulator. So wherever the
procedure for computing SA measures has been worked out for a particular
thread, that procedure will be described in that thread. See the page
:ref:`DiscToolkitSensitivityAnalysis<DiscToolkitSensitivityAnalysis>`
for a discussion of which procedures are available in which threads.

Additional comments
-------------------

Note that although SA is usually presented as being concerned with the
relationship between simulator inputs and outputs, the principal purpose
of a simulator is to represent a particular real-world phenomenon: it is
often built to explore how that real phenomenon behaves and some or all
of its inputs represent quantities in the real world. The simulator
output :math:`f(x)` is intended to predict the value of some aspect of the
real phenomenon when the corresponding real quantities take values
:math:`x`. So in principle we may wish to consider SA in which the
simulator is replaced by reality. This may be developed in a later
version of this toolkit.

In some application areas, the term "probabilistic sensitivity analysis"
is used for what we call uncertainty analysis.

References
----------

Three books are extremely useful guides to the uses of SA in practice,
and for non-MUCM methods for computing some of the most important
measures.

Saltelli, A., Chan, K. and Scott, E. M. (eds.) (2000). `Sensitivity
Analysis <http://eu.wiley.com/WileyCDA/WileyTitle/productCd-0471998923>`__.
Wiley.

Saltelli, A., Tarantola, S., Campolongo, F. and Ratto, M. (2004).
`Sensitivity Analysis in Practice: A guide to assessing scientific
models <http://eu.wiley.com/WileyCDA/WileyTitle/productCd-0470870931>`__.
Wiley.

Saltelli, A., Ratto, M., Andres, T., Campolongo, F., Cariboni, J.,
Gatelli, D., Saisana, M. and Tarantola, S. (2008). `Global Sensitivity
Analysis: The
primer <http://eu.wiley.com/WileyCDA/WileyTitle/productCd-0470059974.html>`__.
Wiley.

MUCM methods for computing SA measures are based on using emulators. The
basic theory was presented in

Oakley, J. E. and O'Hagan, A. (2004). Probabilistic sensitivity analysis
of complex models: a Bayesian approach. *Journal of the Royal
Statistical Society* *Series* *B* 66, 751-769.
(`Online <http://www3.interscience.wiley.com/journal/118808484/abstract>`__)

Whilst the above references are all useful for background information,
the toolkit pages present a methodology for efficient SA using emulators
that is not published elsewhere in such a comprehensive way.
