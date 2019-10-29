.. _DiscSensitivityAndSimplification:

Discussion: Sensitivity analysis measures for simplification
============================================================

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
appropriate to two of those uses - understanding the way that a
:ref:`simulator's<DefSimulator>` output responds to changes in its
inputs, and identifying inputs with little effect on the output with a
view to simplifying either the simulator or a corresponding
:ref:`emulator<DefEmulator>`. We refer to both of these uses as
simplification because understanding is generally best achieved by
looking for simple descriptions.

In probabilistic SA, we assign a function :math:`\omega(x)` that is
formally treated as a probability density function for the inputs. The
interpretation and specification of :math:`\omega(x)` will be considered as
part of the discussion below.

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
understanding and dimension reduction. We also consider the related
topic of validation/criticism of a simulator. All of the SA measures
recommended here are defined and discussed in page
:ref:`DiscVarianceBasedSA<DiscVarianceBasedSA>`.

We assume that interest focuses on a single simulator output.
Understanding of several outputs is best done by exploring each output
separately. However, some remarks on dimension reduction for multiple
outputs may be found in the "Additional comments" section.

Specifying :math:`\omega(x)`
~~~~~~~~~~~~~~~~~~~~~~~~~

Although :math:`\omega(x)` is treated formally as a probability density
function in probabilistic SA, for the purposes of simplification it is
often simply a weight function specifying the relative importance of
different :math:`\strut x` values. Indeed, it is often a uniform density
giving equal importance for all points in a subset of the possible
:math:`\strut x` space.

 Weight function
^^^^^^^^^^^^^^^

Note that we generally wish to study the behaviour of the simulator over
a subset of the whole space of possible inputs, because we are
interested in applications of that simulator to model a particular
instance of the real-world processes being simulated. For example, a
simulator of rainfall runoff through a river catchment has inputs which
range over values such that it can predict flows for a wide variety of
real river catchments, but in a given situation we are interested in its
behaviour for a particular catchment. For this particular instance, the
inputs will have a narrower range to represent the likely values in that
catchment.

Suppose first that we wish to weight all points in the subset equally,
and so use a uniform weight function. If the range of values of input
:math:`x_j` is specified as :math:`a_j \\le x_j \\le b_j`, for
:math:`j=1,2,\ldots,p`, then the area of the subset of interest is

:math:`A=\prod_{j=1}^p (b_j -a_j)\,,`

and the uniform density for values of :math:`\strut x` in this subset is

:math:`\omega(x) = 1/A\,,`

with :math:`\omega(x)=0` for :math:`\strut x` outside that subset. [Since
:math:`\omega(x)` is treated as a probability density function, the total
weight over the subset must be 1, hence the weight at each point is
:math:`1/A`.] Note that in this case the inputs are independent, which
leads to some simplification of the sensitivity measures presented in
:ref:`DiscVarianceBasedSA<DiscVarianceBasedSA>`.

If we are more interested in some parts of the input space than in
others, then we could use a non-uniform weight function.

 Probability density
^^^^^^^^^^^^^^^^^^^

We can also consider situations where it would be appropriate to specify
:math:`\omega(x)` as a genuine probability density function. One is where
there is uncertainty about the proper input values to use, and our wish
for understanding or dimension reduction is in the context of that
uncertainty. In this case, we may also want to carry out SA for
analysing output uncertainty or decision uncertainty, but understanding
and/or dimension reduction are useful preliminary explorations. The
appropriate choice of :math:`\omega(x)` in this case is the probability
distribution that represents the uncertainty about :math:`\strut x`; see
the discussion page on sensitivity measures for output uncertainty
(:ref:`DiscSensitivityAndOutputUncertainty<DiscSensitivityAndOutputUncertainty>`).

Another case can be identified by considering again the example of a
simulator of rainfall runoff in a river catchment. For applications in a
given catchment, over a period of time, inputs defining the rainfall
incidence and amounts of water already in the catchment will vary. A
probability distribution might be chosen to represent the relative
prevalence of different conditions in the catchment.

Dimensionality reduction
~~~~~~~~~~~~~~~~~~~~~~~~

Simulators often have a large number of inputs. Whilst these are all
considered relevant and are all expected to have some impact on the
outputs, many will have only minor effects. This is particularly true
when we are interested in using the simulator for a specific application
where the inputs have restricted ranges. The response of the simulator
outputs over the input subspace of interest may be dominated by only a
small number of inputs. The goal of dimensionality reduction is to
separate the important inputs from those to which the output is
relatively insensitive.

With a large number of inputs, it becomes impractical to consider
varying all the inputs together to perform a full SA. In practice,
various simplified :ref:`screening<DefScreening>` procedures are
generally used to reduce the set of inputs to a more manageable number.
A discussion of screening methods is available at the topic thread on
screening (:ref:`ThreadTopicScreening<ThreadTopicScreening>`).

Formal SA techniques are typically use to explore more carefully the
inputs which cannot be eliminated by simple screening tools, with a view
to finding the small number of most influential inputs. It is in this
final exploratory and confirmatory phase of dimension reduction that SA
methods are most useful.

The SA measure that is best for identifying an input that has little
effect on the simulator output is its total sensitivity index :math:`T_j`.
This is the proportion of uncertainty that would remain if all the
remaining inputs :math:`x_{-j}` were known. In the case of independent
inputs, any :math:`x_j` for which :math:`T_j` is less than, say, 0.01 (or 1%)
could be considered to have only a very small effect on the output. If
inputs are not independent, small :math:`T_j` does not necessarily imply
that :math:`x_j` has little effect, and it is important to check also that
its sensitivity index :math:`S_j` is small.

Understanding
~~~~~~~~~~~~~

Understanding is a less well defined goal than dimension reduction, and
a variety of measures may be useful. Dimension reduction is a good
start, since much simplification is achieved by reducing the number of
inputs to be considered because their effects are appreciable.

Having identified the important inputs, another very useful step is to
split these into groups with only minimal interaction between groups.
Then the response of the output can be considered as a sum of effects
due to the separate groups. A group of inputs :math:`x_J` has negligible
interaction with the remaining inputs :math:`x_{-J}` if their group
sensitivity measure :math:`S_J` is close to their total sensitivity measure
:math:`T_J`.

When the important inputs have been subdivided in this way,
understanding is achieved by looking at the behaviour of the output in
response to each group separately. For a group :math:`x_J`, the most useful
SA measure now is its mean effect :math:`M_J(x_J)`. If the group comprises
just a single input, then its mean effect can simply be plotted to
provide a visual impression of how that input affects the output. This
can be supplemented by looking at regression components. For instance,
if the effect of this input is nearly linear then its linear variance
measure :math:`V^L_j` will be close to its variance :math:`V_j`, and the slope
of the linear regression line will provide a simple description of how
the output responds to this input.

If a group is not single input, and it cannot be split further, then
there are appreciable interactions between inputs within the group. We
can still examine the mean effect :math:`M_j(x_j)` of each individual input
in the group, but we also need to consider the joint effect. Plotting
the two-input mean effect :math:`M_{\{j,j'\}}(x_{\{j,j'\}})` of a pair of
inputs as a contour plot can give a good visual impression of their
joint effect. Similarly, we could view a contour plot of their
interaction effect :math:`I_{\{j,j'\}}(x_{\{j,j'\}})`, but for a group with
more than two inputs it is difficult to get understanding beyond
pairwise effects. Careful exploration of regression terms (analogously
to the conventional statistical method of stepwise regression) may
succeed in identifying a small number of dominant terms - some
discussion of this may be provided in a future version of the toolkit.

Criticism/validation
~~~~~~~~~~~~~~~~~~~~

A simulator is a model of some real-world process. A wish to understand
or simplify a simulator may be motivated by wanting to understand the
real-world process. It is obviously unwise to use the simulator as a
surrogate for reality in this way unless it is known that the simulator
is a good representation of reality. Techniques for modelling and
validating the relationship between a simulator and reality will be
introduced in a later version of the toolkit.

However, there is a useful, almost opposite use for understanding a
simulator. We will often have some clear qualitative understanding or
intuition about how reality behaves, and by examining the behaviour of
the simulator we can check whether it conforms to such
understanding/intuition. If, for instance, we expect increasing :math:`x_1`
to cause the real-world value :math:`y(x)` to increase, then we will be
worried if we find that :math:`f(x)` generally decreases when :math:`x_1`
increases. Similarly, we can check whether the most active inputs, the
presence or absence of interactions and the nature of nonlinearity in
the simulator agrees with how we expect reality to behave.

If there is a mismatch between behaviour of the simulator, as revealed
by SA techniques, and our beliefs about reality then this suggests that
one of them is faulty. Either reality does not actually behave as we
think it should, or else the simulator does not capture reality
adequately. We cannot expect a simulation to be a perfectly accurate
representation of reality, but if we find that its qualitative behaviour
is wrong then this often suggests the kind of modification that might be
made to improve the simulator.

Additional comments
-------------------

Transformation of the output may make it easier to find simplification.
If, for instance, :math:`f(x)` must be positive but can vary through two or
more orders of magnitude (a factor of 100 or more) then working with its
logarithm, :math:`\log f(x)`, is worth considering. There may be fewer
important inputs, fewer interactions and less nonlinearity on the
logarithmic scale.

In the case of multiple outputs, if dimension reduction is applied to
each output separately this will typically lead to a different set of
most active inputs being retained for different outputs. If there is a
wish to simplify the simulator by eliminating the same set of inputs for
all outputs, this can again be done by considering the total sensitivity
index :math::ref:`T_j`, but now this is a matrix (see
`DiscVarianceBasedSA<DiscVarianceBasedSA>`). An input should be
considered to have a suitably small effect if *all* the elements of
:math:`T_j` are small.
