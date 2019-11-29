.. _DiscIterativeRefocussing:

Discussion: Iterative Refocussing
=================================

Description and Background
--------------------------

As introduced in
:ref:`ThreadGenericHistoryMatching<ThreadGenericHistoryMatching>`, an
integral part of the :ref:`history matching<DefHistoryMatching>`
process is the iterative reduction of input space by use of
implausibility cutoffs, a technique known as refocussing. This page
discusses this strategy in more detail, and describes why iterative
refocussing is so useful. The notation used is the same as that defined
in :ref:`ThreadGenericHistoryMatching<ThreadGenericHistoryMatching>`.
Here we use the term model synonymously with the term
:ref:`simulator<DefSimulator>`.

Discussion
----------

We are attempting to identify the set of acceptable inputs :math:`\mathcal{X}`,
and we do this by reducing the input space in a series
of iterations or *waves*. We denote the original input space as
:math:`\mathcal{X}_0`.

Refocussing: Motivation
~~~~~~~~~~~~~~~~~~~~~~~

Consider the univariate implausibility measure :math:`I_{(i)}(x)`
corresponding to the :math:`i`th output of a multi-output
function :math:`f(x)`, introduced in
:ref:`AltImplausibilityMeasures<AltImplausibilityMeasure>`:

.. math::
   I^2_{(i)}(x) = \frac{ ({\rm E}[f_i(x)] - z_i )^2}{{\rm
   Var}[f_i(x)] + {\rm Var}[d_i] + {\rm Var}[e_i]}

If we impose a cutoff :math:`c` on the original input space
:math:`\mathcal{X}_0` such that

.. math::
   I_{i}(x) \le c

then this defines a new volume of input space that we refer to as
non-implausible after wave 1 and denote :math:`\mathcal{X}_1`.
In the first wave of the analysis :math:`\mathcal{X}_1` will be
substantially larger than the set of interest :math:`\mathcal{X}`,
as it will contain many values of :math:`x` that only
satisfy the implausibility cutoff because of a substantial emulator
variance :math:`{\rm Var}[f_i(x)]`. If the emulator was
sufficiently accurate over the whole of the input space that
:math:`{\rm Var}[f_i(x)]` was small compared to the model discrepancy and
the observational error variances :math:`{\rm Var}[d_i]` and
:math:`{\rm Var}[e_i]`, then the non-implausible volume defined
by :math:`\mathcal{X}_1` would be comparable to :math:`\mathcal{X}`
and the history match would be complete. However, to
construct such an accurate emulator would, in most applications, require
an infeasible number of runs of the model. Even if such a large number
of runs were possible, it would be an extremely inefficient method: we
do not need the emulator to be highly accurate in regions of the input
space where the outputs of the model are clearly very different from the
observed data.

This is the main motivation for the iterative approach: in each wave we
design a set of runs only over the current non-implausible volume,
emulate using these runs, calculate the implausibility measure and
impose a cutoff to define a new (smaller) non-implausible volume. This
is referred to as refocusing.

Note that although we use implausibility cutoffs to define a
non-implausible sub-space :math:`\mathcal{X}_1`, we only have an
implicit expression for this sub-space and cannot generate points from
it directly. We can however generate a large space filling design of
points over :math:`\mathcal{X}_0` and discard any points that do
not satisfy the implausibility cutoff and which are hence not in
:math:`\mathcal{X}_1`. We use this technique in the iterative
method discussed below.

Iterative Refocussing
~~~~~~~~~~~~~~~~~~~~~

This method, which has been found to work in practice, can be summarised
as follows. At each iteration or wave:

#. A design for a set of runs over the current non-implausible volume
   :math:`\mathcal{X}_j` is created. This is done by first
   generating a very large :ref:`latin hypercube<ProcLHC>`, or other
   space filling design, over the full input space. Each of these design
   points are then tested to see if they satisfy the implausibility
   cutoffs in every one of the previous waves: if they do then they are
   included in the next set of simulator runs.
#. These runs (including any non-implausible runs from previous waves)
   are used to construct a more accurate emulator, which is defined only
   over the current non-implausible volume :math:`\mathcal{X}_j`,
   as it has been constructed purely from runs that are members of
   :math:`\mathcal{X}_j`.
#. The implausibility measures are then reconstructed over
   :math:`\mathcal{X}_j`, using the new, more accurate emulator.
#. Cutoffs are imposed on the implausibility measures and this defines a
   new, smaller non-implausible volume :math:`\mathcal{X}_{j+1}`
   which should satisfy :math:`\mathcal{X} \subset
   \mathcal{X}_{j+1} \subset \mathcal{X}_{j}`.
#. Unless the stopping criteria have been reached, that is to say,
   unless the emulator variance :math:`{\rm Var}[f(x)]` is found
   to be everywhere far smaller than the combined model discrepancy
   variance and observational errors, or unless computational resources
   have been exhausted, return to step 1.

If it becomes clear from say, projected implausibility plots (Vernon,
2010), that :math:`\mathcal{X}_j` can be enclosed in some
hypercube that is smaller than the original input space :math:`\mathcal{X}_0`,
then we would do this, and use the smaller hypercube
to generate the subsequent large latin hypercubes required in step 1.

As we progress through each iteration the emulator at each wave will
become more and more accurate, but will only be defined over the
previous non-implausible volume given in the previous wave. The increase
in accuracy will allow further reduction of the input space. The process
should terminate when certain stopping criteria are achieved as
discussed below. It is of course possible (even probable) that
computational resources will be exhausted before the stopping criteria
are satisfied. In this case analysis of the size, location and shape of
the final, non-implausible region should be performed (which is often of
major interest to the modeller), possiblywith a subsequent probabilistic
calibration if required.

Improved Emulator Accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~

We usually see significant improvement in emulator accuracy for the
current wave, as compared to previous waves. That is to say, we see the
emulator variance :math:`{\rm Var}[f(x)]` generally decreasing
with increasing wave number, while the emulator itself maintains
satisfactory performance (as judged by emulator
:ref:`diagnostics<ProcValidateCoreGP>`).

We expect this improvement in the accuracy of the emulator for several
reasons. As we have reduced the size of the input space and have
effectively zoomed in on a smaller part of the function, we expect the
function to be smoother and to be more easily approximated by the
regression terms in the emulator (see
:ref:`ThreadCoreBL<ThreadCoreBL>` for details), leading to a better
fit, with reduced residual variances for all outputs. Due to the
increased density of runs over :math:`\mathcal{X}_j` compared to
previous waves, the :ref:`Gaussian process<DefGP>` term in the
emulator, :math:`w(x)`, (which is updated by the new runs) will
be more accurate and have lower variance as the general point :math:`x`
will be on average closer to known evaluation inputs.

Another major improvement in the emulators comes from identifying a
larger set of :ref:`active inputs<DefActiveInput>`. Cutting down the
input space also means that the ranges of the function outputs are
reduced. Dominant inputs that previously had large effects on the
outputs have likewise been constrained, and their effects lessened. This
results in it being easier to identify more active inputs that were
previously masked by a small number of dominant variables. Increasing
the number of active inputs allows more of the function's structure to
be modelled by the regression terms, and has the effect of reducing the
relative size of any :ref:`nugget term<DefNugget>`.

As the input space is reduced, it not only becomes easier to accurately
emulate existing outputs but also to emulate outputs that were not
considered in previous waves. Outputs may not have been considered
previously because they were either difficult to emulate, or because
they were not informative regarding the input space.

Stopping Criteria
~~~~~~~~~~~~~~~~~

The above iterative method should be terminated either when
computational resources are exhausted (a likely outcome for slow
models), or when it is thought that the current non-implausible volume
:math:`\mathcal{X}_j` is sufficiently close to the acceptable
set of inputs :math:`\mathcal{X}`. An obvious case is where we
find the set :math:`\mathcal{X}_j` to be empty: we then conclude
that :math:`\mathcal{X}` is empty and that the model cannot
provide acceptable matches to the outputs. See the end of
:ref:`ThreadGenericHistoryMatching<ThreadGenericHistoryMatching>` for
further discussion of this situation.

A simple sign that we are approaching :math:`\mathcal{X}` is
when the new non-implausible region :math:`\mathcal{X}_{j+1}` is
of comparable size or volume to the previous :math:`\mathcal{X}_j`.
A more advanced stopping criteria involves analysing the emulator
variance :math:`{\rm Var}[f(x)]` over the current volume, and
comparing it to the other uncertainties present, namely the model
discrepancy variance :math:`{\rm Var}[d]` and the observational
errors variance :math:`{\rm Var}[e]`. If :math:`{\rm
Var}[f_i(x)]` is significantly smaller than :math:`{\rm Var}[d_i]`
and :math:`{\rm Var}[e_i]` over all of the current volume,
for all :math:`i`, then even if we proceed with more iterations and
improve the emulator accuracy further, there will not be a significant
reduction in size of the non-implausible volume.

Once the process is terminated for either of the above reasons, various
visualisation techniques can be employed to help analyse the location,
size and structure of the remaining non-implausible volume.

Additional Comments
-------------------

Consider the case where in the :math:`(j-1)`\th wave, we have deemed
certain inputs to be borderline implausible; that is, they lie close to
the non-implausible region :math:`\mathcal{X}_j` and have
implausibilities just higher than the cutoff. We can use the new
:math:`j`\th wave emulator to effectively overrule this decision, if
the new :math:`j`\th wave implausibility measure suggests these
inputs are actually non-implausible at the :math:`j`\th wave. We can
only do this for borderline points that are close to
:math:`\mathcal{X}_j`, as further away from :math:`\mathcal{X}_j`
the new emulator has no validity. The need for such
overruling is mitigated by using conservative implausibility cutoffs.

This iterative refocussing strategy is very powerful and has been used
to successfully history match several complex models, e.g. oil reservoir
models (Craig, 1997) and Galaxy formation models (Bower 2009, Vernon
2010). Also see :ref:`Exam1DHistoryMatch<Exam1DHistoryMatch>` for a
simple example which highlights these ideas.

References
----------

Vernon, I., Goldstein, M., and Bower, R. (2010), "Galaxy Formation: a
Bayesian Uncertainty Analysis," MUCM Technical Report 10/03.

Craig, P. S., Goldstein, M., Seheult, A. H., and Smith, J. A. (1997).
"Pressure matching for hydrocarbon reservoirs: a case study in the use
of Bayes linear strategies for large computer experiments." In Gatsonis,
C., Hodges, J. S., Kass, R. E., McCulloch, R., Rossi, P., and
Singpurwalla, N. D. (eds.), Case Studies in Bayesian Statistics, volume
3, 36–93. New York: Springer-Verlag.

Bower, R., Vernon, I., Goldstein, M., et al. (2009), "The Parameter
Space of Galaxy Formation," to appear in MNRAS; MUCM Technical Report
10/02.
