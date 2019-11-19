.. _DiscImplausibilityCutoff:

Discussion: Implausibility Cutoffs
==================================

Description and Background
--------------------------

As introduced in
:ref:`ThreadGenericHistoryMatching<ThreadGenericHistoryMatching>`, an
integral part of the :ref:`history matching<DefHistoryMatching>`
process is the imposing of cutoffs on the various :ref:`implausibility
measures<DefImplausibilityMeasure>` introduced in
:ref:`AltImplausibilityMeasure<AltImplausibilityMeasure>`. This page
discusses the relevant considerations involved in choosing the cutoffs.
The notation used is the same as that defined in
:ref:`ThreadGenericHistoryMatching<ThreadGenericHistoryMatching>`.
Here we use the term model synonymously with the term
:ref:`simulator<DefSimulator>`.

Discussion
----------

History matching attempts to identify the set :math:`\mathcal{X}`
of all inputs that would give rise to acceptable matches between
model outputs and observed data. This is achieved by imposing cutoffs on
implausibility measures in order to discard inputs :math:`x` that
are highly unlikely to belong to :math:`\mathcal{X}`. The
specific values used for these cutoffs is therefore important, and
several factors must be considered before choosing them.

Univariate Cutoffs
~~~~~~~~~~~~~~~~~~

Consider the univariate implausibility measure :math:`I_{(i)}(x)`
corresponding to the :math:`i`th output of a multi-output
function :math:`f(x)`, introduced in
:ref:`AltImplausibilityMeasure<AltImplausibilityMeasure>`:

.. math::
   I^2_{(i)}(x) = \frac{ ({\rm E}[f_i(x)] - z_i )^2}{ {\rm
   Var}[ {\rm E}[f_i(x)] -z_i] } = \frac{ ({\rm E}[f_i(x)] - z_i )^2}{{\rm
   Var}[f_i(x)] + {\rm Var}[d_i] + {\rm Var}[e_i]}

where the second equality follows from the definition of the best input
approach (see :ref:`DiscBestInput<DiscBestInput>` for details). We
are to impose a cutoff :math:`c` such that values of :math:`x` that satisfy:

.. math::
   I_{(i)}(x) \le c

are to be analysed further, and all other values of :math:`x` are
discarded. This defines a new sub-volume of the input space that we
refer to as the non-implausible volume. Determining a reasonable value
for the cutoff :math:`c` can be achieved using certain
unimodality arguments, which are employed as follows.

Regarding the size of the individual univariate implausibility measure
:math:`I_{(i)}(x)`, we again consider :math:`x` as a
candidate for the best input :math:`x^+`. If we then make the
fairly weak assumption that the appropriate distribution of
:math:`({\rm E}[f_i(x)]-z)`, with :math:`x=x^+ `, is both unimodal and
continuous, then we can use the :math:`3\sigma` rule (Pukelsheim,
1994) which implies quite generally that :math:`I_{(i)}(x) \le 3`
with probability greater than 0.95, even if the distribution is
heavily asymmetric. A value of :math:`I_{(i)}(x)` greater than a
cutoff of :math:`c=3` would suggest that the input :math:`x`
could be discarded.

Consideration of the fraction of input space that is removed for
different choices of :math:`c` may alter this value. For example,
if we find that we can remove a sufficiently large percentage of the
input space with a more conservative choice of say :math:`c=4`,
then we may adopt this value instead. In addition, we may also perform
various diagnostics to check that we are not discarding any acceptable
runs (Vernon, 2010).

Multivariate Cutoffs
~~~~~~~~~~~~~~~~~~~~

We can use the above univariate cutoff :math:`c` as a guide to
determine an equivalent cutoff :math:`c_M` for the maximum
implausibility measure such that we discard all :math:`x` that do
not satisfy

.. math::
   I_{M}(x) \le c_M

If we are dealing with a small number of highly correlated outputs, then
this will be similar to the univariate case and we can use Pukelsheim's
3 sigma rule and choose values for :math:`c_M` that are similar
to or slightly larger than :math:`c=3` (e.g. :math:`c_M =
3.2, 3.5` say). If there are a large number of outputs, and we
believe them to be mostly uncorrelated then a higher value must be
chosen (e.g. :math:`c_M = 4`, 4.5 or 5 say) depending on the fraction
of space cut out. If we want to be
more precise, we can make further assumptions as to the shape of the
individual distributions that are used in the implausibility measures:
for example we can assume they are normally distributed, and look up
suitable tables for the maximum of a collection of independent or even
correlated normals at certain conservative significance levels (we would
recommend 0.99 or higher). Similar considerations can be used to
generate sensible cutoffs for the second and third maximum
implausibility measures. See
:ref:`DiscInformalAssessMD<DiscInformalAssessMD>` for similar
discussions regarding cutoffs, and Vernon, 2010 for examples of their
use.

Consider the full multivariate implausibility measure :math:`I_{MV}(x)`
introduced in
:ref:`AltImplausibilityMeasure<AltImplausibilityMeasure>`:

.. math::
   I_{MV}(x) = (z - {\rm E}[f(x)])^T ({\rm Var}[f(x)] +
   {\rm Var}[d] + {\rm Var}[e])^{-1} (z - {\rm E}[f(x)])

Choosing a suitable cutoff :math:`c_{MV}` for :math:`I_{MV}(x)`
is more complicated. As a simple heuristic, we might
choose to compare :math:`I_{MV}(x)` with the upper critical value
of a :math:`\chi^2`-distribution with degrees-of-freedom equal
to the number of outputs considered. This should then be combined with
considerations of percentage space cutout along with diagnostics as
discussed above.

Additional Comments
-------------------

In order to link the different cutoffs :math:`c`, :math:`c_M`
and :math:`c_{MV}` rigorously, we would of course have to
make full multivariate distributional assumptions over each of the
relevant output quantities. This is a level of detail that is in many
cases not necessary, provided relatively conservative choices for each
of the cutoffs are made.

The history matching process involves applying the above implausibility
cutoffs iteratively, as is described in
:ref:`ThreadGenericHistoryMatching<ThreadGenericHistoryMatching>` and
discussed in detail in
:ref:`DiscIterativeRefocussing<DiscIterativeRefocussing>`. The
implausibility measures are simple and very fast to evaluate (as they
only rely on evaluations of the emulator), and hence the application of
the cutoffs is tractable even for testing large numbers of input points.

References
----------

Pukelsheim, F. (1994). “The three sigma rule.” *The American
Statistician*, 48: 88–91.

Vernon, I., Goldstein, M., and Bower, R. (2010), “Galaxy Formation: a
Bayesian Uncertainty Analysis,” MUCM Technical Report 10/03
