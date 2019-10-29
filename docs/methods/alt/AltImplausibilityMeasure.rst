.. _AltImplausibilityMeasure:

Alternatives: Implausibility Measures
=====================================

Description and Background
--------------------------

As introduced in
:ref:`ThreadGenericHistoryMatching<ThreadGenericHistoryMatching>`, an
integral part of the :ref:`history matching<DefHistoryMatching>`
process is the use of :ref:`implausibility
measures<DefImplausibilityMeasure>`. This page discusses
implausibility measures in more detail and defines several different
types, highlighting their strengths and weaknesses. The notation used is
the same as that defined in
:ref:`ThreadGenericHistoryMatching<ThreadGenericHistoryMatching>`.
Here we use the term model synonymously with the term
:ref:`simulator<DefSimulator>`.

Discussion
----------

Univariate Implausibility Measures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our approach to history matching is based on the assessment of certain
implausibility measures which we now describe. An implausibility measure
is a function defined over the input space which, when large, suggests
that the match between model and system would exceed a stated tolerance
in terms of the relevant uncertainties that are present, such as the
:ref:`model discrepancy<DefModelDiscrepancy>` :math:`\strut{ d }` and
the observational errors :math::ref:`\strut{ e }`; see
`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`
for more details. We may build up this concept of an acceptable match
for a single output :math:`\strut{f_i(x)}` as follows. For a given choice
of input :math::ref:`\strut{x}`, which we now consider as a candidate for the
`best input<DefBestInput>` :math:`\strut{ x^+ }`, we would ideally
like to assess whether the output :math:`\strut{f_i(x)}` differs from the
system value :math:`\strut{y_i}` by more than the tolerance that we allow
in terms of model discrepancy. Therefore, we would assess the
standardised distance

:math:`\strut{ \\frac{(y_i - f_i(x))^2}{{\rm Var}[d_i]} }`

In practice, we cannot observe the system :math:`\strut{y_i}` directly and
so we must compare :math:`\strut{f_i(x)}` with the observation
:math:`\strut{z_i}`, introducing measurement error :math:`\strut{e_i}` with
corresponding standardised distance

:math:`\strut{ \\frac{(z_i - f_i(x))^2}{{\rm Var}[d_i] + {\rm Var}[e_i]} }`

However, for most values of :math:`\strut{x}`, we are not able to evaluate
:math:`\strut{f(x)}` as the model takes a significant time to run, so
instead we use the emulator and compare :math:`\strut{z_i}` with
:math:`\strut{{\rm E}[f_i(x)]}`. Therefore, the implausibility function or
measure is defined as

:math:`\strut{ I^2_{(i)}(x) = \\frac{ ({\rm E}[f_i(x)] - z_i )^2}{{\rm
Var}[{\rm E}[f_i(x)]-z_i ] } = \\frac{ ({\rm E}[f_i(x)] - z_i )^2}{{\rm
Var}[f_i(x)] +{\rm Var}[d_i] + {\rm Var}[e_i]} }`

where the second equality results from the fact that, here,
:math:`\strut{x}` is implicitly being considered as a candidate for the
best input :math::ref:`\strut{ x^+ }`, and hence the independence assumptions
that feature in the `best input<DefBestInput>` approach (see
:ref:`DiscBestInput<DiscBestInput>`) can be used. Note that we could
have a more complex structure describing the link between model and
reality, in which case the denominator of the implausibility measure
would be altered. When :math:`\strut{I_{(i)}(x)}` is large, this suggests
that, even given all the uncertainties present in the problem, namely,
the emulator uncertainty :math:`\strut{{\rm Var}[f(x)]}`, the model
discrepancy :math:`\strut{{\rm Var}[d]}` and the observational errors
:math:`\strut{{\rm Var}[e]}`, we would be unlikely to view as acceptable
the match between model output and observed data were we to run the
model at input :math:`\strut{x}`. Therefore, we consider that choices of
:math:`\strut{x}` for which :math:`\strut{I_{(i)}(x)}` is large can be
discarded as potential members of the set :math:`\strut{\mathcal{X}}` of
all acceptable inputs. We discard regions of the input space by imposing
suitable cutoffs on the implausibility measure as is discussed in
:ref:`DiscImplausibilityCutoff<DiscImplausibilityCutoff>`.

Note that this definition of the univariate implausibility is natural
and comes from the concept of standardised distances. As we will simply
be imposing cutoffs on such measures, any results will be invariant to a
monotonic transformation of the implausibility measure.

Multivariate Implausibility Measures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

So far we have a separate implausibility measure :math:`\strut{I_{(i)}(x)}`
for each of the outputs labelled by :math:`\strut{i}` that were considered
for the history matching process. Note that not all outputs have to be
used: we can select a subset of the outputs that are deemed to be
representative in some sense; e.g., using principal variables (Cumming,
2007). We may now choose to make some intuitive combination of the
individual implausibility measures as a basis of eliminating portions of
the input space, or we may construct the natural multivariate analogue.

A common such combination is the maximum implausibility measure
:math:`\strut{I_M(x)}` defined as

:math:`\strut{ I_M(x) \\;\; = \\;\; \\max_i I_{(i)}(x) }`

Discarding every :math:`\strut{x}` such that :math:`\strut{I_M(x) > c }` is
equivalent to applying the cutoff to every individual
:math:`\strut{I_{(i)}(x)}`. While this is a simple, intuitive and commonly
used measure, it is sensitive to problems caused by inaccurate
emulators: if one of the emulators is performing poorly at input
:math:`\strut{x}`, this could lead to :math:`\strut{x}` being wrongly
rejected. For this reason the second and the third maximum
implausibility measures :math:`\strut{I_{2M}(x)}` and
:math:`\strut{I_{3M}(x)}` are often used (Vernon 2010, Bower 2009), defined
using set notation as:

:math:`\strut{ I_{2M}(x) = \\max_i ( \\; \\{ I_{(i)}(x) \\} \\setminus
I_M(x) \\; ) }`

:math:`\strut{ I_{3M}(x) = \\max_i ( \\; \\{ I_{(i)}(x) \\} \\setminus \\{
I_M(x), I_{2M}(x) \\} \\; ) }`

that is we define :math:`\strut{I_{2M}(x)}` and :math:`\strut{I_{3M}(x)}` to
be the second and third highest value out of the set of univariate
measures :math:`\strut{I_{i}(x)}` respectively. These measures are much
less sensitive to the failings of individual emulators.

We can construct the full multivariate implausibility measure
:math:`\strut{I_{MV}(x)}` provided we have suitable multivariate
expressions for the model discrepancy :math:`\strut{d}` and the
observational errors :math::ref:`\strut{e}` (see
`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`).
Normally a full multivariate emulator is also required, however a
multi-output emulator of sufficient accuracy can also be used (see
:ref:`ThreadVariantMultipleOutputs<ThreadVariantMultipleOutputs>`).

:math:`\strut{I_{MV}(x)}` takes the form:

:math:`\strut{ I_{MV}(x) \\;\;=\;\; (z -{\rm E}[f(x)])^T ({\rm Var}[z-{\rm
E}[f(x)]])^{-1} (z -{\rm E}[f(x)]) }`

which becomes

:math:`\strut{ I_{MV}(x) \\;\;=\;\; (z -{\rm E}[f(x)])^T ({\rm Var}[f(x)] +
{\rm Var}[d] + {\rm Var}[e])^{-1} (z -{\rm E}[f(x)]) }`

Here, :math:`\strut{{\rm Var}[f(x)]}`, :math:`\strut{{\rm Var}[d]}` and
:math:`\strut{{\rm Var}[e]}` are all :math:`\strut{r\times r}` covariance
matrices and :math:`\strut{z}` and :math:`\strut{{\rm E}[f(x)]}` are
:math:`\strut{r}`-vectors, where :math:`\strut{r}` is the number of outputs
chosen for use in the history matching process.

The multivariate form is more effective for screening the input space,
but it does require careful consideration of the covariance structure
for the various quantities involved, especially the model discrepancy
:math:`\strut{d}`.

The history matching process requires that we choose appropriate cutoffs
for each of the above measures: possible choices of such cutoffs are
discussed in
:ref:`DiscImplausibilityCutoff<DiscImplausibilityCutoff>`.

Additional Comments
-------------------

The implausibility measures based on summaries of univariate measures,
such as :math:`\strut{I_M(x)}` and :math:`\strut{ I_{2M}(x)}`, tend to select
inputs that correspond to runs where all or most of the outputs are
within a fixed distance from each of the observed data points. While
this is useful, if the outputs represent points on some physical
function, this type of implausibility measure will not capture the
*shape* of this function. The fully multivariate measure
:math:`\strut{I_{MV}(x)}` on the other hand, can be constructed to favour
runs which mimic the correct shape of the physical function. The extent
of this effect depends critically on the structure of the model
discrepancy and observational error covariance matrices.

References
----------

Cumming, J. A. and Wooff, D. A. (2007), “Dimension reduction via
principal variables,” Computational Statistics & Data Analsysis, 52,
550–565.

Vernon, I., Goldstein, M., and Bower, R. (2010), “Galaxy Formation: a
Bayesian Uncertainty Analysis,” MUCM Technical Report 10/03

Bower, R., Vernon, I., Goldstein, M., et al. (2009), “The Parameter
Space of Galaxy Formation,” to appear in MNRAS; MUCM Technical Report
10/02
