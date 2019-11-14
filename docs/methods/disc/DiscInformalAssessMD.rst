.. _DiscInformalAssessMD:

Discussion: Informal Assessment of Model Discrepancy
====================================================

Description and background
--------------------------

We consider informal :ref:`assessment<DefAssessment>` of the :ref:`model
discrepancy<DefModelDiscrepancy>` term :math::ref:`\strut{ d}`
(`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`),
formulated to account for the mismatch between the
:ref:`simulator<DefSimulator>` :math::ref:`\strut{f}` evaluated at the `best
input<DefBestInput>` :math::ref:`\strut{ x^+}`
(`DiscBestInput<DiscBestInput>`) and real system value
:math::ref:`\strut{y}` (`DefModelDiscrepancy<DefModelDiscrepancy>`).
Such informal assessment will be described mainly in terms of estimating
:math::ref:`\strut{\textrm{Var}[d]}` based on system observations :math:`\strut{
z}` (`DiscObservations<DiscObservations>`) and simulator runs
(:ref:`ThreadCoreGP<ThreadCoreGP>` and
:ref:`ThreadCoreBL<ThreadCoreBL>`).

Discussion
----------

Our aim in this page is to assess informally the variance
:math:`\strut{\textrm{Var}[d]}` of the discrepancy term
:math:`\strut{d=y-f(x^+)}` using system observations :math:`\strut{z}` and
simulator output :math:`\strut{F}` from :math:`\strut{n}` simulator runs at
design inputs :math:`\strut{D}`. Our initial assessment of
:math:`\strut{\textrm{E}[d]}` is usually zero, but subsequent analysis may
suggest there is a \`trend' attributable to variation over and above
that accounted for by :math:`\strut{\textrm{Var}[d]}`.

It should be noted that it is only helpful to assess discrepancy by
matching simulated output to observational data when we have many
observations, as otherwise the effects of over-fitting will dominate.
Also, while we are focussing here on informal model discrepancy
assessment methods, these should be considered in combination with
expert assessment, as discussed in
:ref:`DiscExpertAssessMD<DiscExpertAssessMD>`.

The basic idea is to estimate the variance :math:`\strut{\textrm{Var}[d]}`
using differences :math:`\strut{z-f(x)}` evaluated over a carefully
selected collection of simulator runs. We consider two situations: when
the simulator is fast to run and when it is slow to run. In the fast
case, we consider a further dichotomy: the simulator is either a black
box or an open box, where we have access to the internal code and the
equations governing the mathematical model. In the open box case, we
decompose model discrepancies into internal and external model
discrepancies.

Notation
~~~~~~~~

We use the notation described in the discussion page on structured forms
for model discrepancy (:ref:`DiscStructuredMD<DiscStructuredMD>`),
where there is a \`location' input :math:`\strut{u}` (such as space-time)
which indexes simulator outputs as :math:`\strut{f(u,x)}` and corresponding
system values :math:`\strut{y(u)}` measured as observations
:math:`\strut{z(u)}` with model discrepancies :math:`\strut{d(u)}`.

Suppose the system is observed at :math:`\strut{k}` \`locations'
:math:`\strut{u_1,\ldots, u_k}`. Denote the corresponding observations by
:math:`\strut{z_1,\ldots, z_k}`, where :math:`\strut{z_i\equiv z(u_i)}`, the
measurement errors by :math:`\strut{\epsilon_1,\ldots, \\epsilon_k}`, where
:math:`\strut{\epsilon_i\equiv \\epsilon(u_i)}`, the system values by
:math:`\strut{y_1,\ldots, y_k}`, where :math:`\strut{y_i\equiv y(u_i)}`, the
simulator values at input :math:`\strut{x}` by :math:`\strut{f_1(x),\ldots,
f_k(x)}`, where :math:`\strut{f_i(x) \\equiv f(u_i,x)}`, and the model
discrepancies by :math:`\strut{d_1,\ldots, d_k}`, where :math:`\strut{d_i
\\equiv d(u_i)}`. Then :math:`\strut{y_i=f_i(x) + d_i}` and
:math:`\strut{z_i=y_i+\epsilon_i}` so that :math:`\strut{z_i=f_i(x) +
d_i+\epsilon_i}` for :math:`\strut{i=1, \\ldots, k}`.

Fast black box simulators
~~~~~~~~~~~~~~~~~~~~~~~~~

In this section, we consider black box simulators with a fast run-time
for every input combination :math:`\strut{(u,x)}`, allowing for a detailed
comparison of the system observations :math:`\strut{z_1,\ldots, z_k}` to
the :math:`\strut{n \\times r \\times k}` array of simulator outputs
:math:`\strut{f_i(x_j)=f(u_i, x_j)}` at inputs :math:`\strut{x_1,\ldots, x_n}`
and locations :math:`\strut{u_1,\ldots, u_k}`, where the number of
simulator runs :math:`\strut{n}` is very large compared to the number of
components of :math:`\strut{x}`. We choose a space-filling design
:math:`\strut{D}` over the usually rectangular input space :math:`\strut{\cal
X}:ref:`; for example, a Latin hypercube design
(`ProcLHC<ProcLHC>`).

The aim now is to choose those inputs from :math:`\strut{D}` with outputs
which are \`close' to the system observations according to some sensible
criterion, for which there are many possibilities. One such possibility,
the one we choose here to illustrate the process, is to evaluate the
so-called \`implausibility function'

:math:`\strut{ I(x)= \\max_{1 \\leq i \\leq k} \\, (z_i-f_i(x))^{\textrm{T}}
{\Sigma_i}^{-1}(z_i-f_i(x)) }`

at input :math:`\strut{x}` with output :math:`\strut{f_i(x)}`, where
:math:`\strut{\Sigma_i=\Sigma_\epsilon + \\Sigma_{d_i}}` with
:math:`\strut{\Sigma_\epsilon}` denoting the variance matrix of the
measurement errors (assumed to be known and the same for each
:math:`\strut{\epsilon_i}`) and :math:`\strut{\Sigma_{d_i}}` is the variance
matrix of the discrepancy :math:`\strut{d_i}`. Note that it would be
possible to build a full multivariate implausibility function taking
into account all of the correlations across discrepancies at different
locations but this would require a much more detailed level of prior
specification, so that we often prefer to use the simpler form above.

We can evaluate :math:`\strut{I(x)}` provided the :math:`\strut{\Sigma_{d_i}}`
are known. When they are unknown, the case we are considering here, we
propose a modification based on setting the :math:`\strut{\Sigma_{d_i}=0}`
in the implausibility function :math:`\strut{I(x)}`, defined above.

The key idea is to use the implausibility concept to \`rule out' any
input :math:`\strut{x}` for which this modified :math:`\strut{I(x)}` is \`too
large', according to some suitable cutoff :math:`\strut{C}` determined; for
example, by following the development detailed in Goldstein, M.,
Seheult, A. and Vernon, I. (2010). The distributional form of
:math:`\strut{I(x)}` when :math:`\strut{x=x^+}` can be simply derived and
computed, assuming that the :math:`\strut{k}` components in the maximum are
either independent chi-squared random quantities with :math:`\strut{r}`
degrees-of-freedom, or are completely dependent when they are each set
equal to the same chi-squared random quantity with :math:`\strut{r}`
degrees-of-freedom. Of course, while none of these distributional
assumptions will be true, the values of :math:`\strut{C}` should provide a
useful pragmatic \`yardstick'.

We now select :math:`\strut{x^+}` candidates to be that subset
:math:`\strut{D_J}` of the rows of :math:`\strut{D}` corresponding to those
inputs :math:`\strut{x_j}` for which :math:`\strut{I(x_j)\leq C}`. Denote the
corresponding outputs by :math:`\strut{F_J}`. Note that :math:`\strut{D_J}`
could be empty, suggesting that model discrepancy can be large. In
practice, we increase the value of :math:`\strut{C}` to get a non-empty set
of :math:`\strut{x^+}` candidates.

To simplify the discussion, we focus on the univariate case
:math:`\strut{r=1}` so that the variance matrices in the implausibilty
function are all scalars and write :math:`\strut{\Sigma_i}` as
:math:`\strut{\sigma^2_i= \\sigma^2_\epsilon + \\sigma^2_{d_i}}`. We now
choose :math:`\strut{\sigma_{d_i}}` so that

:math:`\strut{ \\max_{j \\in J}\left\|
\\frac{z_i-f_i(x_j)}{\sigma_i}\right\| \\leq 3 }`

although we could, for example, use another choice criterion, such as
using one element of :math:`\strut{J}` which fits well across all
:math:`\strut{u}`. Note that the choice of :math:`\strut{\sigma_{d_i}}` can be
zero, whichever criterion we opt to use. On the other hand, a large
discrepancy standard deviation :math:`\strut{\sigma_{d_i}}` indicates that
the simulator may fail to predict well for reasons not explained by
measurement error. At this point, we could evaluate the implausibility
function using the new value of :math:`\strut{\sigma_{d_i}}`, which would
be informative about the number of runs that are now close to the
engineered value of :math:`\strut{C=3}` and their location in input space,
which might be of interest. Note that an assessment of zero variance for
a model discrepancy term does not suggest that the model is perfect but
simply reflects the situation that we can find good enough fits from our
modelling to the data that we are not forced to introduce such a term,
so that our views as to the value of introducing model discrepancy can
only be formed from expert scientific judgements as to model
limitations, as described in
:ref:`DiscExpertAssessMD<DiscExpertAssessMD>`.

Slow simulators
~~~~~~~~~~~~~~~

Informal assessment of model discrepancy for a slow simulator is similar
to that for a fast simulator, except we replace the simulator by a
relatively fast :ref:`emulator<DefEmulator>`
(:ref:`ThreadCoreGP<ThreadCoreGP>` and
:ref:`ThreadCoreBL<ThreadCoreBL>`). In the univariate case
:math:`\strut{r=1}`, we replace the definition of the implausibility
function in by

:math:`\strut{ I(x)=\max_{1 \\leq i \\leq k}\left\|
\\frac{z_i-\textrm{E}[f_i(x)]}{\sigma_i(x)}\right\| }`

where :math:`\strut{\textrm{E}[f_i(x)]}` denotes the emulator mean for
input :math:`\strut{x}` at location :math:`\strut{u_i}` and
:math:`\strut{\sigma^2_i(x)}` is the sum of three variances: measurement
error variance, model discrepancy variance and the emulator variance
:math:`\strut{\textrm{Var}[f_i(x)]}` at :math:`\strut{x}`.

Since emulator run time will be fast compared to that for the simulator
it emulates, we can evaluate it at many inputs (as we did for fast
simulators) to help determine implausible inputs. As we did for a fast
simulator, we set the discrepancy variance contribution to
:math:`\strut{\sigma^2_i(x)}` equal to zero to help identify some
\`non-implausible' inputs with which to assess discrepancy standard
deviation, using a procedure analogous to that for fast simulators.

Fast open box simulators
~~~~~~~~~~~~~~~~~~~~~~~~

As stated above, a fast open box simulator refers to a situation where
we have access to the internal code and the equations governing the
mathematical model. In this case, we consider two components of model
discrepancy: internal and external model discrepancy.

Internal discrepancy refers to intrinsic limitations to the model whose
order of magnitude we will quantify using simulator output. For example,
a forcing function :math:`\strut{F(u)}`, such as actual rainfall for a
runoff model, may only be determined within :math:`\strut{10\% }`. Then we
may assess the effect on model output by making a series of model
evaluations with varying values of the forcing function within the
specified limits. We may implement this by specifying a distribution for
the unknown \`true values' :math:`\strut{F_1,\ldots, F_k}` of the forcing
function at locations :math:`\strut{u_1,\ldots, u_k}` that reflects our
beliefs and knowledge about their uncertain values, such as the
:math:`\strut{10\% }` example above. We then sample repeatedly from this
distribution and evaluate the simulator output for each forcing function
sample. The associated variation in simulator output allows us to assess
the internal discrepancy variation due to forcing function uncertainty.
Internal discrepancy variation attributable to other model features,
such as initial and boundary conditions, may be assessed similarly; see
Goldstein, M., Seheult, A. and Vernon, I. (2010) for a detailed account.
Finally, all of the internal discrepancies considered are accumulated
into an approximate overall internal discrepancy variance. While it is
informative to assess each of the individual internal discrepancy
contributions, this does require a large number of model evaluations. If
the model is not fast enough, or if we suspect some of the discrepancy
contributions to be highly correlated, then it is often advisable to
vary simultaneously the intrinsic limitations to assess the overall
internal discrepancy variance, rather than treat them as if they were
independent as in the approximate overall assessment described above. It
should be emphasised that the overall internal discrepancy variance will
be used when the values of each of the intrinsic limitation
contributions (such as a forcing function) are fixed in the simulator,
most likely at their original nominal values.
:ref:`DiscExchangeableModels<DiscExchangeableModels>` suggests an
alternative approach to forcing function uncertainty and other intrinsic
limitations to the mathematical model.

We refer to those remaining aspects of model discrepancy concerning
differences between the model and the system, arising from features
which we cannot quantify by simulation, as external structural
discrepancies. Their overall variation can now be assessed using the
methods described for fast black box simulators, except that selection
of :math:`\strut{x^+}` candidates is improved by adding the internal
discrepancy variance to the measurement error variance in the modified
implausibility function.

Note that there is no hard and fast division between aspects of
discrepancies that we can form a judgement on by tweaking the model and
aspects that we can form a judgement on by expert assessment. Both
methods can be treated in exactly the same way in determining portions
of the overall discrepancy variance in this approach.

Additional comments
-------------------

Where we simplified the discussion in the fast black box simulators
section by focussing on the case :math:`\strut{r=1}`, we can still treat
the general case by stringing out the :math:`\strut{r \\times k}` output
into an :math:`\strut{rk}`-vector and follow the :math:`\strut{r=1}`
procedure. While this is reasonable informally, it only assesses
variances, not covariances; but informal covariance assessment would
require more observations than typically available.

References
----------

Goldstein, M., Seheult, A. and Vernon, I. (2010), "Assessing Model
Adequacy", MUCM Technical Report 10/04.
