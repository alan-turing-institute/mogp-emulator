.. _DiscObservations:

Discussion: The Observation Equation
====================================

Description and background
--------------------------

Observations on the real system, measurement errors and their relation
to the real system are presented in the variant thread on linking models
to reality using :ref:`model discrepancy<DefModelDiscrepancy>`
(:ref:`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`).
Here, we discuss observations on the real system in more detail.

Notation
~~~~~~~~

In accordance with the standard toolkit notation, we denote the model
:ref:`simulator<DefSimulator>` by :math:`\strut{f}` and its inputs by
:math:`\strut{x}`. We denote by :math:`\strut{z}` an observation or
measurement on a real system value :math:`y`, and denote by :math:`\epsilon`
the measurement error
(:ref:`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`).
The notation covers replicate observations, as described in the
discussion page on structured forms for the model discrepancy
(:ref:`DiscStructuredMD<DiscStructuredMD>`).

Discussion
----------

Statistical assumptions
~~~~~~~~~~~~~~~~~~~~~~~

While a value for :math:`\strut{z}` is observed (for example, the
temperature at a particular time and place on the Earth's surface),
neither the real system value :math:`y` (the actual temperature at that
time and place) nor the measurement error :math:`\epsilon` is observed. We
link these random quantities :math:`\strut{z}`, :math:`y` and :math:`\epsilon`
using the observation equation

:math:` z \\;\;=\;\; y + \\epsilon \`

as defined in
:ref:`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`.

Typical statistical assumptions are that :math:`y` and :math:`\epsilon` are
independent or uncorrelated with :math:`\textrm{E}[\epsilon]=0` and
:math:`\textrm{Var}[\epsilon]=\Sigma_\epsilon`.

The variance matrix :math:`\Sigma_\epsilon` is often assumed to be either
completely specified (possibly from extensive experience of using the
measurement process) and is the same for each observation, or have a
particular simple parameterised form; for example,
:math:`\sigma_\epsilon^2\Sigma`, where :math:`\Sigma \` is completely
specified and :math:`\sigma_\epsilon` is an unknown scalar standard
deviation that we can learn about from the field observations
:math::ref:`\strut{z}`, especially when there are replicate observations; see
`DiscStructuredMD<DiscStructuredMD>`.

Simple consequences of the statistical assumptions are
:math:`\textrm{E}[z]=\textrm{E}[y]` and :math:`\textrm{Var}[z]=\textrm{Var}[y]
+ \\Sigma_\epsilon`. Thus, :math:`\strut{z}` is unbiased for the expected
real system value but the variance of :math:`\strut{z}` is the measurement
error variance inflated by :math:`\textrm{Var}[y]` which can be quite
large, as it involves model discrepancy variance: see,
:ref:`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`.

Observations as functions of system values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes our observations :math:`\strut{z}` are known functions :math:`g(y)`
of system values :math:`y` plus measurement error. The linear case :math:` z=Hy
+ \\epsilon`, where :math:`\strut{H}` is a matrix which can either select
a collection of individual components of :math:`y`, or more general linear
combinations, such as averages of certain components of :math:`y`, is
straightforward and can be dealt with using our current methodology. In
this case, both the expectation and variance of :math:`\strut{z}` are
simply expressed. Moreover, :math:`Hy=Hf(x^+) +Hd`, which we can re-write
in an obvious reformulation as :math::ref:`y'=f'(x^+) +d'`, so that
:math:`\strut{x^+}` is the still the `best input<DefBestInput>` and
:math::ref:`\strut{f}'` and :math:`\strut{d'}` are still independent
(`DiscBestInput<DiscBestInput>`).

The case where measurements :math:`z=g(y)+\epsilon` are made on a nonlinear
function :math:`g` can also be reformulared as :math:`y'=f'(x^+) +d'` with the
usual best input assumptions. Thus, if we put :math:`y'=g(y)` and
:math:`f'(x)=g(f(x))`, we may write :math:`z=y'+\epsilon` and
:math:`y'=f'(x^{+'})+d'` with :math:`\strut{d'}` independent of
:math:`\strut{f'}` and :math:`\strut{x^{+'}}`, and analysis may proceed as
before. It should be noted, however, that when :math:`\strut{g}` is
nonlinear, it can be shown that it is incoherent to simultaneously apply
the best input approach to both :math:`\strut{f}` and :math:`\strut{f'}`.
However, in practice, we choose the formulation which best suits our
purpose.

Additional comments
-------------------

Note that we are assuming measurement errors are additive, while in some
situations they may be multiplicative; that is, :math:`z=y\epsilon`, in
which case we can try either to work directly with the multiplicative
relationship or with the additive relationship on the logarithmic scale,
:math:`\log z=\log y+\log \\epsilon`, with neither case being
straightforward. However, note that this case is covered by the
discussion above about nonlinear functions, with :math:`g(y)=\log (y)`.

Sometimes we have replicate system observations: see
:ref:`DiscStructuredMD<DiscStructuredMD>` for a detailed account,
including notation to accommodate generalised indexing such as
space-time location of observations, which are regarded as control
inputs to the simulator :math:`\strut{f}` in addition to inputs
:math:`\strut{x}`.
