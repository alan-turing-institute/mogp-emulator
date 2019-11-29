.. _DiscToolkitSensitivityAnalysis:

Discussion: Sensitivity analysis in the toolkit
===============================================

Description and background
--------------------------

This page is part of the topic thread on :ref:`sensitivity
analysis<DefSensitivityAnalysis>` (SA). The basic ideas of SA
are introduced in the topic thread on sensitivity analysis
(:ref:`ThreadTopicSensitivityAnalysis<ThreadTopicSensitivityAnalysis>`),
where the various uses of SA are outlined. In the
:ref:`MUCM<DefMUCM>` toolkit, the favoured method of SA is
probabilistic SA, see the discussion page
:ref:`DiscWhyProbabilisticSA<DiscWhyProbabilisticSA>`. Measures of
sensitivity for two particular forms of probabilistic SA are developed
in the discussion pages on Variance based SA
(:ref:`DiscVarianceBasedSA<DiscVarianceBasedSA>`) and decision based
SA (:ref:`DiscDecisionBasedSA<DiscDecisionBasedSA>`). Practical
issues concerning the usage of these methods are discussed in the
sensitivity analysis measures for simplification page
(:ref:`DiscSensitivityAndSimplification<DiscSensitivityAndSimplification>`),
the sensitivity measures for output uncertainty page
(:ref:`DiscSensitivityAndOutputUncertainty<DiscSensitivityAndOutputUncertainty>`)
and the sensitivity measures for decision uncertainty page
(:ref:`DiscSensitivityAndDecision<DiscSensitivityAndDecision>`). Here
we consider how SA is dealt with in the toolkit, and provide links to
relevant pages within the main toolkit threads.

The MUCM toolkit is primarily concerned with methods to understand and
manage uncertainty in the outputs of a :ref:`simulator<DefSimulator>`
through building an :ref:`emulator<DefEmulator>`. The key advantage
of this approach is computational. Simulators are often very large
computer programs requiring substantial computer power and time to run.
Consequently, the number of times that the simulator can be run is
typically small, and this makes it difficult to carry out various kinds
of tasks that we might wish to perform using the simulator. The MUCM
approach consists basically of first using a relatively small number of
runs to build the emulator, and then using the emulator (rather than the
simulator itself) to carry out the desired task.

SA is a very good example of a task that can be tackled more efficiently
using emulators. The various SA measures presented in
:ref:`DiscVarianceBasedSA<DiscVarianceBasedSA>` or
:ref:`DiscDecisionBasedSA<DiscDecisionBasedSA>` all involve
integration of the simulator output with respect to uncertainty in one
or more of the inputs. In order to compute such integrals exactly, we
must effectively run the simulator at every possible value of the
uncertain inputs, which (in theory at least) requires an infinite number
of runs. Any attempt to compute such an integral in practice is
therefore subject to computational error. A widely used method is Monte
Carlo, in which a random sample of values are drawn for the uncertain
inputs, and the simulator is run for each of the sampled input sets. To
make the sampling error in the resulting estimates of SA measures
appropriately small will typically require thousands of simulator runs
(and fresh samples are usually required to compute each SA measure of
interest). Using emulators, the number of runs required for comparable
levels of computational accuracy may be orders of magnitude smaller.

The discussion here is in two parts. First we consider how the
additional uncertainty due to emulation (known as :ref:`code
uncertainty<DefCodeUncertainty>`) interacts with the measures of
sensitivity. The second part points to where specific methods for SA
computations are presented in the toolkit's main threads (i.e. the core,
variant and generic threads; see
:ref:`MetaToolkitStructure<MetaToolkitStructure>`).

Discussion
----------

Code uncertainty
~~~~~~~~~~~~~~~~

Notation
^^^^^^^^

Computation of any SA measure, such as the sensitivity index :math:`V_j`
for input :math:`x_j`, using emulator techniques is subject to code
uncertainty. The emulator is an expression of our knowledge or beliefs
about a simulator's output :math:`f(x)` based on a :ref:`training
sample<DefTrainingSample>` of simulator runs. In the fully
:ref:`Bayesian<DefBayesian>` form of emulator based on a :ref:`Gaussian
process<DefGP>` (GP), the emulator is a posterior distribution
for :math:`f(x)`. A :ref:`Bayes linear<DefBayesLinear>` (BL) emulator
provides instead the adjusted mean and variance for :math:`f(x)` based on
the training sample. The posterior distribution or adjusted mean and
variance represent uncertainty about :math:`f(x)` due to emulation, called
code uncertainty.

Consequently, any computation of SA measures is subject also to code
uncertainty; expressed as a posterior distribution in the case of GP
emulation or adjusted mean and variance for BL emulation. The posterior
mean or adjusted mean, denoted by :math:`\mathrm{E}^*`, is the computed
value or estimate of the measure, and the posterior variance or adjusted
variance, denoted by :math:`\mathrm{Var}^*`, characterises the degree of
code uncertainty about its value.

For example, :math:`\mathrm{E}^*[V_j]` is the computed value or estimate of
the sensitivity index :math:`V_j`, with code uncertainty quantified by
:math:`\mathrm{Var}^*[V_j]`.

Code uncertainty in variance-based SA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In both probabilistic SA and :ref:`uncertainty
analysis<DefUncertaintyAnalysis>` (UA) the focus is on
uncertainty that is induced by uncertainty about simulator inputs, and
the question arises as to whether code uncertainty should be combined in
some way with input uncertainty. If we first consider UA, then as
described in
:ref:`DiscSensitivityAndOutputUncertainty<DiscSensitivityAndOutputUncertainty>`
the uncertainty variance :math:`V=\mathrm{Var}[f(X)]` expresses overall
uncertainty about the simulator output :math:`f(X)` when there is
uncertainty about the inputs (in which case we conventionally use the
capital letter :math:`X` to rerpresent a random variable).
Similarly, :math:`M=\mathrm{E}[f(X)]` is generally thought of as the best
estimate of :math:`f(X)` in the presence of input uncertainty. But
uncertainty about the simulator output is not confined to uncertainty
about :math:`X`; we are also uncertain about the simulator itself,
:math:`f(\cdot)`, and this uncertainty is code uncertainty.

In the presence of both input and code uncertainty, we would estimate
:math:`f(x)` by

.. math::
   M^* = \mathrm{E}^*[M],

with overall uncertainty comprising

.. math::
   V^* = \mathrm{E}^*[V] + \mathrm{Var}^*[M].

So in addition to the posterior estimated or computed value of :math:`V`
we also have code uncertainty about :math:`M`.

In variance based SA, set out in
:ref:`DiscVarianceBasedSA<DiscVarianceBasedSA>`, we consider what
part of the overall output uncertainty :math:`V` is attributable to
a single input :math:`x_j` or a group of inputs :math:`x_J`. Should code
uncertainty also be a part of these computations? In particular, should
variance based SA analyse how much of :math:`V^*` (rather than
:math:`V`) is attributable to :math:`x_j` or :math:`x_J`?

The answer basically is that learning about the uncertain inputs does
not affect code uncertainty. No matter how much and in what way we
reduce uncertainty about :math:`X`, the code uncertainty
:math:`\mathrm{Var}^*[M]` will always be a component of the overall
uncertainty about :math:`f(x)`. The computed sensitivity variances
:math:`\mathrm{E}^*[V_J]` or interaction variances :math:`\mathrm{E}^*[V^I_J]`
quantify reductions in the uncertainty of :math:`f(x)`, whether we think of
this as being just the computed input uncertainty variance
:math:`\mathrm{E}^*[V]` or the overall variance :math:`V^*`.

In the case of independent inputs, the computed main effect and
interaction variances, :math:`\mathrm{E}^*[V_j]` and
:math:`\mathrm{E}^*[V^I_J]`, provide a partition of :math:`\mathrm{E}^*[V]` in
exactly the way described in the discussion page on the theory of
variance based SA
(:ref:`DiscVarianceBasedSATheory<DiscVarianceBasedSATheory>`), while
:math:`V^*` is partitioned by these terms plus the code uncertainty
:math:`\mathrm{Var}^*[M]`. We can define a sensitivity index for :math:`x_J`
as :math:`\mathrm{E}^*[V_J]/\mathrm{E}^*[V]` or allowing for code
uncertainty as :math:`\mathrm{E}^*[V_J]/V^*`. In the latter case, code
uncertainty has its own index :math:`\mathrm{Var}^*[M]/V^*`.

Code uncertainty in decision-based SA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The situation in decision-based SA is potentially more complex because
code uncertainty is not just a fixed addition to the computed values of
the measures presented in
:ref:`DiscDecisionBasedSA<DiscDecisionBasedSA>`. We can contrast, on
the one hand, decision making in the presence of code uncertainty with,
on the other hand, code uncertainty about a decision.

If we simply consider the code uncertainty as part of the overall
uncertainty about the simulator outputs, along with that induced by
input uncertainty, then we define the expected utility to be the
expectation with respect to both code uncertainty and input uncertainty,
and so define

.. math::
   \bar{L}^*(d) = {\rm E}^*[\bar{L}(d)],

and the optimal decision :math:`\strut M^*` minimises this expected loss.
With analogous definitions of :math:`\bar{L}_J^*(d,x_J)` and
:math:`M_J^*(x_J)` when we learn the value of :math:`x_J`, we can define the
expected value of information

.. math::
   V_J^* = \bar{L}^*(M^*) - {\rm E}[\bar{L}_J^*(M_J^*(X_J),X_J)].

However, this analysis is for decision making in the presence of code
uncertainty. It implicity supposes that we learn about :math:`x_J` without
learning any more about the simulator, i.e. it is a value of information
for reducing input uncertainty without reducing code uncertainty. If we
are computing value of information measures for the purpose of
prioritising research then it would be right to ignore any additional
code uncertainty, because the computational cost of making enough
simulator runs to render code uncertainty negligible will typically be
much less than any research cost to reduce input uncertainty.

In this context, we are concerned with code uncertainty about a
decision. First define the expected value of eliminating code
uncertainty as

.. math::
   V^* = \bar{L}^*(M^*) -{\rm E}^*[\bar{L}(M)].

Then the expected value of information from learning the value of
:math:`X_J` becomes simply :math:`{\rm E}^*[V_J]`. If we can also calculate
:math:`{\rm Var}^*[V_J]` then this will quantify code uncertainty about
that value of information.

SA in toolkit threads
~~~~~~~~~~~~~~~~~~~~~

The core, variant and generic threads in the toolkit are designed to
take the reader through the process of building an emulator and then
through to using it for various standard tasks, one of which is SA. So
in principle procedures for carrying out SA should feature in each of
these threads. In practice, not all of them yet have detailed SA
procedures, although in due course it is the intention for all to do so.

-  In the core thread for GP emulators,
   :ref:`ThreadCoreGP<ThreadCoreGP>`, there are detailed procedures
   for variance based SA in :ref:`ProcVarSAGP<ProcVarSAGP>`.
-  In the core thread for BL emulators,
   :ref:`ThreadCoreBL<ThreadCoreBL>` refers also to
   :ref:`ProcVarSAGP<ProcVarSAGP>`. In Bayes linear theory it is more
   usual to express knowledge about uncertain quantities by means,
   variances and covariances, rather than complete distributions. So
   although the use of a full probability distribution :math:`\omega(x)`
   for the inputs, which is needed for probabilistic SA, is legitimate
   within BL theory (and allows :ref:`ProcVarSAGP<ProcVarSAGP>` to be
   used) it does not fit so comfortably in the BL thread.
-  In the variant thread dealing with multiple outputs,
   :ref:`ThreadVariantMultipleOutputs<ThreadVariantMultipleOutputs>`,
   there are as yet no detailed SA procedures.
-  In the variant thread dealing with dynamic emulators,
   :ref:`ThreadVariantDynamic<ThreadVariantDynamic>`, there are as
   yet no detailed SA procedures.
-  In the variant thread dealing with two-level emulation,
   :ref:`ThreadVariantTwoLevelEmulation<ThreadVariantTwoLevelEmulation>`,
   the emphasis is on using many runs of a cheap simulator to construct
   prior information. The result is an emulator built according to the
   core GP or BL methods, so that the relevant details are given again
   in :ref:`ProcVarSAGP<ProcVarSAGP>`.
-  In the variant thread dealing with using observed derivatives,
   :ref:`ThreadVariantWithDerivatives<ThreadVariantWithDerivatives>`,
   there are as yet no detailed procedures for SA.
-  In the generic thread dealing with emulating derivatives,
   :ref:`ThreadGenericEmulateDerivatives<ThreadGenericEmulateDerivatives>`,
   it is remarked that derivatives are used for local SA (see
   :ref:`DiscWhyProbabilisticSA<DiscWhyProbabilisticSA>`).
-  In the generic thread dealing with combining multiple emulators,
   :ref:`ThreadGenericMultipleEmulators<ThreadGenericMultipleEmulators>`,
   there are as yet no detailed procedures for SA.

Additional comments
-------------------

Although none of the threads has any procedures for decision based SA,
it is hoped to address this topic in some threads in future.
