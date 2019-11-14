.. _DiscUANugget:

Discussion: Nugget effects in uncertainty and sensitivity analyses
==================================================================

Description and Background
--------------------------

The possibility of including a :ref:`nugget<DefNugget>` term in the
correlation function for a :ref:`Gaussian process<DefGP>`
:ref:`emulator<DefEmulator>` is discussed in the alternatives page on
emulator prior correlation function
(:ref:`AltCorrelationFunction<AltCorrelationFunction>`). In the
procedure pages for uncertainty analysis (:ref:`ProcUAGP<ProcUAGP>`)
and variance based sensitivity analysis
(:ref:`ProcVarSAGP<ProcVarSAGP>`) using a GP emulator, concerned
respectively with :ref:`uncertainty analysis<DefUncertaintyAnalysis>`
and :ref:`variance based<DefVarianceBasedSA>` :ref:`sensitivity
analysis<DefSensitivityAnalysis>` for the :ref:`core
problem<DiscCore>`, closed form expresssions are given for
various integrals for some cases where a nugget term may be included.
This page provides a technical discussion of the reason for some
apparent anomalies in those formulae.

Discussion
----------

The closed form expressions are derived for the generalised Gaussian
correlation function with nugget (see
:ref:`AltCorrelationFunction<AltCorrelationFunction>`)

:math:`c(x,x') = \\nu I_{x=x'} + (1-\nu)\exp\{-(x-x')^T C (x-x')\} \`

where :math:`I_{x=x'}` equals 1 if :math:`x=x'` but is otherwise zero, and
:math:`\nu` represents the nugget term. They also assume a normal
distribution for the uncertain inputs.

In both :ref:`ProcUAGP<ProcUAGP>` and
:ref:`ProcVarSAGP<ProcVarSAGP>` a number of integrals are computed
depending on :math:`c(x,x')`, and we might expect the resulting integrals
in each case to have two parts, one multiplied by :math:`\nu` and the other
multiplied by :math:`(1-\nu)`. In most cases, however, only the second term
is found.

The reason for this is the fact that the nugget only arises when the two
arguments of the correlation function are equal. The integrals all
integrate with respect to the distribution of the uncertain inputs
:math:`x`, and since this is a normal distribution it gives zero
probability to :math:`x` equalling any particular value. We therefore find
that in almost all of these integrals the nugget only appears in the
integrand for a set of probability zero, and so it does not appear in
the result of evaluating that integral. The sole exception is the
integral denoted by :math:`U_p`, where the nugget does appear leading to a
result :math:`U_p=\nu+(1-\nu)=1`.

The heuristic explanation for this is that a nugget term is a white
noise addition to the smooth simulator output induced by the generalised
Gaussian correlation term. If we average white noise over even a short
interval the result is necessarily zero because in effect we are
averaging an infinite number of independent terms.

The practical implication is that the nugget reduces the amount that we
learn from the training data, and also reduces the amount that we could
learn in sensitivity analysis calculations by learning the values of a
subset of inputs.
