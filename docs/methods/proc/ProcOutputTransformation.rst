.. _ProcOutputTransformation:

Procedure: Transformed outputs
==============================

Description and Background
--------------------------

It is sometimes appropriate to build an :ref:`emulator<DefEmulator>`
of some *transformation* of the :ref:`simulator<DefSimulator>` output
of interest, rather than the output itself. See the discussion page on
the Gaussian assumption
(:ref:`DiscGaussianAssumption<DiscGaussianAssumption>`) for the
background to using output transformations.

The emulator allows inference statements about the transformed output,
and for instance can be used to conduct :ref:`uncertainty
analysis<DefUncertaintyAnalysis>` or :ref:`sensitivity
analysis<DefSensitivityAnalysis>` of the transformed output.
However, the interest lies in making such inferences and analyses on the
original, untransformed output. The procedure explains how to construct
these from a fully :ref:`Bayesian<DefBayesian>` emulator of the
transformed output.

In the case of a :ref:`Bayes linear<DefBayesLinear>` emulator,
entirely different methods are needed to make inferences about the
original output.

Inputs
------

The input is a fully Bayesian emulator for the transformed simulator
output.

We will use the following notation. For any given input configuration
:math:`x`, let the original output be :math:`f(x)`, and let the transformed
output be :math:`t(x) = g\{f(x)\}`, so that :math:`g` denotes the
transformation. The emulator therefore provides a probability
distribution for :math:`t(x)` at any or all values of :math:`x`. We suppose
that the transformation is one-to-one and that the inverse
transformation is :math:`g^{-1}`, i.e. :math:`f(x) = g^{-1}\{t(x)\}`.

Outputs
-------

Outputs are any desired inferences about properties of the original
output. For instance, if we let the inputs be random, denoting them now
by :math:`X`, then uncertainty analysis of :math:`f(X)` might include as one
specific inference the expectation (with respect to the :ref:`code
uncertainty<DefCodeUncertainty>`) of the uncertainty mean :math:`M =
\\textrm{E}[f(X)]`. In this case the property is :math:`M` and the
inference is the mean (interpreted as an estimate of :math:`M`).

Procedure
---------

The simplest procedure is to use simulation. The method of :ref:`simulation
based inference<ProcSimulationBasedInference>` for emulators
requires only a little modification. The method involves drawing random
realisations from the emulator distribution, and then computing the
property in question for each such realisation. The set of property
values so derived is a sample from the (code uncertainty) distribution
of that parameter. From this sample we compute the necessary inferences.

When we have a transformed output, we add one more step. We draw random
realisations from the emulator for :math:`t(x)`, but we now apply the
inverse transformation :math:`g^{-1}` to every point on the realisation
before computing the property value. This ensures that the parameter
values are now a sample from the distribution of that property as
defined for the original output.

Additional Comments
-------------------

There are specific cases where we can do better than this. For some
transformations we can derive the distribution of :math:`f(x)` for any
given :math::ref:`\strut x` analytically, at least conditionally on
`hyperparameters<DefHyperparameter>`. In some cases, we may even
be able to derive uncertainty analysis or sensitivity analysis. This is
an area for ongoing research.
