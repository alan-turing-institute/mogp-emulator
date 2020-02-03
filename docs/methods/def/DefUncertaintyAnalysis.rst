.. _DefUncertaintyAnalysis:

Definition of Term: Uncertainty analysis
========================================

One of the most common tasks for users of
:ref:`simulators<DefSimulator>` is to assess the uncertainty in the
simulator output(s) that is induced by their uncertainty about the
inputs. Input uncertainty is a feature of most applications, since
simulators typically require very many input values to be specified and
the user will be unsure of what should be the best or correct values for
some or all of these.

Uncertainty regarding inputs is characterised by a (joint) probability
distribution for their values. This distribution induces a (joint)
probability distribution for the output, and uncertainty analysis
involves identifying that distribution. Specific tasks might be to
compute the mean and variance of the output uncertainty distribution.
The mean can be considered as a best estimate for the output in the face
of input uncertainty, while the variance measures the amount of output
uncertainty. In some applications, it is important to evaluate the
probability that the output would lie above or below some threshhold.

A simple way to compute these things in practice is the Monte Carlo
method, whereby random configurations of inputs are drawn from their
input uncertainty distribution, the model is run for each such
configuration, and the set of outputs obtained comprises a random sample
from the output distribution. If a sufficiently large sample can be
taken, then this allows the uncertainty distribution, mean, variance,
probabilities etc., to be evaluated to any desired accuracy. However,
this is often impractical because the simulator takes too long to run.
Monte Carlo methods will typically require 1,000 to 10,000 runs to
achieve accurate estimates of the uncertainty measures, and if a single
simulator run takes more than a few seconds the computing time can
become prohibitive.

The :ref:`MUCM<DefMUCM>` approach of first building an
:ref:`emulator<DefEmulator>` has been developed to enable tasks such
as uncertainty analysis to be carried out more efficiently.

Another group of tools that are commonly required are known as
:ref:`sensitivity analysis<DefSensitivityAnalysis>`. In particular,
the :ref:`variance-based<DefVarianceBasedSA>` form of sensitivity
analysis identifies what proportion of the variance of the uncertainty
distribution is attributable to uncertainty in individual inputs, or
groups of inputs.
