.. _ProcPredictMultiOutputFunction:

Procedure: Predicting a function of multiple outputs
====================================================

Description and Background
--------------------------

When we have a multiple output simulator, we will sometimes be
interested in a deterministic function of one or more of the outputs.
Examples include:

-  A simulator with outputs that represent amounts of rainfall at a
   number of locations, and we wish to predict the total rainfall over a
   region, which is the sum of the rainfall outputs at the various
   locations (or perhaps a weighted sum if the locations represent
   subregions of different sizes).
-  A simulator that outputs the probability of a natural disaster and
   its consequence in loss of lives, and we are interested in the
   expected loss of life, which is the product of these two outputs.
-  A simulator that outputs atmospheric :math:`CO_2` concentration and
   global temperature, and we are interested in using them to compute
   the gross primary productivity of an ecosystem.

If we know that we are *only* interested in one particular function of
the outputs, then the most efficient emulation method is to build a
single output emulator for the output of that function. However, there
are situations when it is better to first build a multivariate emulator
of the raw outputs of the simulator

-  when we are interested in both the raw outputs and one or more
   functions of the outputs;
-  when we are interested in function(s) that depend not just on the raw
   outputs of the simulator, but also on some other auxiliary variables.

In such situations we build the multivariate emulator by following
:ref:`ThreadVariantMultipleOutputs<ThreadVariantMultipleOutputs>`.
The multivariate emulator can then be used to predict any function of
the outputs, at any set of auxiliary variable values, by following the
procedure given here.

We consider a simulator :math:`f(.)` that has :math:`r` outputs, and a
function of the outputs :math:`g(.)`. The procedure for predicting
:math:`g(f(.))` is based on generating random samples of output values from
the emulator, using the procedure
:ref:`ProcOutputSample<ProcOutputSample>`.

Inputs
------

-  A multivariate emulator, which is either a multivariate GP obtained
   using the procedure
   :ref:`ProcBuildMultiOutputGP<ProcBuildMultiOutputGP>`, or a
   multivariate t-process obtained using the procedure
   :ref:`ProcBuildMultiOutputGPSep<ProcBuildMultiOutputGPSep>`,
   conditional on hyperparameters.
-  :math:`s` sets of hyperparameter values.
-  A single point :math:`x^\prime` or a set of :math:`n^\prime` points
   :math:`x^\prime_1, x^\prime_2,\ldots,x^\prime_{n^\prime}` at which
   predictions are required for the function :math:`g(.)`.
-  :math:`N`, the size of the random sample to be generated.

Outputs
-------

-  Predictions of :math:`g(.)` at :math:`x^\prime_1,
   x^\prime_2,\ldots,x^\prime_{n^\prime}` in the form of a sample of
   size :math:`N` of values from the predictive distribution of :math:`g(.)`.

Procedure
---------

For :math:`j=1,...,N`,

#. Pick a set of hyperparameter values at random from the :math:`s` sets
   that are available.
#. Generate a :math:`n^\prime r \\times 1` random vector :math:`F^{j}` from
   the emulator, using the procedure set out in the \`Multivariate
   output general case' section of
   :ref:`ProcOutputSample<ProcOutputSample>`.
#. Form the :math:`r \\times n^\prime` matrix :math:`M^{j}` such that
   :math:`\mathrm{vec}[M^{jT}]=F^{j}`.
#. For :math:`\ell=1,...,n^\prime`, let :math:`m^{j}_\ell` be the :math:`\ell`th
   column of :math:`M^{j}` and let :math:`G^{j}_\ell=g(m_\ell)`

The sample is then :math:`\{G^{j} : j=1,...,N\}`, where
:math:`G^{j}=(G^{j}_1,...,G^{j}_{n^\prime})`.
