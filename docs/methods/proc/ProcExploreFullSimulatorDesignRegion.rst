.. _ProcExploreFullSimulatorDesignRegion:

Procedure: Explore the full simulator design region to identify a suitable single step function design region
=============================================================================================================

Description and Background
--------------------------

This page is concerned with task of :ref:`emulating<DefEmulator>` a
:ref:`dynamic simulator<DefDynamic>`, as set out in the variant
thread on dynamic emulation
(:ref:`ThreadVariantDynamic<ThreadVariantDynamic>`).

We have an emulator for the :ref:`single step
function<DefSingleStepFunction>` :math:` w_t=f(w_{t-1},a_t,\phi) \`
given an initial assessment of the input region of interest :math:`
\\mathcal{X}_{single} \`, and some training data. However, the
'correct' region :math:` \\mathcal{X}_{single} \` depends on the input
region of interest :math:` \\mathcal{X}_{full} \` for the full simulator,
and the single step function :math:` f(.)`. Here, we use simulation to make
an improved assessment of :math:` \\mathcal{X}_{single} \` given :math:`
\\mathcal{X}_{full} \` and an emulator for :math:` f(.) \`.

Inputs
------

-  An emulator for the single step function :math::ref:` w_t=f(w_{t-1},a_t,\phi)
   \`, formulated as a `GP<DefGP>` or
   :ref:`t-process<DefTProcess>` conditional on hyperparameters,
   :ref:`training inputs<DefTrainingSample>` :math:`\strut D \` and
   training outputs :math:`f(D) \`.
-  A set :math:`\{\theta^{(1)},\ldots,\theta^{(s)}\} \` of emulator
   hyperparameter values.
-  The input region of interest :math:` \\mathcal{X}_{full} \` for the full
   simulator.

Outputs
-------

-  A set of :math:`\strut N` simulated joint time series :math:`
   \\{(w_{0}^{(i)},a_1^{(i)}),\ldots, (w_{T-1}^{(i)},a_T^{(i)})\}` for
   :math:`i=1,\ldots,N \`.

Procedure
---------

#. Choose a set of design points :math:` \\{x_1^{full},\ldots,x_N^{full}\}
   \` from :math:` \\mathcal{X}_{full} \`, with :math:`
   x_i^{full}=(w_0^{(i)},a_1^{(i)},\ldots,a_T^{(i)},\phi^{(i)})`. Both
   :math:`\strut N` and the design points can be chosen following the
   principles in the alternatives page on training sample design
   (:ref:`AltCoreDesign<AltCoreDesign>`), but see additional comments
   at the end. Then for :math:`i=1,\ldots,N`:
#. For :math:` x_i^{full}` and :math:` \\theta^{(i)}` , generate one random
   time series :math:` w_1^{(i)},\ldots,w_T^{(i)} \` using the simulation
   method given in the procedure page
   :ref:`ProcExactIterateSingleStepEmulator<ProcExactIterateSingleStepEmulator>`
   (use :math:`\strut R=1 \` within this procedure). Note that we have
   assumed :math:`\strut N\le s \` here. If it is not possible to increase
   :math:`\strut s` and we need :math:`N>s \`, then we suggest cycling round
   the set :math:`\{\theta^{(1)},\ldots,\theta^{(s)}\} \` for each
   iteration :math:`\strut i`.
#. Organise the forcing inputs from :math:` x_i^{full}` and simulated state
   variables into a joint time series :math:` (w_{t-1}^{(i)},a_t^{(i)}) \`
   for :math:`t=1,\ldots,T \`.

Additional Comments
-------------------

Since generating a single time series :math:` w_1^{(i)},\ldots,w_T^{(i)} \`
should be a relatively quick procedure, we can use a larger value of
:math::ref:`\strut N \` than might normally be considered in
`AltCoreDesign<AltCoreDesign>`. The main aim here is to
establish the boundaries of :math:` \\mathcal{X}_{single} \`, and so we
should check that any such assessment is stable for increasing values of
:math:`\strut N \`.
