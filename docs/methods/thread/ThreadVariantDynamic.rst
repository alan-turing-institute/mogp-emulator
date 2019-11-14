.. _ThreadVariantDynamic:

Thread: Dynamic Emulation
=========================

Overview
--------

This thread describes how to construct an
:ref:`emulator<DefEmulator>` for a :ref:`dynamic
simulator<DefDynamic>`. Here, we describe an approach based on
emulating what we call the :ref:`single step
function<DefSingleStepFunction>`. An alternative, though
potentially less practical approach is to build a multiple output
emulator using the techniques described in the variant thread for
analysing a simulator with multiple inputs
(:ref:`ThreadVariantMultipleOutputs<ThreadVariantMultipleOutputs>`).
For a comparison of the two methods see the alternatives page on dynamic
emulation approaches
(:ref:`AltDynamicEmulationApproach<AltDynamicEmulationApproach>`).

Readers should be familiar with the analysis of the core model, as
described in the relevant core thread
(:ref:`ThreadCoreGP<ThreadCoreGP>`), before continuing here. The task
of building the emulator of the single step function is an example of
:ref:`the core problem<DiscCore>` (though the multivariate extension
described in
:ref:`ThreadVariantMultipleOutputs<ThreadVariantMultipleOutputs>`
will be necessary in most cases). This thread does not currently
describe the analysis for dynamic simulators within the :ref:`Bayes
linear<DefBayesLinear>` framework, but methods will be added to
these pages as they are developed.

Simulator specifications and notation
-------------------------------------

We make the distinction between the **full simulator** and the
**simulator single step function**. The full simulator is the simulator
that we wish to analyse, and has the following inputs and outputs.

Full simulator inputs:

-  Initial conditions :math::ref:` w_0 \`.
-  A time series of external `forcing inputs<DefForcingInput>`
   :math:` a_1,\ldots,a_T \`.
-  Constant parameters :math:` \\phi \`.

Full simulator outputs:

-  A time series :math:` w_1,\ldots,w_T \`.

We refer to :math::ref:` w_t \` as the `state variable<DefStateVector>`
at time :math:`\strut t \`. Each of :math:` w_t, a_t \` and :math:` \\phi \` may
be scalar or vector-valued. The value of :math:`\strut T` may be fixed, or
varied by the simulator user. We represent the full simulator by the
function :math:` (w_1,\ldots,w_T)=f_{full}(w_0,a_1,\ldots,a_T,\phi) \`. We
define :math:` \\mathcal{X}_{full} \` to be the input region of interest
for the full simulator, with a point in :math:` \\mathcal{X}_{full} \`
represented by :math:`(w_0,a_1,\ldots,a_T,\phi) \`.

The full simulator produces the output :math:` (w_1,\ldots,w_T)` by
iteratively applying a function of the form :math:` w_t=f(w_{t-1},a_t,\phi)
\`, with :math:` f(.) \` known as the single step function. Inputs and
outputs for the single step functions are therefore as follows.

Single step function inputs:

-  The current value of the state variable :math:` w_{t-1} \` .
-  The associated forcing input :math:` a_{t} \`.
-  Constant parameters :math:` \\phi \`.

Single step function output:

-  The value of the state variable at the next time point :math:` w_{t}`.

**In this thread, we require the user to be able to run the single step
function at any input value**. Our method for emulating the full
simulator is based on emulating the single step function. We define :math:`
\\mathcal{X}_{single} \` to be the input region of interest for the
single step function, with a point in :math:` \\mathcal{X}_{single} \`
represented by :math:`x=(w,a,\phi) \`.

:ref:`Training inputs<DefTrainingSample>` for the single step
emulator are represented by :math:`D=\{x_1,\ldots,x_n\}`. Note that the
assignment of indices :math:`1,\ldots,n` to the training inputs is
arbitrary, so that there is no relationship between :math:`x_i` and
:math:`x_{i+1}`.

Emulating the full simulator: outline of the method
---------------------------------------------------

#. Build the single step emulator: an emulator of the single step
   function.
#. Iterate the single step emulator to randomly sample :math:`
   w_1,\ldots,w_T \` given a full simulator input :math:` (w_0,
   a_1,\ldots,a_T,\phi) \`. Repeat for different choices of full
   simulator input within :math:` \\mathcal{X}_{full}`.
#. Inspect the distribution of sampled trajectories :math:` w_1,\ldots,w_T
   \` obtained in step 2 to determine whether the training data for the
   single step emulator are adequate. If necessary, obtain further runs
   of the single step function and return to step 1.

Step 1: Build an emulator of the single step function :math:` w_t=f(w_{t-1},a_t,\phi) \`
-------------------------------------------------------------------------------------

This can be done following the procedures in
:ref:`ThreadCoreGP<ThreadCoreGP>`, (or
:ref:`ThreadVariantMultipleOutputs<ThreadVariantMultipleOutputs>` if
the state variable is a vector). Two issues to consider in particular
are the choice of mean function, and the design for the training data.

1) Choice of single step emulator mean function

(See the alternatives page on emulator prior mean function
(:ref:`AltMeanFunction<AltMeanFunction>`) for a general discussion of
the choice of mean function). The user should think carefully about the
relationship between :math:` w_t \` and :math:` (w_{t-1},a_t,\phi) \`. The
state variable at time :math:`\strut t \` is likely to be highly correlated
with the state variable at time :math:`\strut t-1 \`, and so the constant
mean function is unlikely to be suitable.

2) Choice of single step emulator :ref:`design<DefDesign>`

Design points for the single step function can be chosen following the
general principles in the alternatives page on training sample design
for the core problem (:ref:`AltCoreDesign<AltCoreDesign>`). However,
there is one feature of the dynamic emulation case that is important to
note: we can get feedback from the emulator to tell us if we have
specified the input region of interest :math:` \\mathcal{X}_{single} \`
appropriately. If the emulator predicts that :math:` w_t \` will move
outside the original design space for some value of :math:`\strut t \`,
then we will want to predict :math:` f(w_t,a_{t+1},\phi) \` for an input
:math:` (w_t,a_{t+1},\phi) \` outside our chosen :math:` \\mathcal{X}_{single}
\`. Alternatively, we may find that the state variables are predicted
to lie in a much smaller region than first thought, so that some
training data points may be wasted. Hence it is best to choose design
points sequentially; we choose a first set based on our initial choice
of :math:` \\mathcal{X}_{single} \`, and then in steps 2 and 3 we identify
whether further training runs are necessary.

We have not yet established how many training runs are optimal at this
stage (or the optimal proportion of total training runs to be chosen at
this stage), though this will depend on how well :math:`
\\mathcal{X}_{single} \` is chosen initially. In the application in
Conti et al (2009), with three state variable and two forcing inputs, we
found the choice of 30 initial training runs and 20 subsequent training
runs to work well.

As we will need to iterate the single step emulator over many time
steps, we emphasise the importance of
:ref:`validating<DefValidation>` the emulator, using the procedure
page on validating a Gaussian process emulator
(:ref:`ProcValidateCoreGP<ProcValidateCoreGP>`).

Step 2: Iterate the single step emulator over the full simulator input region of interest
-----------------------------------------------------------------------------------------

We now iterate the single step emulator to establish whether the initial
choice of design points :math:`\strut D \` is suitable . We do so by
choosing points from :math:` \\mathcal{X}_{full} \`, and iterating the
single step emulator given the specified :math:` (w_0,a_1,\ldots,a_T,\phi)
\:ref:`. A procedure for doing so is described in
`ProcExploreFullSimulatorDesignRegion<ProcExploreFullSimulatorDesignRegion>`.

Step 3: Inspect the samples from step 2 and choose additional training runs
---------------------------------------------------------------------------

Following step 2, we have now have samples
:math:`(w_{t-1}^{(i)},a_t^{(i)},\phi^{(i)})` for :math:`t=1,\ldots,T` and
:math:`i=1,\ldots,N`. These samples give us a revised assessment of :math:`
\\mathcal{X}_{single} \`, as the simulation in step 2 has suggested
that we wish to predict :math:`f(.)` at each point
:math:`(w_{t-1}^{(i)},a_t^{(i)},\phi^{(i)})`. We now compare this
collection of points with the original training design :math:`\strut D` to
see if additional training data are necessary. If further training data
are obtained, we re-build the single step emulator and return to step 2.

We do not currently have a simple procedure for choosing additional
training data, as the shape of :math:` \\mathcal{X}_{single} \` implied by
the sampled :math:`(w_{t-1}^{(i)},a_t^{(i)},\phi^{(i)})` is likely to be
quite complex. A first step is to compare the marginal distribution of
each state vector element in the sample with the corresponding elements
in the training design :math:`\strut D`, as this may reveal obvious
inadequacies in the training data. It is also important to identify the
time :math:`\strut t^*` when a sampled time series
:math:`(w_{t-1}^{(i)},a_t^{(i)},\phi^{(i)})` for :math:`t=1,\ldots,T` *first*
moves outside the design region. The single step emulator may validate
less well the further the input moves from the training data, so that
samples :math:`(w_{t-1}^{(i)},a_t^{(i)},\phi^{(i)})` for :math:`t>t^*` may be
less 'reliable'.

Tasks
-----

Having obtained a satisfactorily working emulator, the MUCM methodology
now enables efficient analysis of a number of tasks that regularly face
users of simulators.

Prediction
~~~~~~~~~~

The simplest of these tasks is to use the emulator as a fast surrogate
for the simulator, i.e. to predict what output the simulator would
produce if run at a new point in the input space. We have two methods
for doing this: the exact simulation method described in the procedure
page
:ref:`ProcExactIterateSingleStepEmulator<ProcExactIterateSingleStepEmulator>`
(used in step 2 in the construction of the emulator) and an
approximation described in the procedure page
:ref:`ProcApproximateIterateSingleStepEmulator<ProcApproximateIterateSingleStepEmulator>`
which can be faster to implement. (See the alternatives page
:ref:`AltIteratingSingleStepEmulators<AltIteratingSingleStepEmulators>`
for a comparison of the two).

Uncertainty analysis
~~~~~~~~~~~~~~~~~~~~

:ref:`Uncertainty analysis<DefUncertaintyAnalysis>` is the process of
predicting the simulator output when one or more of the inputs are
uncertain. The procedure page on uncertainty analysis for dynamic
emulators (:ref:`ProcUADynamicEmulator<ProcUADynamicEmulator>`)
explains how this is done.

Additional Comments, References, and Links
------------------------------------------

Methods for other tasks such as :ref:`sensitivity
analysis<DefSensitivityAnalysis>` will be added to these pages
as they are developed.

The methodology described here is based on

Conti, S., Gosling, J. P., Oakley, J. E. and O'Hagan, A. (2009).
Gaussian process emulation of dynamic computer codes. Biometrika 96,
663-676.
