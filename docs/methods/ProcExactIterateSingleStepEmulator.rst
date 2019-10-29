.. _ProcExactIterateSingleStepEmulator:

Procedure: Iterate the single step emulator using an exact simulation approach
==============================================================================

Description and Background
--------------------------

This page is concerned with task of :ref:`emulating<DefEmulator>` a
:ref:`dynamic simulator<DefDynamic>`, as set out in the variant
thread on dynamic emulation
(:ref:`ThreadVariantDynamic<ThreadVariantDynamic>`).

We have an emulator for the :ref:`single step
function<DefSingleStepFunction>` :math:` w_t=f(w_{t-1},a_t,\phi)
\:ref:`, and wish to predict the full time series :math:` w_1,\ldots,w_T \` for
a specified initial `state variable<DefStateVector>` :math::ref:` w_0`,
time series of `forcing variables<DefForcingInput>` :math:`
a_1,\ldots,a_T \` and simulator parameters :math:` \\phi`. It is not
possible to derive analytically a distribution for :math::ref:` w_1,\ldots,w_T
\` if :math:` f(.) \` is modelled as a `Gaussian Process<DefGP>`,
so here we use simulation to sample from the distribution of :math:`
w_1,\ldots,w_T \`. We simulate a large number of series :math:`
w_1^{(i)},\ldots,w_T^{(i)} \` for :math:`i=1,\ldots,R \`, and then use the
simulated series for making predictions and reporting uncertainty.

Inputs
------

-  An emulator for the single step function :math::ref:` w_t=f(w_{t-1},a_t,\phi)
   \`, formulated as a GP or `t-process<DefTProcess>`
   conditional on hyperparameters, :ref:`training
   inputs<DefTrainingSample>` :math:`D=\{x_1,\ldots,x_n\} \` and
   training outputs :math:`f(D) \`.
-  A set :math:`\{\theta^{(1)},\ldots,\theta^{(s)}\} \` of emulator
   hyperparameter values.
-  The initial value of the state variable :math:` w_0`.
-  The values of the forcing variables :math:` a_1,\ldots,a_T \`.
-  The values of the simulator parameters :math:` \\phi`.
-  Number of realisations :math:`\strut R \` required.

For notational convenience, we suppose that :math:`\strut R\le s`. For
discussion of the choice of :math:`\strut R`, including the case :math:`R>s`,
see the discussion page on Monte Carlo estimation
(:ref:`DiscMonteCarlo<DiscMonteCarlo>`).

Outputs
-------

-  A set of simulated state variable time series :math:`
   w_1^{(i)},\ldots,w_T^{(i)} \` for :math:`i=1,\ldots,R \`

Procedure
---------

Note: methods for simulating outputs from Gaussian process emulators are
described in the procedure page
:ref:`ProcOutputSample<ProcOutputSample>`.

A single time series :math:` w_1^{(i)},\ldots,w_T^{(i)} \` can be generated
as follows.

1) Using the emulator with hyperparameters :math:` \\theta^{(i)}`, sample
from the distribution of :math:` f(w_0,a_1,\phi) \` to obtain
:math:`w_1^{(i)}`. Then iterate the following steps 2-4 for :math:`
t=1,\ldots,T -1`.

2) Construct a new training dataset of inputs :math:` (D, D^{(i,t)})`,
where :math:`D^{(i,t)}=\{(w_0,a_1,\phi),(w_1^{(i)},
,a_2,\phi),\ldots,(w_{t-1}^{(i)},a_t,\phi)\}` and outputs
:math:`\{f(D),w_1^{(i)},\ldots,w_t^{(i)}\}`. To clarify, training inputs
and outputs are paired as follows:

:math:`\mbox{Training inputs: }\left(\begin{array}{c}x_1\\\ \\vdots \\\\ x_n
\\\\ (w_0,a_1,\phi) \\\\ (w_1^{(i)},a_2,\phi)\\\ \\vdots \\\\
(w_{t-1}^{(i)},a_t,\phi)\end{array}\right)\mbox{\hspace{2cm}}
\\mbox{Training outputs: }\left(\begin{array}{c}f(x_1)\\\ \\vdots \\\\
f(x_n) \\\\ w_1^{(i)} \\\\ w_2^{(i)}\\\vdots \\\\
w_t^{(i)}\end{array}\right)`

3) Re-build the single step emulator given the new training data defined
in step 2. It may be necessary to thin the new training data first
before building the emulator. The set of inputs :math:` (D,D^{(i,t)})` may
contain points close together, which can make inversion of :math:`
A=c\{(D,D^{(i,t)}),(D,D^{(i,t)})\} \` difficult. See discussion in
Additional Comments.

4) Sample from the distribution of :math:` f(w_t^{(i)},a_{t+1},\phi) \` to
obtain :math:`w_{t+1}^{(i)}`

The whole process is repeated to obtain :math:` R \` simulated time series
:math:` w_1^{(i)},\ldots,w_T^{(i)} \` for :math:`i=1,\ldots R \`. The sample
:math:` w_1^{(i)},\ldots,w_T^{(i)} \` for :math:`i=1,\ldots R \` is a sample
from the joint distribution of :math:` w_1,\ldots,w_T \` given the emulator
training data and :math:` w_0,a_1,\ldots,a_T , \\phi`.

Additional Comments
-------------------

As commented in step 3, computational difficulties can arise if the
training set of inputs :math:` (D,D^{(i,t)})` contains inputs that are too
close together. This is likely to occur, as :math:`(w_{t+1}^{(i)},a_{t+2})`
is likely to be close to :math:`(w_{t}^{(i)},a_{t+1})`. This is problem is
not unique to the use of dynamic emulators, and is discussed in the page
on computational issues in building a GP emulator
(:ref:`DiscBuildCoreGP<DiscBuildCoreGP>`). A strategy that has been
used with some success for this procedure is to consider the emulator
variance of :math:` f(w_t^{(i)},a_{t+1},\phi) \` given training inputs :math:`
(D,D^{(i,t)})` and outputs :math:`\{f(D), w_1^{(i)},\ldots,w_t^{(i)}\}`,
and only add the new training input :math:`(w_t^{(i)},a_{t+1},\phi) \` and
associated output to the training data at iteration :math:`t+1` if the
variance of :math:` f(w_t^{(i)},a_{t+1},\phi) \` is sufficiently large. If
the variance is very small, so that the emulator already 'knows' the
value of :math:` f(w_t^{(i)},a_{t+1},\phi) \`, then adding this point to
the training data will little effect on the distribution of :math:` f(.)
\`.
