.. _DiscActiveInputs:

Discussion: Active and inactive inputs
======================================

Description and Background
--------------------------

:ref:`Simulators<DefSimulator>` usually have many inputs. When we
wish to build an :ref:`emulator<DefEmulator>` to represent the
simulator over some region of the input space, it is desirable to
minimise the dimensionality of that space (i.e. the number of inputs).
This dimension reduction is motivated by two considerations.

First, decreasing the number of inputs decreases the complexity of the
emulation, and in particular reduces the number of simulator runs that
we need in the :ref:`training sample<DefTrainingSample>`. If the
number of inputs is large, for instance more than 30, emulation may
become impractical. Difficulty may be experienced in estimating
:ref:`hyperparameters<DefHyperparameter>` (because there are now many
hyperparameters to estimate) and in doing other computations (because
the large training sample leads to large, possibly ill-conditioned,
matrix inversions).

The second motivation is that we often find in practice that most of the
inputs have only small influences on the output(s) of interest. If, when
we vary an input over the range for which we wish to build the emulator,
the output changes only slightly, then we can identify this as an
:ref:`inactive input<DefInactiveInput>`. If we can build the emulator
using only :ref:`active inputs<DefActiveInput>` then not only will
the emulation task become much simpler but we will also have an emulator
that represents the simulator accurately but without unnecessary
complexity.

Discussion
----------

Identifying active inputs
~~~~~~~~~~~~~~~~~~~~~~~~~

The process of identifying active inputs is called
:ref:`screening<DefScreening>` (or dimension reduction). Screening
can be thought of as an application of :ref:`sensitivity
analysis<DefSensitivityAnalysis>`, the active inputs being those
which have the highest values of an appropriate sensitivity measure.

A detailed discussion of screening, with procedures for recommended
methods, is available at the screening topic thread
(:ref:`ThreadTopicScreening<ThreadTopicScreening>`).

Excluding inactive inputs
~~~~~~~~~~~~~~~~~~~~~~~~~

There are two ways to exclude inactive inputs when building and using an
emulator.

Inactive inputs fixed
^^^^^^^^^^^^^^^^^^^^^

The first method is to fix all the inactive inputs at some specified
values. The fixed values may be arbitrary, since the notion of an
inactive input is that its value is unimportant. We are thereby limiting
the input space to that part in which the inactive inputs take these
fixed values and only the active inputs are varied.

In this approach, a :ref:`deterministic<DefDeterministic>` simulator
remains deterministic. It is a function only of the active inputs, and
the emulator only represents the simulator in this reduced input space.
Although the values of inactive inputs are unimportant, this does not
mean that varying the inactive inputs has no effect at all on the
output. So the emulator cannot strictly be used to predict the output of
the simulator at any other values of the inactive inputs than the chosen
fixed values.

Notice that the emulator is now built using a training sample in which
all the inactive variables are fixed at their chosen values in every
simulator run. The screening process will generally have required a
number of simulator runs in which all the inputs are varied, in order to
identify which are inactive. The fixed inactive inputs approach then
cannot use these runs as part of its training sample, so the training
sample involves a new set of simulator runs.

Inactive inputs ignored
^^^^^^^^^^^^^^^^^^^^^^^

The second method is to ignore the values of inactive inputs in training
data. The values of inactive inputs are allowed to vary, and so
simulator runs that were used for screening may be reusable for building
the emulator. However, now the simulator output is not a deterministic
function of the active inputs only.

We therefore model the output from the simulator as stochastic, equal to
a function of the active inputs (which we emulate) plus a random noise
term to account for the ignored values of the inactive inputs. Since the
inactive inputs have only a small effect on the output, the noise term
should be small, but it needs to be accounted for in the analysis.

When building the emulator, we only use the values of the active inputs.
The :ref:`GP<DefGP>` correlation function involves an added
:ref:`nugget<DefNugget>` term as discussed in the alternatives page
on emulator prior correlation function
(:ref:`AltCorrelationFunction<AltCorrelationFunction>`).

Prediction also needs to allow for the inactive inputs. Using the
emulator with the correlation function including the nugget term allows
us to predict the simulator outputs at any input configurations. The
emulator will ignore the inactive input values, but will allow for the
extra noise in predicting actual outputs via the nugget term. Removing
the nugget term (by setting the corresponding hyperparameter to zero)
will predict average values of the output for the given values of the
active inputs and averaged over the inactive inputs.

Additional Comments
-------------------

The choice of active inputs is a trade-off between emulating the
simulator as well as possible and achieving manageable and stable
computations. It also depends on context. For instance, we may initially
emulate the simulator using only a very small number of the most active
inputs in order to explore the simulator quickly. However, this choice
will entail a larger noise term (nugget) and if the emulator predictions
are not sufficiently precise we can refit the emulator using more active
inputs.

A similar strategy may be employed when calibrating a simulator using
real-world observations. Initial coarse emulation suffices to narrow the
search for plausible values of calibration inputs, but this is then
iteratively refined. At each stage, we need more precise emulation,
which may be achieved partly by larger training samples but may also
require the use of more active inputs.
