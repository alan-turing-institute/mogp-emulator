.. _DiscCore:

Discussion: The core problem
============================

Description and Background
--------------------------

The :ref:`MUCM<DefMUCM>` toolkit is
:ref:`structured<MetaToolkitStructure>` in various ways, one of which
is via a number of threads that take the user through building and using
various kinds of :ref:`emulator<DefEmulator>`. The core thread
concerns the simplest kind of emulator, while variant threads build on
the methods in the core thread to address more complex problems.

The situation to which the core thread applies is called the core
problem, and is characterised as follows:

-  We are only concerned with one :ref:`simulator<DefSimulator>`.
-  The simulator only produces one output, or (more realistically) we
   are only interested in one output.
-  The output is :ref:`deterministic<DefDeterministic>`.
-  We do not have observations of the real world process against which
   to compare the simulator.
-  We do not wish to make statements about the real world process.
-  We cannot directly observe derivatives of the simulator.

This discussion page amplifies on this brief specification, and
indicates how each of these requirements might be relaxed in the variant
threads.

Discussion
----------

We consider each of the core problem requirements in turn.

One simulator
~~~~~~~~~~~~~

Practical problems concerning the use of
:ref:`simulators<DefSimulator>` may involve more than one simulator,
in which case we need to build several connected emulators.

An example of one such situation which often arises is if we wish to use
climate simulation to tell us about the consequences of future climate
change, for then we have a number of climate simulators available. We
may wish to use several such simulators in order to contrast their
predictions, or to combine them to make some kind of average prediction.

Another situation in which multiple simulators are of interest is when a
single simulator may be implemented using different levels of detail.
For instance, a simulator of an oilfield will represent the geology of
the field in terms of blocks of rock, each block having its own
properties (such as permeability). If it is run using a small number of
very large blocks, the simulator will run quickly but will be a poor
representation of the underlying real-world oilfield. It can also be run
with an enormous number of very small blocks, which will represent the
real oilfield much more accurately but the computer program will take
many days to run. It is then often of interest to combine a few runs of
the accurate simulator with a larger number of runs of a coarser
simulator. A version of this problem is addressed in the variant thread
for two level emulation of the core model using a fast approximation
(:ref:`ThreadVariantTwoLevelEmulation<ThreadVariantTwoLevelEmulation>`).

In the core problem, we suppose that we are only interested in one
simulator. In a future release of the toolkit we intend to introduce a
variant thread ThreadVariantMultipleSimulators describing how to handle
the emulation of multiple simulators.

One output
~~~~~~~~~~

Simulators almost invariably produce many outputs. For example, a
climate simulator may output various climate properties, such as
temperature and rainfall, at each point in a grid covering the
geographical region of interest and at many points in time. We will
often wish to address questions that relate to more than one output.

However, for the core problem we suppose that interest focuses on a
single output, so that we only wish to emulate that one output. The
variant thread for the analysis of a simulator with multiple outputs
using Gaussian process methods
(:ref:`ThreadVariantMultipleOutputs<ThreadVariantMultipleOutputs>`)
presents ways to handle the emulation of multiple outputs. A special
case of a dynamic simulator producing a time series of outputs is
considered in the variant thread for dynamic emulation
(:ref:`ThreadVariantDynamic<ThreadVariantDynamic>`).

Note that the restriction to a single output is weakened by the fact
that we can be flexible about what we define a simulator output to be.
For instance, a simulator may produce temperature forecasts for each day
in a year, whereas we are interested in the mean temperature for that
year. Whilst the mean temperature is not actually produced as an output
by the simulator we can readily compute it from the actual outputs. We
can formally treat the combination of the simulator with the
post-processing operation of averaging the temperatures as itself a
simulator that outputs mean temperature. In general, the output of
interest may be any function of the simulator's actual outputs.

Deterministic
~~~~~~~~~~~~~

Simulators can be :ref:`deterministic<DefDeterministic>` or
:ref:`stochastic<DefStochastic>`. A deterministic emulator returns
the same output when we run it again at the same inputs, and so the
output may be regarded as a well-defined function of the inputs.

The core problem supposes that the output which we wish to emulate is
deterministic. In a future release of the toolkit we intend to introduce
a variant thread ThreadVariantStochastic concerned with extending the
core methods to stochastic simulators.

No real-world observations
~~~~~~~~~~~~~~~~~~~~~~~~~~

An important activity for many users of simulators is to compare the
simulator outputs with observations of the real-world process being
simulated. In particular, we often wish to use such observations for
:ref:`calibration<DefCalibration>` of the simulator, i.e. to learn
which values of certain inputs are the best ones to use in the
simulator.

The core problem assumes that we do not have real-world observations
that we wish to apply to test or calibrate the simulator. If we do have
observations of the real world to consider then we must build a
representation of the relationship between the simulator and the real
system - e.g. how good is the simulator? This can be handled by the same
techniques as are used for multiple simulators, by thinking of the real
system as a perfect simulator. Hence the appropriate thread for this
case will be ThreadVariantMultipleSimulators.

No real-world statements
~~~~~~~~~~~~~~~~~~~~~~~~

A related assumption in the core problem is that we do not wish to make
statements about the real-world system. Of course, we very often *do*
want to make such statements, and indeed the whole purpose of using
simulators is naturally seen as to help us to predict, understand or
control the real system. However, to make such statements we must again
build a representation of how the simulator relates to reality, and so
this will again be handled in ThreadVariantMultipleSimulators.

Although the core problem allows us only to make statements about the
simulator, not reality, we can predict what outputs the simulator would
produce at different input configurations and we can analyse in various
ways how the simulator responds to uncertain inputs. These are important
tasks in their own right.

No observations of derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will build the emulator using a training set of runs of the
simulator, so that we know the output values that are obtained with the
input configurations in the training data. Some simulators are able to
deliver not just the output in question but also the (partial)
derivatives of that output with respect to one or more (or all) of the
inputs. This is often done by the creation of what is called an adjoint
to the simulator.

The core problem assumes that we do not have such derivatives available.
Obviously, if we had some derivatives in addition to the other training
data, we should be able to create a more accurate emulator. In a future
release of the toolkit this is expected to be dealt with in a variant
thread ThreadVariantDerivatives.

Additional Comments
-------------------

The ways that the toolkit is structured, and in particular the threads,
are discussed in the :ref:`Toolkit structure<MetaToolkitStructure>`
page.
