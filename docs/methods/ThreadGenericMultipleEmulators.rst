.. _ThreadGenericMultipleEmulators:

Thread: Generic methods for combining two or more emulators
===========================================================

Overview
--------

In this thread we consider the situation in which two or more
independent :ref:`emulators<DefEmulator>` have been built. Each
emulates a single output of a :ref:`simulator<DefSimulator>`, and we
assume that they all have the same set of inputs. This situation will
usually arise when the emulators are emulating different outputs from
the same simulator - see the alternatives page on approaches to
emulating multiple outputs
(:ref:`AltMultipleOutputsApproach<AltMultipleOutputsApproach>`). It
may also arise when they are emulating outputs from different
simulators, and here the requirement for all the emulators to have the
same set of inputs may be met by assembling all the inputs for the
different simulators into a single combined set (so that an individual
emulator formally has this pooled set as its inputs although it only
depends on a subset of them).

The emulators have been built separately and are supposed to be
independent. Our objective in this thread is to combine the emulators to
address tasks which do not relate to a single output. Here are some
examples.

-  A simulator of an engine outputs the fuel consumption in each minute
   during a loading test. In order to estimate consumption at different
   time points we have considered the alternative emulation approaches
   discussed in
   :ref:`AltMultipleOutputsApproach<AltMultipleOutputsApproach>` and
   decided to build separate, independent emulators. However, we are
   also interested in the cumulative consumption at the end of the first
   hour, which is the sum of the separate minute-by-minute consumption
   outputs.
-  A simulator outputs the average electricity demand per capita in a
   city on a given day. Its inputs include weather conditions and
   demographic data about the city. Separate emulators are built for a
   number of different cities, in which the demographic data for each
   city are fixed and the emulator takes just the weather variables as
   inputs. (It is common to use emulation in this way, to emulate the
   simulator output for a specific application in which the simulator
   inputs specific to that application are fixed.) We now wish to
   estimate total electricity demand over these cities on that day,
   which is the sum of the various per capita demands weighted by the
   city populations. Note that although we may be able to assume that
   the temperature is the same in all the cities on that day, the
   rainfall and cloud cover may vary between cities, so we would need to
   use the device of pooling inputs.

This thread therefore deals with how to combine two or more independent
emulators to address such questions.

The emulators
-------------

We consider emulators built using any of the methods described in other
threads, for instance the core threads
:ref:`ThreadCoreGP<ThreadCoreGP>` and
:ref:`ThreadCoreBL<ThreadCoreBL>`. Therefore they may be fully
:ref:`Bayesian<DefBayesian>` :ref:`Gaussian process<DefGP>` (GP)
emulators or :ref:`Bayes linear<DefBayesLinear>` (BL) emulators. The
two forms of emulator are different, so this thread addresses separate
issues for combining GP or BL emulators where appropriate.

-  A GP emulator is in the form of an updated GP (or a related process
   called a :ref:`t-process<DefTProcess>`) conditional on
   hyperparameters, plus one or more sets of representative values of
   those hyperparameters.
-  A BL emulator consists of a set of updated means, variances and
   covariances for the simulator output.

We suppose therefore that we are interested in one or more functions of
the simulator outputs. These functions may be linear, as in the above
examples, or nonlinear.

We also suppose that the emulators have been
:ref:`validated<DefValidation>` as described in the relevant thread.

Tasks
-----

Having obtained a working emulator, the MUCM methodology now enables
efficient analysis of a number of tasks that regularly face users of
simulators. In these tasks, the procedure may simply reduce to combining
the results of performing the corresponding task on each emulator
separately; however, some tasks require additional computations.

Prediction
~~~~~~~~~~

The simplest task for an emulator is as a fast surrogate for the
simulator, i.e. to predict what output the simulator would produce if
run at a new point in the input space. Procedures for predicting one or
more functions of independently emulated outputs are set out in the
procedure page
:ref:`ProcPredictMultipleEmulators<ProcPredictMultipleEmulators>`.

Uncertainty analysis
~~~~~~~~~~~~~~~~~~~~

:ref:`Uncertainty analysis<DefUncertaintyAnalysis>` is the process of
predicting the simulator output when one or more of the inputs are
uncertain. The procedure page
:ref:`ProcUAMultipleEmulators<ProcUAMultipleEmulators>` explains how
this is done for functions of independently emulated outputs.

Additional Comments, References, and Links
------------------------------------------

Other tasks that can be addressed include sensitivity analysis (studying
how outputs are influenced by individual inputs), optimisation (finding
the values of one or more inputs that will minimise or maximise the
output) and decision analysis (finding an optimal decision according to
a formal description of utilities). We expect to add procedures for
these tasks for multiple emulators in due course.
