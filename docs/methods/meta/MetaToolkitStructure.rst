.. _MetaToolkitStructure:

Meta-pages: Toolkit Structure
=============================

Main threads
------------

The toolkit contains a large number of pages, and there are several ways
to use it. One would be just to browse - pick a page from this :ref:`page
list<MetaToolkitPageList>` and just follow connections - but we
have also created more structured entry points. The first of these is
the set of main threads. These will take the user through the complete
process of a :ref:`MUCM<DefMUCM>` application - designing and
building an :ref:`emulator<DefEmulator>` to represent a
:ref:`simulator<DefSimulator>`, and then using it to carry out tasks
such as :ref:`sensitivity analysis<DefSensitivityAnalysis>` or
:ref:`calibration<DefCalibration>`.

The simplest main threads are the core threads. There are two of these,
one for the :ref:`Gaussian Process<DefGP>` (GP, or fully
:ref:`Bayesian<DefBayesian>`) version of the core, and one for the
:ref:`Bayes Linear<DefBayesLinear>` (BL) version. The underlying
differences between these two approaches are discussed in the
alternatives page on Gaussian Process or Bayes Linear based emulator
(:ref:`AltGPorBLEmulator<AltGPorBLEmulator>`). In both cases, the
core thread deals with developing emulators for a single output of a
simulator that is :ref:`deterministic<DefDeterministic>`. Further
details of the core model are given in the threads themselves,
:ref:`ThreadCoreGP<ThreadCoreGP>` and
:ref:`ThreadCoreBL<ThreadCoreBL>`.

Further main threads deal with variations on the basic core model. In
principle, these threads will try to handle both GP and BL approaches.
However, there will be various components of these threads where the
relevant tools have only been developed fully for one of the two
approaches, and methods for the other approach will only be discussed in
general terms.

Other main threads will address generic extensions that apply to both
core and variant threads.

The main threads are currently planned to be as follows:

-  :ref:`ThreadCoreGP<ThreadCoreGP>` - the core model, dealt with by
   fully Bayesian, Gaussian Process, emulation
-  :ref:`ThreadCoreBL<ThreadCoreBL>` - the core model, dealt with by
   Bayes Linear emulation
-  :ref:`ThreadVariantMultipleOutputs<ThreadVariantMultipleOutputs>`
   - a variant of the core model in which we emulate more than one
   output of a simulator
-  :ref:`ThreadVariantDynamic<ThreadVariantDynamic>` - a special case
   of multiple outputs is when the simulator outputs are generated
   iteratively as a time series
-  :ref:`ThreadVariantTwoLevelEmulation<ThreadVariantTwoLevelEmulation>`
   - a variant in which a fast simulator is used to help build an
   emulator of a slow simulator
-  ThreadVariantMultipleSimulators - variant of the core model in which
   we emulate outputs from more than one related simulator, a special
   case of which is when the real world is regarded as a perfect
   simulator
-  ThreadVariantStochastic - variant of the core model in which the
   simulator output is :ref:`stochastic<DefStochastic>`
-  :ref:`ThreadVariantWithDerivatives<ThreadVariantWithDerivatives>`
   - variant of the core model in which we include derivative
   information
-  :ref:`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`
   - a variant that deals with modelling the relationship between the
   simulator outputs and the real-world process being simulated
-  :ref:`ThreadGenericMultipleEmulators<ThreadGenericMultipleEmulators>`
   - a thread showing how to combine independent emulators (for outputs
   of different simulators or different outputs of one simulator) to
   address tasks relating to combinations of the outputs being simulated
-  :ref:`ThreadGenericEmulateDerivatives<ThreadGenericEmulateDerivatives>`
   - a thread showing how to emulate derivatives of outputs
-  :ref:`ThreadGenericHistoryMatching<ThreadGenericHistoryMatching>`
   - a thread using observations of the real system to learn about the
   inputs of the model

It should be noted that simulators certainly exist in which we need to
take account of more than one of the variations. For instance, we may be
interested in multiple outputs which are also stochastic. Where the
relevant tools have been developed, they will be included through
cross-linkages between the main threads, or by additional main threads.

The toolkit is being released to the public in stages, so that only some
of the threads are currently available. Others will be included in
future releases.

Topic Threads
-------------

Another way to use the toolkit is provided through a number of topic
threads. Whereas the various main threads take the user through the
process of modelling, building an emulator and using that emulator to
carry out relevant tasks, a topic thread focusses on a particular aspect
of that process. For instance, a topic thread could deal with issues of
modelling, and would describe the core model, variants on it associated
with the other main threads, and go on to more complex variants for
which tools are as yet unavailable or are under development. Another
topic thread might consider the design of training samples for building
an emulator, or the task of sensitivity analysis.

Topic threads provide a technical background to their topics that is
common across different kinds of emulators. They allow toolkit users to
gain a more in-depth understanding and to appreciate relationships
between how the topic is addressed in different main threads. Whereas
the main threads are aimed at toolkit users who wish to apply the MUCM
tools, the topic threads will be intended more for researchers in the
field or for users who want to gain a deeper understanding of the tools.

Topic threads now available or under development include:

-  :ref:`ThreadTopicSensitivityAnalysis<ThreadTopicSensitivityAnalysis>`
-  :ref:`ThreadTopicScreening<ThreadTopicScreening>`
-  :ref:`ThreadTopicExperimentalDesign<ThreadTopicExperimentalDesign>`

Other Page Types
----------------

Apart from the Threads, the other pages of the Toolkit belong to one of
the following categories

-  Procedure - The description of an operation or an algorithm.
   Procedure pages should provide sufficient information to allow the
   implementation of the operation that is being described.
-  Discussion - Pages that discuss issues that may arise during the
   implementation of a method, or other optional details.
-  Alternatives - These pages present available options when building a
   specific part of an emulator (e.g. choosing a covariance function)
   and provide some guidance for doing the selection.
-  Definition - Definition of a term or a concept.
-  Example - A page that provides a worked example of a thread or
   procedure.
-  Meta - Any page that does not fall in one of the above categories,
   usually pages about the Toolkit itself.

Page types are identifiable by the start of the page name - Thread,
Proc, Disc, Alt, Def, Exam or Meta.
