.. _DefDesign:

Definition of Term: Design
==========================

In order to build an :ref:`emulator<DefEmulator>` for a
:ref:`simulator<DefSimulator>`, we need data comprising a :ref:`training
sample<DefTrainingSample>` of simulator runs. The training
sample design is the set of points in the space of simulator inputs at
which the simulator is run.

There are a number of related design contexts which arise in the
:ref:`MUCM<DefMUCM>` toolkit.

-  A validation sample design is a set of points in the input space at
   which additional simulator runs are made to
   :ref:`validate<DefValidation>` the emulator.

-  In the context of :ref:`calibrating<DefCalibration>` a simulator,
   we may design an observational study by specifying a set of points in
   the space of input values which we can control when making
   observations of the real-world process.

-  Where we have more than one simulator available, a design may specify
   sets of points in the input spaces of the various simulators.

-  For complex tasks using the emulator, a general solution is via
   simulated realisations of the output function :math:`f(\cdot)`. For this
   purpose we need a set of input configurations at which to simulate
   outputs, which we refer to as a realisation design.

Some general remarks on design options for the core problem are
discussed in the alternatives page on training sample design for the
core problem (:ref:`AltCoreDesign<AltCoreDesign>`).
