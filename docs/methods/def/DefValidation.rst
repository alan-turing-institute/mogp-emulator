.. _DefValidation:

Definition of Term: Validation
==============================

An :ref:`emulator<DefEmulator>` for a
:ref:`simulator<DefSimulator>` is a statistical representation of our
knowledge about the simulator. In particular, it enables us to predict
what outputs the simulator would produce at any given configurations of
input values. The process of validation consists of checking whether the
actual simulator outputs at those input configurations are consistent
with the statistical predictions made by the emulator.

In conjunction with a comparable statistical specification of knowledge
about how the simulator relates to the real-world process that it is
intended to represent, we can use an emulator to make statistical
predictions of the real-world process. These can then also be validated
by comparison with the corresponding real-world values.

The validation of a Gaussian process emulator is described in the
procedure for validating a Gaussian process emulator
(:ref:`ProcValidateCoreGP<ProcValidateCoreGP>`).
