.. _AltIteratingSingleStepEmulators:

Alternatives: Iterating single step emulators
=============================================

Overview
--------

This page is concerned with task of :ref:`emulating<DefEmulator>` a
:ref:`dynamic simulator<DefDynamic>`, as set out in the variant
thread for dynamic emulation
(:ref:`ThreadVariantDynamic<ThreadVariantDynamic>`).

We have an emulator for the :ref:`single step
function<DefSingleStepFunction>` :math:` w_t=f(w_{t-1},a_t,\phi) \`
in a dynamic simulator, and wish to iterate the emulator to obtain the
distribution of :math::ref:` w_1,\ldots,w_T \` given :math:` w_0,a_1,\ldots,a_T \`
and :math:` \\phi \`. For a `Gaussian Process<DefGP>` emulator, it
is not possible to do so analytically, and so we consider two
alternatives: a simulation based approach and an approximation based on
the normal distribution.

The Nature of the Alternatives
------------------------------

#. Recursively simulate random outputs from the single step emulator,
   adding the simulated outputs to the training data at each iteration
   to obtain an exact draw from the distribution of :math:` w_1,\ldots,w_T
   \`
   (:ref:`ProcExactIterateSingleStepEmulator<ProcExactIterateSingleStepEmulator>`).
#. Approximate the distribution of :math:` w_t=f(w_{t-1},a_t,\phi) \` at
   each time step with a normal distribution. The mean and variance of
   :math::ref:` w_t \` for each :math:` t \` are computed recursively.
   (`ProcApproximateIterateSingleStepEmulator<ProcApproximateIterateSingleStepEmulator>`).

Choosing the Alternatives
-------------------------

The simulation method can be computationally intensive, but has the
advantage of producing exact draws from the distribution of :math:`
w_1,\ldots,w_T \`. The approximation method is computationally faster
to implement, and so is to be preferred if the normal approximation is
sufficiently accurate. The approximation is likely to be accurate when

-  uncertainty about :math:`f(.)` is small, and
-  the simulator is approximately linear over any small part of the
   input space.

As always, it is important to validate the emulator (see the procedure
page on validating a GP emulator
(:ref:`ProcValidateCoreGP<ProcValidateCoreGP>`)), but particular
attention should be paid to emulator uncertainty in addition to emulator
predictions.

Additional comments
-------------------

If the approximation method is to be used for many different choices :math:`
w_0,a_1,\ldots,a_T \` and :math:` \\phi \`, it is recommended that both
methods are tried for a subset of these choices, to test the accuracy of
the approximation.
