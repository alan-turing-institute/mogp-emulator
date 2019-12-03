.. _DefMultilevelEmulation:

Definition of Term: Multilevel Emulation
========================================

Multilevel emulation (or *multiscale emulation*) is the application of
:ref:`emulation<DefEmulator>` methodology to problems where we have
two or more versions of the same :ref:`simulator<DefSimulator>` that
produce outputs at different levels of accuracy or resolution.
Typically, the lower-accuracy simulators are less expensive to evaluate
than the high accuracy models. Multilevel emulation seeks to supplement
the restricted amount of information on the most-accurate model, with
information gained from larger numbers of evaluations of lower-accuracy
simulators to improve the performance of the final emulator.
