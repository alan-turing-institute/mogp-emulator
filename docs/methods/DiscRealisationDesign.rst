.. _DiscRealisationDesign:

Discussion: Design for generating emulator realisations
=======================================================

Description and Background
--------------------------

The procedure page for simulating realisations of an emulator
(:ref:`ProcSimulationBasedInference<ProcSimulationBasedInference>`)
presents a method for drawing random realisations of an
:ref:`emulator<DefEmulator>`. In order to do this it is necessary to
sample values of the emulator's predictive distributions for a finite
set of input points
:math:`x^\prime_1,x^\prime_2,\ldots,x^\prime_{n^\prime}`. We discuss here
the choice of these points, the realisation design.

Discussion
----------

The underlying principle is that if we rebuild the emulator, augmenting
the original training data with the sampled values at the realisation
design points, then the resulting emulator would be sufficiently
accurate that it would have negligible uncertainty in its predictions.
To create a suitable realisation design, we need to decide how many
points it should have, and where in the input space those points should
be.

If we consider the second decision first, suppose we have decided on
:math:`n^\prime`. Then it is clear that the combined design of :math:`n`
original training design points and the :math:`n^\prime` realisation design
points must be a good design, since good design involves getting an
accurate emulator - see the alternatives page on training sample design
for the core problem (:ref:`AltCoreDesign<AltCoreDesign>`). For
instance, we might choose this design by generating random Latin
hypercubes (LHCs) of :math:`n^\prime` points and then choosing the
composite design (original training design plus LHC) to satisfy the
maximin criterion - see the procedure for generating an optimised Latin
hypercube design (:ref:`ProcOptimalLHC<ProcOptimalLHC>`).

To determine a suitable value for :math:`n^\prime` will typically involve a
degree of trial and error. If we generate a design with a certain
:math:`n^\prime` and the predictive variances are not all small, then we
need a larger sample!

Additional Comments
-------------------

This topic is under investigation in MUCM, and fuller advice may be
available in due course.
