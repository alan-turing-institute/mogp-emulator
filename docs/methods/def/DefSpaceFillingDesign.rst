.. _DefSpaceFillingDesign:

Definition of Term: Space filling design
========================================

Several toolkit operations call for a set of points to be specified in
the :ref:`simulator<DefSimulator>` space of inputs, and the choice of
such a set is called a design. Space-filling designs are a class of a
simple general purpose designs giving sets of points, or :ref:`training
samples<DefTrainingSample>` at which the simulator is run to
provide data to build the emulator. They spread points in an
approximately uniform way over the design space. They could be any
design of the following types.

-  a design based on restricted sampling methods.
-  a design based on measures of distance between points.
-  a design known to have low "discrepancy" interpreted as a formal
   measure of the closeness to uniformity.
-  a design that is a hybrid of, or variation on, these designs.
