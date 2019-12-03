.. _DefModelBasedDesign:

Definition of Term: Model Based Design
======================================

Several toolkit operations call for a set of points to be specified in
the :ref:`simulator’s<DefSimulator>` space of inputs, and the choice
of such a set is called a design. For instance, to create an
:ref:`emulator<DefEmulator>` of the simulator we need a set of
points, the :ref:`training sample<DefTrainingSample>`, at which the
simulator is to be run to provide data in order to build the emulator.
The question can then be posed, what constitutes a good, or an optimal,
design? We cannot answer without having some idea of what the results
might be of different choices of design, and in general the results are
not known at the time of creating the design – for instance, when
choosing a training sample we do not know what the simulator outputs
will be at the design points, and so do not know how good the resulting
emulator will be. General-purpose designs rely on qualitative generic
features of the problem and aim to be good for a wide variety of
situations.

In general we should only pick an optimal design, however, if we can
specify (a) an optimality criterion (see the alternatives page
:ref:`AltOptimalCriteria<AltOptimalCriteria>` for some commonly used
criteria) and (b) a model for the results (e.g. simulator outputs) that
we expect to observe from any design. If one has a reliable model for a
prospective emulator, perhaps derived from knowledge of the physical
process being modelled by the simulator, then an optimal design may give
a more focused experiment. For these reasons we may use the terms
"model-based design" which is short for "model based optimal design" .

For optimal design the choice of design points becomes an optimisation
problem; e.g. minimise, over the choice of the design, some special
criterion. Since the objective is very often to produce an emulator that
is as accurate as possible, the optimality criteria are typically
phrased in terms of posterior variances (in the case of a fully
:ref:`Bayesian<DefBayesian>` emulator, or adjusted variances in the
case of a :ref:`Bayes linear<DefBayesLinear>` emulator).

There are related areas where choice of sites at which to observe or
evaluate a function are important. Optimisation is itself such an area
and one can also mention search problems involving "queries" in computer
science, numerical integration and approximation theory.
