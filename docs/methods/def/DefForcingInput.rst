.. _DefForcingInput:

Definition of Term: Forcing Input
=================================

A :ref:`dynamic<DefDynamic>` :ref:`simulator<DefSimulator>` models
a process that is evolving, usually in time. Many dynamic simulators
have forcing inputs that specify external influences on the process that
vary in time. In order to simulate the evolution of the process at each
point in time, the simulator must take account of the values of the
forcing inputs at that time point.

*Example:* A simulator of water flow through a river catchment will have
forcing inputs that specify the rainfall over time. Humidity may be
another forcing input, since it will govern the rate at which water in
the catchment evaporates.
