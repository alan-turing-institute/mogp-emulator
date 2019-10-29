.. _DefDynamic:

Definition of Term: Dynamic simulator
=====================================

A dynamic :ref:`simulator<DefSimulator>` models a real-world process
that evolves over time (or sometimes in space, or both time and space).
Its inputs typically include an initial :ref:`state
vector<DefStateVector>` that defines the initial state of the
process, as well as other fixed inputs governing the operation of the
system. It often also has external :ref:`forcing
inputs<DefForcingInput>` that provide time-varying impacts on
the system. The dynamic nature of the simulator means that as it runs it
updates the state vector on a regular time step, to model the evolving
state of the process.

*Example:* A simulator of a growing plant will have a state vector
describing the sizes of various parts of the plant, the presence of
nutrients and so on. Fixed inputs parameterise the biological processes
within the plant, while forcing inputs may include the temperature,
humidity and intensity of sunlight at a given time point.

At time :math:`\strut t`, the simulator uses the time :math:`\strut t` value
of its state vector, the fixed inputs and the time :math:`\strut t` values
of the forcing inputs to compute the value of its state vector at time
:math:`t+\Delta t` (where :math:`\Delta t` is the length of the simulator's
time step). It then repeats this process using the new state vector, the
fixed inputs and new forcing input values to move forward to time
:math:`t+2\Delta t`, and so on.
