.. _DefDataAssimilation:

Definition of Term: Data Assimilation
=====================================

Data assimilation is a term that is widely used in the context of using
observations of the real-world process to update a
:ref:`simulator<DefSimulator>`. It generally applies to a
:ref:`dynamic<DefDynamic>` simulator that models a real-world process
that is evolving in time. At each time step, the simulator simulates the
current state of the system as expressed in the :ref:`state
vector<DefStateVector>`. In data assimilation, observation of
the real-world process at a given time point is used to learn about the
true status of the process and hence to adjust the state vector of the
simulator. Then the simulator's next time step starts from the adjusted
state vector, and should therefore predict the system state better at
subsequent time steps.

Data assimilation is thus also a dynamic process. It is similar to
:ref:`calibration<DefCalibration>` in the sense that it uses
real-world observations to learn about simulator parameters, but the
term calibration is generally applied to imply a one-off learning about
fixed parameters/inputs.
