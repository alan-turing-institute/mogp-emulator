.. _DefCalibration:

Definition of Term: Calibration
===============================

The inputs to a :ref:`simulator<DefSimulator>` that are the correct
or best values to use to predict a particular real-world system are very
often uncertain.

*Example*: An atmospheric dispersion simulator models the way that
pollutants are spread through the atmosphere when released. Its outputs
concern how much of a pollutant reaches various places or susceptible
targets. In order to use the simluator to predict the effects of a
specific pollutant release, we need to specif the appropriate input
values for the location of the release, the quantity released, the wind
speed and direction and so on. All of these may in practice be
uncertain.

If we can make observations of the real-world system, then we can use
these to learn about those uncertain inputs. A crude way to do this is
to adjust the inputs so as to make the simulator predict as closely as
possible the actual observation points. This is widely done by model
users, and is called calibration. The best fitting values of the
uncertain parameters are then used to make predictions of the system.

*Example*: Observing the amount of pollutant reaching some specific
points, we can then calibrate the model and use the best fitting input
values to predict the amounts reaching other points and hence assess the
consequences for key susceptible targets.

In :ref:`MUCM<DefMUCM>`, we take the broader view that such
observations allow us to learn about the uncertain inputs, but not to
eliminate uncertainty. We therefore consider calibration to be the
process of using observations of the real system to modify (and usually
to reduce) the uncertainty about specific inputs.

See also :ref:`data assimilation<DefDataAssimilation>`.
