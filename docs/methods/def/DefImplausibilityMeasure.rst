.. _DefImplausibilityMeasure:

Definition of Term: Implausibility Measure
==========================================

An implausibility measure is a function :math:`I(x)` defined over
the whole input space which, if large for a particular :math:`x`,
suggests that there would be a substantial disparity between the
simulator output :math:`f(x)` and the observed data :math:`z`,
were we to evaluate the model at :math:`\strut{x}`.

In the simplest case where :math:`f(x)` represents a single
output and :math:`z` a single observation, the univariate
implausibility would look like:

.. math::
   I^2(x) = \frac{ ({\rm E}[f(x)] - z )^2}{ {\rm Var}[{\rm E}[f(x)]-z] }
          = \frac{ ({\rm E}[f(x)] - z )^2}{{\rm Var}[f(x)] + {\rm Var}[d] + {\rm Var}[e]}

where :math:`{\rm E}[f(x)]` and :math:`{\rm Var}[f(x)]`
are the emulator expectation and variance respectively; :math:`d`
is the :ref:`model discrepancy<DefModelDiscrepancy>`, and :math:`e`
is the observational error. The second equality follows from the
definition of the :ref:`best input approach<DefBestInput>`.

Several different implausibility measures can be defined in the case
where the simulator produces multiple outputs.
