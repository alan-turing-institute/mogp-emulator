.. _DefNugget:

Definition of Term: Nugget
==========================

A covariance function :math::ref:`v(x,x^\prime)` expresses the covariance
between the outputs of a `simulator<DefSimulator>` at input
configurations :math:`x` and :math:`x^\prime`. When :math:`x=x^\prime`,
:math:`v(x,x)` is the variance of the output at input :math:`x`. A nugget is
an additional component of variance when :math:`x=x^\prime`. Technically
this results in the covariance function being a discontinuous function
of its arguments, because :math:`v(x,x)` does not equal the limit as
:math:`x^\prime` tends to :math:`x` of :math:`v(x,x^\prime)`.

A nugget may be introduced in a variance function in
:ref:`MUCM<DefMUCM>` methods for various reasons. For instance, it
may represent random noise in a :ref:`stochastic<DefStochastic>`
simulator, or the effects of :ref:`inactive inputs<DefInactiveInput>`
that are not included explicitly in the :ref:`emulator<DefEmulator>`.
A small nugget term may also be added for computational reasons when
working with a :ref:`deterministic<DefDeterministic>` simulator.
