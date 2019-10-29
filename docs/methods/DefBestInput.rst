.. _DefBestInput:

Definition of Term: Best Input
==============================

No matter how careful a particular model of a real system has been
formulated, there will always be a discrepancy between reality,
represented by the system value :math::ref:`y`, and the
`simulator<DefSimulator>` output :math:`\strut{f(x)}` for any valid
input :math::ref:`\strut{x}`. Perhaps, the simplest way of incorporating this
`model discrepancy<DefModelDiscrepancy>` into our analysis is to
postulate a best input :math:`\strut{x^+}`, representing the best fitting
values of the input parameters in some sense, such that the difference
:math:`d=y-f(x^+)` is independent or uncorrelated with :math:`\strut{f}`,
:math:`\strut{f(x^+)}` and :math:`\strut{x^+}`.

While there are subtleties in the precise interpretation of
:math:`\strut{x^+}`, especially when certain inputs do not correspond to
clearly defined features of the system, the notion of a best input is
vital for procedures such as :ref:`calibration<DefCalibration>`,
history matching and prediction for the real system.
