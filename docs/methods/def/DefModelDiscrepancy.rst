.. _DefModelDiscrepancy:

Definition of Term: Model Discrepancy
=====================================

No matter how careful a particular model of a real system has been
formulated, there will always be a difference between reality,
represented by the system value :math:`y`, and the
:ref:`simulator<DefSimulator>` output :math:`f(x)` for any valid input
:math:`x`. The difference :math:`d=y-f(x^+)`, where :math:`\strut{x}^+`
is the :ref:`best input<DefBestInput>`, is referred to as the model
discrepancy, which should be incorporated into our analysis in order to
make valid statements about the real system. In particular, model
discrepancy is vital for procedures such as
:ref:`calibration<DefCalibration>`, history matching and prediction
for the real system.
