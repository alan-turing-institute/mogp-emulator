.. _DefCorrelationLength:

Definition of Term: Correlation length
======================================

The degree of :ref:`smoothness<DefSmoothness>` of an
:ref:`emulator<DefEmulator>` is determined by its correlation
function. Smoothness is in practice controlled by
:ref:`hyperparameters<DefHyperparameter>` in the correlation function
that define how slowly the correlation between the simulator outputs at
two input points declines as the distance between the points increases.
These hyperparameters are typically called correlation length
parameters. Their precise role in determining smoothness depends on the
form of correlation function.
