.. _DefSmoothingKernel:

Definition of term: Smoothing kernel
====================================

A smoothing kernel is a non-negative real-valued integrable function
:math:`\kappa()` satisfying the following two requirements:

:math:`\int_{-\infty}^{+\infty}\kappa(u)\,du` is finite;

:math:`\kappa(-u) = \\kappa(u) \\mbox{ for all values of } u\,.`

In other words, any scalar multiple of a symmetric probability density
function constitutes a smoothing kernel.

Smoothing kernels are used in constructing multivariate covariance
functions (as discussed in
:ref:`AltMultivariateCovarianceStructures<AltMultivariateCovarianceStructures>`),
in which case they depend on some hyperparameters. An example of a
smoothing kernel in this context is

:math:`\kappa(x)=\exp\{-2 \\sum_{i=1}^p (x_i/\delta_i)^2 \\} \\, ,`

where :math:`p` is the length of the vector :math:`x`. In this case the
hyperparameters are :math:`\delta=(\delta_1,...,\delta_p)`.
