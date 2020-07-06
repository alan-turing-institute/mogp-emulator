.. _gpdemos:

Gaussian Process Demo (Python)
==============================

This demo illustrates some various examples of fitting a GP emulator to results of the projectile
problem discussed in the :ref:`tutorial`. It shows a few different ways of estimating the
hyperparameters. The first two use Maximum Likelihood Estimation with two different kernels
(leading to similar performance), while the third uses a linear mean function and places prior
distributions on the hyperparameter values. The MAP estimation technique generally leads to
significantly better performance for this problem, illustrating the benefit of setting priors.

.. literalinclude::
   ../../mogp_emulator/demos/gp_demos.py