.. _excalibur_workshop_demo:

Gaussian Process Demo with Small Sample Size
============================================

This demo includes an example shown at the ExCALIBUR workshop held online on
24-25 September, 2020. The example shows the challenges of fitting a GP
emulator to data that is poorly sampled, and how a mean function and
hyperparameter priors can help constrain the model in a situation where
a zero mean and Maximum Likelikhood Estimation perform poorly.

The specific example uses the projectile problem discussed in the
:ref:`tutorial`. It draws 6 samples, which might be a typical sampling
density for a high dimensional simulator that is expensive to run, where
you might be able to draw a few samples per input parameter. It shows the
true function, and then the emulator means predicted at the same points
using Maximum Likelihood Estimation and a linear mean function combined with
Maximum A Posteriori Estimation. The MLE emulator is completely useless,
while the MAP estimation technique leads to significantly better performance
and an emulator that is useful despite only drawing a small number of samples.

.. literalinclude::
   ../../mogp_emulator/demos/excalibur_workshop_demo.py
