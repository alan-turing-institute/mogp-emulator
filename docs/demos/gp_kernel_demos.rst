.. _gpkerneldemos:

Gaussian Process Kernel Demos (Python)
======================================

This demo illustrates use of some of the different kernels available
in the package and how they can be set. In particular, it shows use
of the ``ProductMat52`` kernel and the ``UniformSqExp`` kernel and
how these kernels give slightly different optimal hyperparameters
on the same input data.

.. literalinclude::
   ../../mogp_emulator/demos/gp_kernel_demos.py