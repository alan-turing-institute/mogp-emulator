.. _gpdemoGPU:

Gaussian Process Demo (GPU)
=========================
This demo illustrates a simple example of fitting a GP emulator to results of the projectile
problem discussed in the :ref:`tutorial`, using the GPU implementation of the emulator.

Note that in order for this to work, it must be run on a machine with an Nvidia GPU, and with
CUDA libraries available.  It also depends on Eigen and pybind.

The example uses Maximum Likelihood Estimation with a Squared Exponential kernel, which is currently
the only kernel supported by the GPU implementation.


.. literalinclude::
   ../../mogp_emulator/demos/gp_demo_gpu.py
