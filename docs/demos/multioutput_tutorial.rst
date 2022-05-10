.. _multioutput_tutorial:

Multi-Output Tutorial
=====================

*Note: This tutorial requires Scipy version 1.4 or later to run the simulator.*

This page includes an end-to-end example of using ``mogp_emulator`` to perform model calibration
with a simulator with multiple outputs. Note that this builds on the main tutorial with a
second output (in this case, the velocity of the projectile at the end of the simulation),
which is able to further constrain the NROY space as described in the first tutorial.

.. literalinclude:: ../../mogp_emulator/demos/multioutput_tutorial.py

One thing to note about multiple outputs is that they must be run as a script with a 
``if __name__ == __main__`` block in order to correctly use the multiprocessing
library. This can usually be done as in the example for short scripts, while for more
complex analyses it is usually better to define functions (as in the benchmark for
multiple outputs).

More Details
------------

More details about these steps can be found in the :ref:`methods` section, or on the following page
that goes into :ref:`more details <methoddetails>` on the options available in this software library.
For more on the specific implementation detials, see the various
:ref:`implementation pages <implementation>` describing the software components.
