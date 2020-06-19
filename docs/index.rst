.. Multi-Output GP Emulator documentation master file, created by
   sphinx-quickstart on Mon Mar 11 13:49:57 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Multi-Output GP Emulator's documentation!
====================================================

``mogp_emulator`` is a Python package for fitting Gaussian Process Emulators to computer simulation results.
The code contains routines for fitting GP emulators to simulation results with a single or multiple target
values, optimizing hyperparameter values, and making predictions on unseen data. The library also implements
experimental design, dimension reduction, and calibration tools to enable modellers to understand complex
computer simulations.

The following pages give a brief overview of the package, instructions for installation, and an end-to-end
tutorial describing a Uncertainty Quantification workflow using ``mogp_emulator``. Further pages outline
some additional examples, more background details on the methods in the MUCM Toolkit, full implementation
details, and some included benchmarks.

.. toctree::
   :maxdepth: 1
   :caption: Introduction and Installation:

   intro/overview
   intro/installation
   intro/tutorial
   intro/methoddetails

.. toctree::
   :maxdepth: 1
   :caption: Some more specific demos and tutorial illustrating how the various package components can
             be used are:

   demos/gp_demos
   demos/mice_demos
   demos/gp_demoR

.. toctree::
   :maxdepth: 1
   :caption: For a more detailed description of some of the Uncertainty Quantification methods used in
             this package, see the MUCM toolkit pages:

   methods/methods

.. toctree::
   :maxdepth: 1
   :caption: Detailed information on all implemented classes and functions are described in the following pages:

   implementation/implementation

.. toctree::
   :maxdepth: 1
   :caption: For some more speicifs on benchmarks involving the implementation, see the following benchmark examples:

   benchmarks/benchmarks



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
