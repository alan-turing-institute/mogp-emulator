.. Multi-Output GP Emulator documentation master file, created by
   sphinx-quickstart on Mon Mar 11 13:49:57 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Multi-Output GP Emulator's documentation!
====================================================

`mogp_emulator` is a Python package for fitting Gaussian Process Emulators to computer simulation results.
The code contains routines for fitting GP emulators to simulation results with a single or multiple target
values, optimizing hyperparameter values, and making predictions on unseen data. The library also implements
experimental design, dimension reduction, and calibration tools to enable modellers to understand complex
computer simulations.

The following pages give a brief overview of the package, instructions for installation, and an end-to-end
tutorial describing a Uncertainty Quantification workflow using ``mogp_emulator``.

.. toctree::
   :maxdepth: 1
   :caption: Introduction and Installation:

   intro/overview
   intro/installation
   intro/tutorial
   intro/methoddetails

Some more specific demos and tutorial illustrating how the various package components can be used are:

.. toctree::
   :maxdepth: 1

   demos/gp_demos
   demos/mice_demos
   demos/gp_demoR

For a more detailed description of some of the Uncertainty Quantification methods used in this
package, see the MUCM toolkit pages:

.. toctree::
   :maxdepth: 1

   methods/methods

Detailed information on all implemented classes and functions are described in the following pages:

.. toctree::
   :maxdepth: 1

   implementation/GaussianProcess
   implementation/MultiOutputGP
   implementation/fitting
   implementation/MeanFunction
   implementation/formula
   implementation/Kernel
   implementation/Priors
   implementation/DimensionReduction
   implementation/ExperimentalDesign
   implementation/SequentialDesign
   implementation/HistoryMatching
   implementation/MCMC

For some more speicifs on benchmarks involving the implementation, see the following benchmark examples:

.. toctree::
   :maxdepth: 2

   benchmarks/benchmarks



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
