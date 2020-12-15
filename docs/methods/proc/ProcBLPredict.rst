.. _ProcBLPredict:

Procedure: Predict simulator outputs using a BL emulator
========================================================

Description and Background
--------------------------

A :ref:`Bayes linear<DefBayesLinear>` (BL)
:ref:`emulator<DefEmulator>` is a stochastic representation of
knowledge about the outputs of a :ref:`simulator<DefSimulator>` based
on a :ref:`second-order belief specification<DefSecondOrderSpec>` for
an unknown function. The unknown function in this case is the simulator,
viewed as a function that takes inputs and produces one or more outputs.
One use for the emulator is to predict what the simulator would produce
as output when run at one or several different points in the input
space. This procedure describes how to derive such predictions in the
case of a BL emulator such as is produced by
:ref:`ProcBuildCoreBL<ProcBuildCoreBL>`.

Inputs
------

-  An adjusted Bayes linear emulator
-  A single point :math:`x^\prime` or a set of points :math:`x^\prime_1,
   x^\prime_2,\ldots,x^\prime_{n^\prime}` at which predictions are
   required for the simulator output(s)

Outputs
-------

-  In the case of a single point, outputs are the adjusted expectation
   and variance at that point
-  In the case of a set of points, outputs are the adjusted expectation
   vector and adjusted variance matrix for that set

Procedure
---------

The adjusted Bayes linear emulator will supply the following necessary
pieces of information:

-  An adjusted expectation :math:`\text{E}_F[\beta]` and variance
   :math:`\text{Var}_F[\beta]` for the trend coefficients :math:`\beta` given
   the model runs :math:`F`
-  An adjusted expectation :math:`\text{E}_F[w(x)]` and variance
   :math:`\text{Var}_F[w(x)]` for the residual process :math:`w(x)`, at any
   point :math:`x`, given the model runs :math:`F`
-  An adjusted covariance :math:`\text{Cov}_F[\beta,w(x)]` between the
   trend coefficients and the residual process

The adjusted expectation and variance at the new point :math:`x'` are
obtained by application of :ref:`ProcBLAdjust<ProcBLAdjust>` to the
emulator as described below.

Predictive mean (vector)
~~~~~~~~~~~~~~~~~~~~~~~~

Then our adjusted beliefs about the expected simulator output at a
single further input configuration :math:`x'` are given by:

.. math::
   \text{E}_F[f(x')] = h(x')^T \text{E}_F[\beta] + \text{E}_F[w(x').

In the case of a set of additional inputs :math:`X'`, where :math:`X'` is the
matrix with rows :math:`x^\prime_1, x^\prime_2,\ldots,x^\prime_{n^\prime}`,
the adjusted expectation is:

.. math::
   \text{E}_F[f(X')] = H(X')^T \text{E}_F[\beta] + \text{E}_F[w(X')

where :math:`f(X)` is the :math:`n^\prime`-vector of simulator values with
elements :math:`(f(x^\prime_1), f(x^\prime_2),\ldots,
f(x^\prime_{n^\prime}))`, :math:`H(X')` is the :math:`n^\prime\times q`
matrix with rows :math:`h(x^\prime_1), h(x^\prime_2),\ldots,
h(x^\prime_{n^\prime})`, and :math:`w(X)` is the :math:`n^\prime`-vector with
elements :math:`(w(x^\prime_1), w(x^\prime_2),\ldots,
w(x^\prime_{n^\prime}))`.

Predictive variance (matrix)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our adjusted variance of the simulator output at a single further input
configuration :math:`x'` is given by:

.. math::
   \text{Var}_F[f(x')] = h(x')^T \text{Var}_F[\beta] h(x')
   +\text{Var}_F[w(x')]+2h(x')^T\text{Cov}_F[\beta,w(x')]

In the case of a set of additional inputs :math:`X'`, the adjusted variance
is:

.. math::
   \text{Var}_F[f(X')] =& H(X')^T \text{Var}_F[\beta] H(X')
   +\text{Var}_F[w(X')]+ \\
   & H(X')^T\text{Cov}_F[\beta,w(X')] + \text{Cov}_F[w(X'),\beta] H(X').
