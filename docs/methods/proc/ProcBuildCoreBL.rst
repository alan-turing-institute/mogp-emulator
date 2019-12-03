.. _ProcBuildCoreBL:

Procedure: Building a Bayes linear emulator for the core problem (variance parameters known)
============================================================================================

Description and Background
--------------------------

The preparation for building a :ref:`Bayes linear<DefBayesLinear>`
(BL) :ref:`emulator<DefEmulator>` for the :ref:`core
problem<DiscCore>` involves defining the prior mean and
covariance functions, specifying prior expectations and variances for
the :ref:`hyperparameters<DefHyperparameter>`, creating a
:ref:`design<DefDesign>` for the :ref:`training
sample<DefTrainingSample>`, then running the
:ref:`simulator<DefSimulator>` at the input configurations specified
in the design. All of this is described in the thread for Bayes linear
emulation for the core model (:ref:`ThreadCoreBL<ThreadCoreBL>`). In
this case, we consider taking those various ingredients with specified
point values for the variance parameters and creating the BL emulator.

Inputs
------

-  Basis functions, :math:`h(\cdot)` for the prior mean function
   :math:`m_\beta(\cdot)`
-  Prior expectation, variance and covariance specifications for the
   regression coefficients :math:`\beta`
-  Prior expectation for the residual process :math:`w(x)`
-  Prior covariance between the coefficients and the residual process
-  Specified correlation form for :math:`c_\delta(x,x')`
-  Specified values for :math:`\sigma^2` and :math:`\delta`
-  Design :math:`X` comprising points :math:`\{x_1,x_2,\ldots,x_n\}` in the
   input space
-  Output vector :math:`F=(f(x_1),f(x_2),\ldots,f(x_n))^T`, where
   :math:`f(x_j)` is the simulator output from input vector :math:`x_j`

Outputs
-------

-  Adjusted expectations, variances and covariances for :math:`\beta`
-  Adjusted residual process
-  Adjusted covariance between :math:`\beta` and the residual process

These outputs, combined with the form of the mean and covariance
functions, define the emulator and allow all necessary computations for
tasks such as prediction of the simulator output, :ref:`uncertainty
analysis<DefUncertaintyAnalysis>` or :ref:`sensitivity
analysis<DefSensitivityAnalysis>`.

Procedure
---------

The procedure of building the Bayes linear emulator is simply the
:ref:`adjustment<DefBLAdjust>` (as described in the procedure page on
calculating the adjusted expectation and variance
(:ref:`ProcBLAdjust<ProcBLAdjust>`)) of the emulator by the observed
simulator outputs.

Adjustment
~~~~~~~~~~

To adjust our beliefs for :math:`\beta` and the residual process :math:`w(x)`
we require the following prior specifications:

-  :math:`\textrm{E}[\beta]`, :math:`\textrm{Var}[\beta]` - prior expectation
   and variance for the regression coefficients :math:`\beta`
-  :math:`\textrm{E}[w(x)]`, :math:`\textrm{Var}[w(x)]` - prior expectation
   and variance for the residual process :math:`w(\cdot)` at any point
   :math:`x` in the input space
-  :math:`\textrm{Cov}[w(x),w(x')]` - prior covariance between the residual
   process :math:`w(\cdot)` at any pair of points :math:`(x,x')`
-  :math:`\textrm{Cov}[\beta,w(x)]` - prior covariance between the
   regression coefficients :math:`\beta` and the residual process
   :math:`w(\cdot)` at any point :math:`x`

Given the relationship :math:`f(x)=h(x)^T\beta+w(x)`, define the following
quantities obtained from the prior specifications:

Adjusted expectation and variance for trend coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define :math:`H(X)` to be the :math:`n\times q` matrix of basis functions over
the design with rows :math:`h(x_1),h(x_2),\dots,h(x_n)`, and :math:`w(X)` to
be the :math:`n`-vector of emulator trend residuals with elements
:math:`w(x_1),w(x_2),\dots,w(x_n)` where :math:`x_i` is the i-th point in the
design :math:`X`. Then the adjusted expectation and variance for :math:`\beta`
are given by:

.. math::
   \textrm{E}_F[\beta] &=& \textrm{E}[\beta] + \textrm{Var}[\beta] H(X)
   \{H(X)^T\textrm{Var}[\beta]H(X) + \textrm{Var}[w(X)] \}^{-1}
   \times (F - H(X)^T\textrm{E}[\beta] - \textrm{E}[w(X)]) \\
   \textrm{Var}_F[\beta] &=& \textrm{Var}[\beta] - (\textrm{Var}[\beta] H(X))
   \{H(X)^T\textrm{Var}[\beta]H(X) + \textrm{Var}[w(X)] \}^{-1}
   (H(X)^T\textrm{Var}[\beta])

Adjusted expectation and variance for residual process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The adjusted expectation and variance for :math:`w(\cdot)` at any further
input point :math:`x`, and the adjusted covariance between any further pair
of points :math:`(x,x')` are given by:

.. math::
   \textrm{E}_F[w(x)] &=& \textrm{E}[w(x)] + \textrm{Cov}[w(x),w(X)]
   \{H(X)^T\textrm{Var}[\beta]H(X) + \textrm{Var}[w(X)] \}^{-1}
   \times(F - H(X)^T\textrm{E}[\beta] - \textrm{E}[w(X)]) \\
   \textrm{Var}_F[w(x)] &=& \textrm{Var}[w(x)] - \textrm{Cov}[w(x),w(X)]
   \{H(X)^T\textrm{Var}[\beta]H(X) + \textrm{Var}[w(X)] \}^{-1}
   \textrm{Cov}[w(X),w(x)] \\
   \textrm{Cov}_F[w(x),w(x')] &=& \textrm{Cov}[w(x),w(x')] - \textrm{Cov}[w(x),w(X)]
   \{H(X)^T\textrm{Var}[\beta]H(X) + \textrm{Var}[w(X)]\}^{-1}
   \textrm{Cov}[w(X),w(x')]

Adjusted covariance between trend coefficients and residual process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The adjusted covariance between the trend coefficients and the residual
process :math:`w(\cdot)` at any further input point :math:`x` is given by:

.. math::
   \textrm{Cov}_F[\beta,w(x)] = \textrm{Cov}[\beta,w(x)]-\textrm{Var}[\beta]
   H(X)\{H(X)^T\textrm{Var}[\beta]H(X) + \textrm{Var}[w(X)] \}^{-1}
   \textrm{Cov}[w(X),w(x)]
