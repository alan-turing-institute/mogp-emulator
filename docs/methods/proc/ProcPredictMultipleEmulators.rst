.. _ProcPredictMultipleEmulators:

Procedure: Predict functions of simulator outputs using multiple independent emulators
======================================================================================

Description and Background
--------------------------

Where separate, independent :ref:`emulators<DefEmulator>` have been
built for different :ref:`simulator<DefSimulator>` outputs, there is
often interest in predicting some function(s) of those outputs. The
procedures here are for making such predictions. We assume that,
whatever method was used to build each emulator the corresponding
toolkit thread also describes how to make predictions for that emulator
alone.

The individual emulators may be :ref:`Gaussian process<DefGP>` (GP)
or :ref:`Bayes linear<DefBayesLinear>` (BL) emulators, although some
of the specific procedures given here will only be applicable to GP
emulators.

Inputs
------

-  Emulators for :math:`r` simulator outputs :math:`f_u(x)`,
   :math:`u=1,2,\ldots,r`
-  :math:`r'` prediction functions :math:`f_w^*(x)=g_w\{f_1
   (x),\ldots,f_r(x)\}`, :math:`w=1,2,\ldots,r'`
-  A single point :math:`x^\prime` or a set of points :math:`x^\prime_1,
   x^\prime_2,\ldots,x^\prime_{n^\prime}` at which predictions are
   required for the prediction function(s)

Outputs
-------

-  Predictions in the form of statistical quantities, such as expected
   values, variances and covariances or a sample of values from the
   (joint) predictive distribution of the prediction function(s) at the
   required prediction points

Procedures
----------

The simplest case is when the prediction functions are linear in the
outputs. Then

:math:`f_w^*(x)=a_w + f(x)^T b_w`,

where :math:`a_w` is a known constant, :math:`f(x)` is the vector of :math:`r`
outputs :math:`f_1(x),\ldots,f_r(x)` and :math:`b_w` is a known :math:`r\times 1`
vector. It is straightforward to derive means, variances and covariances
for the functions :math:`f_w^*(x)` at the prediction point(s) when the
prediction functions are linear. For nonlinear functions, the procedure
is to draw a large sample from the predictive distribution and to
compute means, variances and covariances from this sample (but note that
this is only possible for GP emulators).

Predictive means
~~~~~~~~~~~~~~~~

Suppose that we have linear prediction functions. Let the :math:`n'\times
1` predictive mean vector for the :math:`u`-th emulator at the :math:`n'`
prediction points be :math:`m_u`. (If we only wish to predict a single
point, then this is a scalar.) Let the :math:`n'\times r` matrix :math:`M`
have columns :math:`m_1,\ldots,m_r`. Then the predictive mean (vector) of
:math:`f_w^*(x)` at the :math:`n'` prediction points is :math:`a_w 1_{n'} +
M\,b_w`, where :math:`1_{n'}` denotes a vector of :math:`n'` ones, so that
:math:`a_w 1_{n'}` is a :math:`n'\times 1` vector with all elements equal to
:math:`a_w`.

Predictive variances and covariances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose that we have linear prediction functions. Let the :math:`n'\times
n'` predictive variance matrix for the :math:`\strut u`-th emulator at the
:math:`n'` prediction points be :math:`V_u`. (If we only wish to predict a
single point, then this is a scalar.) Let :math:`b_{uw}` be the :math:`\strut
u`-th element of :math:`b_w`. Then the variance matrix for :math:`f_w^*(x)`
at the :math:`n'` prediction points is

:math:`\sum_{u=1}^r b_{uw}^2 V_u`,

and the covariance matrix between :math:`f_w^*(x)` and :math:`f_{w'}^*(x)` at
those points is

:math:`\sum_{u=1}^r b_{uw} b_{uw'} V_u`.

Sample of predictions
~~~~~~~~~~~~~~~~~~~~~

Suppose we wish to draw a sample of :math:`N` values from the (joint)
predictive distribution of the prediction functions at the input
:math:`x^\prime`, or at the points :math:`x^\prime_1,
x^\prime_2,\ldots,x^\prime_{n^\prime}`. For GP emulators, such samples
can be drawn from the predictive distributions of the individual
outputs. Let :math:`f_u^{(I)}(x'_t)` be the :math:`\strut I`-th sampled value
of :math:`f_u(x)` at :math:`\strut t`-th prediction point.

Then the :math:`\strut I`-th sampled value, :math:`I=1,2,\ldots,N`, of
:math:`f^*_w(x'_t)` is :math:`g_w\{f_1^{(I)}(x'_t),\ldots,f_r^{(I)}(x'_t)\}`.
