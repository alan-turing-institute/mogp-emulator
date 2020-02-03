.. _DiscBuildCoreGP:

Discussion: Computational issues in building a Gaussian process emulator for the core problem
=============================================================================================

Description and Background
--------------------------

The procedure for building a :ref:`Gaussian process<DefGP>`
:ref:`emulator<DefEmulator>` for the :ref:`core problem<DiscCore>`
is described in page :ref:`ProcBuildCoreGP<ProcBuildCoreGP>`.
However, computational problems can arise in implementing that
procedure, and these problems are discussed here.

Inversion of the correlation matrix of the training sample
----------------------------------------------------------

Problems arise primarily in the computation of the inverse, :math:`A^{-1}
\equiv c(D,D)^{-1}`, of the correlation matrix of the :ref:`training
sample<DefTrainingSample>` data. This can happen with any form
of correlation function when the training sample is large or when the
:ref:`correlation function<AltCorrelationFunction>` exhibits a high
degree of :ref:`smoothness<DefSmoothness>` (e.g. large values of
correlation length parameters) relative to the size of the design
region. It is particularly problematic for highly
:ref:`regular<DefRegularity>` correlation functions (such as the
Gaussian form).

Non invertibility of the correlation matrix can imply that two or more
points are so close in the input space, that offer little or no
information to the emulator. A way of sidestepping this problem is to
identify these points using the :ref:`pivoted
Cholesky<ProcPivotedCholesky>` decomposition. The result of this
decomposition are two matrices, :math:`R`, which is an upper
triangular containing the Cholesky coefficients, and :math:`P`,
which is a permutation matrix. For example if

.. math::
   A = \left[ \begin{array}{ccc} 1&0.1&0.2 \\ 0.1&3&0.3 \\
   0.2&0.3&2\end{array} \right]

then

.. math::
   R = \left[ \begin{array}{ccc} 1.73&0.17&0.06 \\ 0&1.40&0.14 \\
   0&0&0.99\end{array} \right]

and

.. math::
   P = \left[ \begin{array}{ccc} 0&0&1 \\ 1&0&0 \\
   0&1&0\end{array} \right] \, {\rm or} \, piv = [2, 3, 1]

Of interest here is the permutation matrix :math:`P`. The row number
of the '1' in the first column, shows the position of the element with
the larger variance (in this example it is the second element).
Similarly, the row number of the '1' in the second column shows the
position of the element with the largest variance, conditioned on the
first element, and so forth. If the matrix :math:`A` is non
invertible, and therefore, non positive definite, the :math:`k` last
elements in the main diagonal of :math:`R` will be zero, or very
small. If this is the case, the design points that correspond to the row
number where the ones appear in the last :math:`k` columns of matrix
:math:`P` can be excluded from the design matrix, and the emulator
can be built using the remaining points, without losing information in
principle.

Matrix inversion using the Cholesky decomposition
-------------------------------------------------

The parameter estimation and prediction formulae presented in
:ref:`ProcBuildCoreGP<ProcBuildCoreGP>` contain a fair amount of
matrix inversions. Although these can be calculated with general purpose
inversion algorithms, there are more efficient ways of carrying out the
inversion, by taking into account the special structure of these
matrices; that is, by taking into account that the matrices or matrix
expressions that need to be inverted are positive definite.

The majority of the matrix inverses come in two forms: a left inverse
matrix multiplication (i.e. :math:`A^{-1}f(D)`) and a quadratic term
of the form :math:`H^{\rm{T}}A^{-1}H`. Both of these forms can be
efficiently calculated using the Cholesky decomposition and the left
matrix division. We consider the Cholesky decomposition of a matrix to
the product of a lower triangular matrix :math:`L` and its
transpose, i.e.

.. math::
   A = LL^{\rm T}

The left matrix division, which we denote with backslash
:math:`(\backslash)`, represents the solution to a linear system of
equations; that is if :math:`Ax = y`, then :math:`x = A\backslash
y`. Furthermore, the fact that the Cholesky decomposition results in
triangular matrices, means that we can calculate expressions of the form
:math:`L\backslash y` using backsubstitution, taking advantage of
its efficiency.

Using the left matrix division and the Cholesky decomposition,
expressions of the form :math:`A^{-1} f(D)` can be calculated as

.. math::
   A^{-1} f(D) \equiv L^{\rm T}\backslash (L\backslash f(D))

On the other hand, quadratic expressions of the form
:math:`H^{\rm{T}}A^{-1}H` can be calculated using an intermediate vector
:math:`w`, as

.. math::
   w = L\backslash H, \qquad H^{\rm{T}}A^{-1}H \equiv w^{\rm T}w

The Cholesky decomposition is also useful in the calculation of
logarithms of derivatives, yielding more numerically stable results. The
logarithm of the derivative of :math:`A`, which appears in various
likelihood expressions, is best calculated using the identity

.. math::
   \ln|A| \equiv 2\sum \ln(L_{ii})

with :math:`L_{ii}` being the i-th element on the main diagonal of
:math:`L`.

Example
~~~~~~~

As an example, we show how the expression for :math:`\hat{\beta}` can be
calculated, using the Cholesky decomposition. The original expression is

.. math::
   \hat{\beta} = (H^{\rm T}A^{-1}H)^{-1}H^{\rm T}A^{-1}f(D)

This can be calculated with the following five steps

-  :math:`L = chol(A)`
-  :math:`w = L\backslash H`
-  :math:`Q = w^{\rm T} w \quad (=H^{\rm T}A^{-1}H)`
-  :math:`K = chol(Q)`
-  :math:`\hat{\beta} = K^{\rm T}\backslash (K\backslash H^{\rm T})(L^{\rm
   T}\backslash(L\backslash f(D)))`

Even though this implementation might be more cumbersome, it is
numerically more stable compared to an implementation that uses general
purpose matrix inversion routines.

High input dimensionality
-------------------------

Problems may also arise through having a large number of parameters to
estimate. When the number of :ref:`simulator<DefSimulator>` inputs is
large, there are many elements in the correlation function's
:ref:`hyperparameter<DefHyperparameter>` vector :math:`\delta`, and it
can be computationally very demanding then to find a suitable single
estimate or to compute a sample using Markov chain Monte Carlo.
Experience suggests that in practice in a complex simulator with many
inputs many of those inputs will be more or less redundant, having
negligible influence on the output in the practical context of interest.
Thus it is important to be able to apply a preliminary
:ref:`screening<DefScreening>` process to reduce the input dimension
- see the screening topic thread
(:ref:`ThreadTopicScreening<ThreadTopicScreening>`).

Additional Comments
-------------------

Techniques for addressing these problems are being developed within
:ref:`MUCM<DefMUCM>`, and will be incorporated in this page in due
course.
