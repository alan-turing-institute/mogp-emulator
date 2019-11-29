.. _ProcPivotedCholesky:

Procedure: Pivoted Cholesky decomposition
=========================================

Description and Background
--------------------------

The Cholesky decomposition is named after Andre-Louis Cholesky, who
found that a symmetric positive-definite matrix can be decomposed into a
lower triangular matrix and the transpose of the lower triangular
matrix. Formally, the Cholesky method decomposes a symmetric positive
definite matrix :math:`A` uniquely into a product of a lower
triangular matrix :math:`L` and its transpose, i.e. :math:`A =
LL^T`, or equivalently :math:`A = R^TR` where :math:`R` is a
upper triangular matrix.

The Pivoted Cholesky decomposition, or the Cholesky decomposition with
complete pivoting, of a matrix :math:`A` returns a permutation
matrix :math:`P` and the unique upper triangular matrix :math:`R`
such that :math:`P^TAP = R^TR`. The permutation matrix is an
orthogonal matrix, so the matrix :math:`A` can be rewritten as
:math:`A = (RP^T)^T RP^T`

An arbitrary permutation of rows and columns of matrix :math:`A` can
be decomposed by the Cholesky algorithm, :math:`P^TAP=R^TR`, where
:math:`P` is a permutation matrix and :math:`R` is an upper
triangular matrix. The permutation matrix is an orthogonal matrix, so
the matrix :math:`A` can be rewritten as :math:`A = P R^T RP^T`,
so that :math:`C =P R^T` is the pivoted Cholesky decomposition of
:math:`A`.

**Pivoted Cholesky algorithm**: This algorithm computes the Pivoted
Cholesky decomposition :math:`P^TAP=R^TR` of a symmetric positive
semidefinite matrix :math:`A \in \Re^{n \times n}`. The nonzero
elements of the permutation matrix :math:`P` are given by
:math:`P(piv(k),k)=1`, :math:`k=1,\ldots,n`

More details about the numerical analysis of the pivoted Cholesky
decomposition can be found in Higham (2002).

Inputs
------

-  Symmetric positive-definite matrix :math:`A`

Outputs
-------

-  Permutation matrix :math:`P`
-  Unique upper triangular matrix :math:`R`

Procedure
---------

**Initialise**

:math:`R = \mathrm{zeros}(\mathrm{size}(A))`

:math:`piv(k) = k,\, \forall k\in[1,n]`

**Repeat** for :math:`k=1:n`

   | {Finding the pivot}
   | :math:`\quad B = A(k:n,k:n)`
   | :math:`\quad l = \left\{ i : A(i,i) == \max diag \left( B \right) \right\}`
   | {Swap rows and columns}
   | :math:`\quad A(:,k) <=> A(:,l)`
   | :math:`\quad R(:,k) <=> R(:,l)`
   | :math:`\quad A(k,:) <=> A(l,:)`
   | :math:`\quad piv(k) <=> piv(l)`
   | {Cholesky decomposition}
   | :math:`\quad R(k,k) = \sqrt{A(k,k)}`
   | :math:`\quad R(k,k+1:n) = R(k,k)^{-1} A(k,k+1:n)`
   | {Updating :math:`A`}
   | :math:`\quad A(k+1:n,k+1:n) = A(k+1:n,k+1:n) - R(k,k+1:n)^TR(k,k+1:n)`

**End repeat**

Additional Comments
-------------------

If :math:`A` is a variance matrix, the pivoting order given by the
permutation matrix :math:`P` has the following interpretation: the
first pivot is the index of the element with the largest variance, the
second pivot is the index of the element with the largest variance
conditioned on the first element, the third pivot is the index of the
element with the largest variance conditioned on the first two elements
element, and so on.

References
----------

-  Higham, N. J. Accuracy and Stability of Numerical Algorithms. Society
   for Industrial and Applied Mathematics, Philadelphia, PA, USA, 2002.
   ISBN 0-89871-521-0. Second Edition.
