.. _AltNumericalSolutionForKarhunenLoeveExpansion:

Alternatives: Numerical solution for Karhunen Loeve expansion
=============================================================

Overview
--------

The analytical solutions of the Fredholm equation only exist for some
particular processes, that is, for particular covariance functions. For
this reason the numerical solutions are usually essential. There are two
main methods: the integral method and the expansion method.

Choosing the Alternatives
-------------------------

The eigenfunction is not explicitly defined when using the integral
method.Therefore the expansion method is preferred where the
eigenfunction is approximated using some orthonormal basis functions.

The Nature of the Alternatives
------------------------------

Integral method
~~~~~~~~~~~~~~~

This is a direct approximation for the integral, i.e.
:math:`\int_{0}^{1}f(t)dt \approx \sum_{i=0}^{n+1}\omega_if(t_i).`

The bigger the :math:`n`, the better the approximation.

Expansion method
~~~~~~~~~~~~~~~~

This method is known as Galerkin method. It employs a set of basis
functions. The commonly used basis functions are Fourier (trigonometric)
or wavelets. Implementation is discussed in
:ref:`ProcFourierExpansionForKL<ProcFourierExpansionForKL>` and
:ref:`ProcHaarWaveletExpansionForKL<ProcHaarWaveletExpansionForKL>`.
Note that although we have presented the methods separately, there are
many common features which would comprise a generic method which could
be used, in principle, for any approximation based on a particularly
basis.

Additional Comments, References, and Links
------------------------------------------

The solution of Fredholm equations of the second kind has a strong
background in applied mathematics and numerical analysis. A useful
volume is:

K. E. Atkinson. Numerical solutions of integral equations of the second
kind. Cambridge, 1997.
