.. _ProcHaarWaveletExpansionForKL:

Procedure: Haar wavelet expansion
=================================

This procedure is used to find the eigenvalues and the eigenfunctions of
the covariacne kernel when the analytical solution is not available. The
othonoromal basis function :math:`\strut{\{\psi_i\}}` is used to solve the
problem. This procedure uses Haar wavelet basis functions as the set of
orthonormal basis functions.

The eigenfunction :math:`\phi_{i}(t)` is expressed as a linear combination
of Haar orthonormal basis functions

:math:`\phi_{i}(t)=\sum_{k=1}^M d_{ik} \\psi_k(t)=\theta(t)^T
D_i=D_i^T\psi(t)`.

The Haar wavelet, the simplest wavelet basis function, is defined as

:math:`\psi(x)=\left\{ \\begin{array}{cc} 1 & 0<x<\frac{1}{2} \\\\ -1 &
\\frac{1}{2} \\leq x <1 \\\\ 0 & \\mbox{otherwise} \\end{array}\right
.`.

Inputs
------

#. The covariance function , see
   :ref:`AltCorrelationFunction<AltCorrelationFunction>`, and the
   estimates of its parameters.
#. :math:`\strut{p}` the number of eigenvalues and eigenfunctions required
   to truncate the Karhunen Loeve Expansion at.
#. :math:`\strut{M=2^n}` orthogonal basis functions on :math:`\strut{[0,1]}`
   constructed in the following way

:math:`\strut{\psi_1=1}`; :math:`\psi_i=\psi_{j,k}(x)`; :math:`i=2^j+k+1`;
:math:`j=0,1, \\cdots, n-1`; :math:`k=0,1, \\cdots, 2^j-1` where
:math:`\psi(x)=\left\{ \\begin{array}{cc} 1 & k2^{-j}<x<2^{-j-1}+k2^{-j}
\\\\ -1 & 2^{-j-1}+k2^{-j} \\leq x <2^{-j}+k2^{-j} \\\\ 0 &
\\mbox{otherwise} \\end{array}\right .`

Output
------

#. The eigenvalues :math:`\lambda_i` , :math:` i=1\cdots p`.
#. The matrix of unknown coefficients :math:`\strut{D}`.
#. An approximated covariance function; :math:`R(s,t)=\Psi(s)^T D^T \\Lambda
   D \\Psi(t)`.

Procedure
---------

#. Write the eigenfunction as :math:`\phi_i(t)=\sum_{k=1}^M d_{ik}
   \\psi_{k}(t)=\Psi^T(t) D_i` so that, :math:`R(s,t)=\sum_{m=1}^M
   \\sum_{n=1}^M a_{mn}\psi_m(s) \\psi_n(t)=\Psi(s)^T A \\Psi(t)`.
#. Choose :math:`\strut{M}` time points such that :math:`t_j=\frac{2i-1}{2M}, 1
   \\leq i \\leq M`.
#. Compute the covariance function :math:`\strut{C}` for those
   :math:`\strut{M}` points.
#. Apply the 2D wavelet transform (discrete wavelet transform) on
   :math:`\strut{C}` to obtain the matrix :math:`\strut{A}`.
#. Substitute :math:`R(s,t)=\Psi(s)^T A \\Psi(t)` in
   :math:`\lambda_i\phi_i(t)=\int_{0}^{1} R(s,t)\phi_i(s)`, then we have
   :math:`\lambda_i\Psi^T(t)D_i=\Psi^T(t)AHD_i`.
#. Define :math:`\strut{H}` as a diagonal matrix with diagonal elements
   :math:`h_{11}=1`, :math:`h_{ii}=2^{-j}` :math:`i=2^{j}+k+1, \\quad j=0,1,
   \\cdots, n-1` and :math:`k=0,1, \\cdots, 2^j-1`.
#. Define the whole problem as :math:`\Lambda_{p \\times p} D_{p \\times M}
   \\Psi(t)_{M \\times 1}= D_{p \\times M} H_{M \\times M } A_{M \\times
   M} \\Psi(t)_{M \\times 1}`.
#. From (7), we have :math:`\Lambda D=D H A`.
#. Multiply both sides of (8) by :math:`H^{\frac{1}{2}}`
#. Express the eigen-problem as :math:`\Lambda D H^{\frac{1}{2}}=D
   H^{\frac{1}{2}} H^{\frac{1}{2}} A H^{\frac{1}{2}}` or :math:`\Lambda
   \\hat{D}=\hat{D}. \\hat{A}` where :math:`\hat{D}=DH^{\frac{1}{2}}` and
   :math:`\hat{A}=H^{\frac{1}{2}} A H^{\frac{1}{2}}`.
#. Solve the eigen-problem in (10), then :math:`\Phi(t)=D\Psi(t)=\hat{D}
   H^{-\frac{1}{2}}\Psi(t)`.

Additional Comments, References, and Links
------------------------------------------

The application of the Haar wavelet procedure shows a better
approximation and faster implementation than the Fourier procedure given
in :ref:`ProcFourierExpansionForKL<ProcFourierExpansionForKL>`. For
one dimension implementation shows that :math:`\strut{2^5 = 32} \` provides
very accurate and quite fast approximations.
