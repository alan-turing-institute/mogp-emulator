.. _ProcFourierExpansionForKL:

Procedure: Fourier Expansion
============================

Description and Background
--------------------------

This procedure is used to find the eigenvalues and the eigenfunctions of
the covariance kernel when the analytical solution is not available. The
orthonormal basis :math:`\strut{\{\theta_i\}}` is used to solve the
problem. This procedure uses Fourier basis functions as the set of
orthonormal basis functions.

In general, the eigenfunction :math:`\strut{\phi_{i}(t)}` is written as

:math:`\phi_{i}(t)=\sum_{k=1}^M d_{ik} \\theta_k(t)=\theta(t)^T
D_i=D_i^T\theta(t)` ,

where :math:`\{\theta_i(t)\}` is the set of orthonormal basis functions and
:math:`\strut{\{d_{ik}\}}` is the set of unknown coefficients for the
expansion.

Inputs
------

#. The covariance function, see
   :ref:`AltCorrelationFunction<AltCorrelationFunction>`, and
   estimates of its parameters.
#. :math:`\strut{p}`, the number of eigenvalues and eigenfunctions required
   to truncate the Karhunen Loeve Expansion at.
#. A set of :math:`\strut{M}` adequate basis functions, where
   :math:`\strut{M}` is chosen to be odd. The basis functions are written
   as

:math:`\theta_1(t)=1, \\theta_2(t)=\cos(2 \\pi t), \\theta_3=\sin(2 \\pi
t),` :math:` \\theta_{2i}(t)=\cos(2\pi it), \\theta_{2i+1}(t)=\sin(2\pi
it), i=1, 2, \\cdots \\frac{M-1}{2}.`

Output
------

#. The set of eigenvalues :math:`\strut{\{\lambda_i\}}`, :math:`\strut{
   i=1\cdots p}`.
#. The matrix of unknown coefficients, :math:`\strut{D}`.
#. An approximated covariance kernel; :math:`R(s,t)=\sum_{i=1}^p \\lambda_i
   \\phi_i(s) \\phi_i(t)=\phi(s)^T\Lambda \\phi(t)=\theta(s)^T D^T
   \\Lambda D \\theta(t)`.

Procedure
----------

#. Replace :math:`\phi_i(s)` in the Fredholm equation
   :math:`\int_0^1R(s,t)\phi_i(s)ds=\lambda_i\phi_i(t)` with
   :math:`D_i^T\theta(t)`, then we have :math:`D_i^T\int_0^1 R(s,t)
   \\theta(s)ds=D_i^T\lambda_i\theta(t)`.
#. Multiply both sides of :math:`D_i^T\int_0^1 R(s,t)
   \\theta(s)ds=D_i^T\lambda_i\theta(t)` by :math:`\theta(t)`.
#. Integrate both sides of :math:` D_i^T\int_0^1 R(s,t)
   \\theta(s)\theta(t)^Tds=D_i^T\lambda_i\theta(t)\theta(t)^T` with
   respect to :math:`\strut{t}`.
#. Define :math:`A=\int_0^1\int_0^1R(s,t)\theta(s)\theta(t)dsdt, \\quad
   B=\int_0^1\theta(t)\theta^T(t)dt` where :math:`\strut{A}` is a
   symmetric positive definite matrix and :math:`\strut{B}` is a diagonal
   positive matrix.
#. Matrix implemenation. Write the integration in (4) as :math:`\strut{D_{p
   \\times M}A_{M \\times M}=\Lambda_{p \\times p} DB_{M \\times M}}`
   which is equivalent to :math:`\strut{AD^T=BD^T\Lambda}`.
#. Express :math:`\strut{B}` as :math:`B^{\frac{1}{2}}B^{\frac{1}{2}}`.
#. Express the form in (5) as
   :math:`AD^T=B^{\frac{1}{2}}B^{\frac{1}{2}}D^T\Lambda`.
#. Multilpy both sides of (7) by :math:`\strut{B^{-\frac{1}{2}}}`, so that
   :math:`\strut{B^{-\frac{1}{2}}AB^{-\frac{1}{2}}B^{\frac{1}{2}}D^T=B^{-\frac{1}{2}}D^T\Lambda}`.
#. Assume :math:`\strut{E=B^{-\frac{1}{2}}D^T}` then
   :math:`\strut{B^{-\frac{1}{2}}AB^{-\frac{1}{2}}E=E\Lambda}`.
#. Solve the eigen-problem of
   :math:`\strut{B^{-\frac{1}{2}}AB^{-\frac{1}{2}}E=E\Lambda}`.
#. Compute :math:`\strut{D}` using :math:`\strut{D=E^T B^{-\frac{1}{2}}}`.

Additional Comments, References, and Links
------------------------------------------

For the spatial case of dimension :math:`\strut{d}`, the procedure is
repeated :math:`\strut{d}` times, and the number of eigenvalues, similarly
the number of eigenfunctions, is equal to :math:`\strut{p^d}`.
