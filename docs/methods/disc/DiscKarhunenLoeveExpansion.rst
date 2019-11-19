.. _DiscKarhunenLoeveExpansion:

Discussion: the Karhunen-Loeve expansion for Gaussian processes
===============================================================

Description and Background
--------------------------

Let :math:`\{Z(x)\}` be a second order stochastic process with zero mean
and continuous covariance function defined on an interval :math:`[a,b]` :

.. math::
   R(s,t)=Cov(Z(s),Z(t))=E(Z(s)Z(t)).

The Karhunen-Loeve (K-L) expansion is a commonly used result of Mercer's
theorem and Reproducing Kernel Hilbert Space (RKHS).

Using K-L expansion of the covariance function :math:`R(s,t)` is
:math:`R(s,t)=\sum_{i=0}^{\infty} \lambda_i\phi_i(s)\phi_i(t)
\label{kernel}` where the :math:`\{\lambda_i,i\in N\}` and
:math:`\{\phi_i,i\in N\}` are respectively the nonzero eigenvalues and the
eigenfunctions of the following integral equation known as "Fredholm
integral equation of the second kind".

.. math::
   \int_a^b R(s,t)\phi_i(t)dt=\lambda_i \phi_i(s).

Theory shows that the K-L expansion of :math:`R(s,t)` converges uniformly
in both variables :math:`s` and :math:`t`.

The stochastic process :math:`Z(x)` itself can then be expressed as

.. math::
   Z(x) = \sum_{i=0}^{\infty} \lambda_i \zeta_i \phi_i(x)

Where we assume that the :math:`\phi_i` orthonormal functions with
respect to uniform measure on :math:`[a,b]` and the :math:`\zeta_i`
are independent, zero mean, unit variance Gaussian random variables.

Discussion: Karhunen-Loeve for the spatial (emulator) case
----------------------------------------------------------

For the spatial case, a given covariance function between
:math:`s,t` represented as :math:`R(s,t)=Cov [Z(s_1,s_2, \cdots,
s_d),Z(t_1,t_2, \cdots, t_d)]` can be approximated using the truncated
K-L expansion (see below). As in the one-dimensional case, a set of
eigenfunctions can be used to represent the covariance function and the
process. The basic idea, for use in emulation and optimal design, is
that the behaviour of the process will be captured by the first
:math:`p` eigenvalues and eigenfunctions.

In the case that :math:`R(s,t)` is separable let the eigenvalues and
eigenfunctions of the correlation function in the :math:`k`-th
dimension be :math:`\{\lambda_k^{(i)}\}` and :math:`\{\phi_k^{(i)}(x_k)\}`.
Then

.. math::
   Z(x)=Z(x_1,x_2,\cdots,
   x_d)=\sum_i\sqrt{\lambda_i}\zeta_i\phi_i(x_1,x_2,\cdots, x_d)

where

.. math::
   \lambda_i=\Pi_{k=1}^{d}\lambda_k^{(i_k)},\;\; \phi_i(x_1,x_2,\cdots,
   x_d)=\Pi_{k=1}^{d}\phi_k^{(i_k)}(x_k),\; i=(i_1, \cdots, i_d)

and the sum is over all such :math:`i`. That is to say we take all
products of all one dimensional expansions.

For the proposed application to the spatial sampling represented by the
GP emulator we replace the problem of estimating the covariance
parameters by converting the problem (approximately) to a standard Bayes
optimal design problem by truncating the K-L expansion:

.. math::
   R(s,t)=\sum_{i=0}^{p} \lambda_i \phi_i(s) \phi_i(t)

In its simplest form take the same :math:`\lambda` to be the
(prior) variances for a Bayesian random regression obtained by the
equivalent truncation of the process itself:

.. math::
   Z(x)=\sum_{i=1}^{p}\sqrt{\lambda_i}\zeta_i \phi(x)

Theory shows that the K-L expansion truncated to depth :math:`p`
has the minimum integrated mean squared approximation to the full
process, among all depth :math:`p` orthogonal expansions. This
reduces the problem to finding the first :math:`p` eigenvalues and
eigenfunctions of the Fredholm equation. The truncation point
:math:`p` should depend on the required level of accuracy in the
reconstruction of the covariance function. Because the reconstructed
covariance function will typically be singular if the number of design
points :math:`n` is greater than :math:`p` one should add a
:ref:`nugget<DefNugget>` term in this case. One way to interpret this
is that all the higher frequency terms hidden in the original covariance
function are absorbed into the nugget.

To summarise, the original output process in matrix terms, :math:`Y=X\beta +
Z`, where :math:`Z` is from the Gaussian process, is replaced by
:math:`Y = X \beta + \Phi \gamma + \varepsilon`, where
:math:`\Phi` is the design matrix from the :math:`\phi_1, \cdots,
\phi_p` and :math:`\varepsilon \sim N(0, \sigma^2)`, with
small :math:`\sigma`, is the nugget term. The default covariance
matrix for the :math:`\gamma` parameters is to take the diagonal matrix
with entries :math:`\lambda_1,\cdots, \lambda_p`.

When the approximation is satisfactory and with a suitable nugget
attached the problem is reduced to a Bayesian random regression optimal
design method as in :ref:`AltOptimalCriteria<AltOptimalCriteria>`,
with an appropriate criterion.

The principle at work is that the :math:`\phi_j` for larger
:math:`j` express the higher frequency components of the process
and conversely the smaller :math:`j` the lower frequency
components. It is expected that this will be a useful platform for a
fully adaptive version of :ref:`ProcASCM<ProcASCM>`, where the
variances attached to the :math:`\gamma` parameters should adapt to the
smoothness of the process.

Finding the eigenvalues and the eigenfunctions needs numerical solution
of the eigen-equations (Fredholm integral equation of the second kind).
Analytical solutions only exist for some a limited range of processes.
Fast numerical solutions are described in the alternatives page
:ref:`AltNumericalSolutionForKarhunenLoeveExpansion<AltNumericalSolutionForKarhunenLoeveExpansion>`.

Additional Comments, References, and Links
------------------------------------------

Although the methods to approximate the K-L expansion, particularly the
Haar method, are fast, the number of basis functions in the K-L
expansion in :math:`d` dimensions, when the product form described
above is taken, can be large: :math:`p^d` if we approximate to
order :math:`p` in every dimension. To partly avoid this blow up
one can take, not every product basis function, but limit the "degree"
of the approximations across all dimensions: e.g. :math:`0 \leq
\sum_{j=1}^d p_j \leq p`. This is the analogue of taking a
polynomial basis up to a particular "total degree" as in a quadratic
response surface. An alternative method is to order the products of the
eigenvalues and truncate by the value. Another point to note is that for
very smooth function :math:`p` need not be large because the
:math:`\lambda_i` decline rapidly. Values of :math:`p` from 3
to 5 are effective.

S. P. Huang, S. T. Quek and K. K. Phoon: Convergence study of the
truncated Karhunen-Loeve expansion for simulation of stochastic
processes, *Intl J. for Numerical Methods in Engineering*, 52:1029-1043,
2001.
