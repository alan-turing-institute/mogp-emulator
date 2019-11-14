.. _DefTProcess:

Definition of Term: T-process
=============================

A t-process is a probability model for an unknown function.

If a function f has argument :math:`x`, then the value of the function at
that argument is :math:`f(x)`. A probability model for such a function must
provide a probability distribution for :math:`f(x)`, for any possible
argument value :math:`x`. Furthermore, if we consider a set of possible
values for the argument, which we can denote by :math:`x_1, x_2, ..., x_n`,
then the probability model must provide a joint probability distribution
for the corresponding function values :math:`f(x_1), f(x_2), ..., f(x_n)`.

In :ref:`MUCM<DefMUCM>` methods, an :ref:`emulator<DefEmulator>`
is (at least in the fully Bayesian approach) a probability model for the
corresponding :ref:`simulator<DefSimulator>`. The simulator is
regarded as a function, with the simulator inputs comprising the
function's argument and the simulator output(s) comprising the function
value.

A t-process is a particular probability model in which the distribution
for a single function value is a t distribution (also often called a
Student-t distribution), and the joint distribution of a set of function
values is multivariate t. The t-process is related to the :ref:`Gaussian
process<DefGP>` (GP) in the same way that a univariate or
multivariate t distribution is related to a univariate or multivariate
normal (or Gaussian) distribution. That is, we can think of a t-process
as a GP with an uncertain (or random) scaling parameter in its
covariance function. Normal and t distributions play a central role in
statistical practice by virtue of their mathematical simplicity, and for
similar reasons the GP and t-process play a central role in (fully
:ref:`Bayesian<DefBayesian>`) MUCM methods.

A t-process is identified by its mean function, its covariance function
and a degrees of freedom. If the degrees of freedom is sufficiently
large, which will in practice always be the case in MUCM applications,
the mean function is :math:`\text{E}[f(x)]`, regarded as a function of
:math:`x`, and the covariance function is :math:`\text{Cov}[f(x_1), f(x_2)]`,
regarded as a function of both :math:`x_1` and :math:`x_2`. Note in particular
that the variance of :math:`f(x)` is the value of the covariance function
when both :math:`x_1` and :math:`x_2` are equal to :math:`x`.
