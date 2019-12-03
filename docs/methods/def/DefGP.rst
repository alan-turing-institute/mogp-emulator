.. _DefGP:

Definition of Term: Gaussian process
====================================

A Gaussian process (GP) is a probability model for an unknown function.

If a function :math:`f` has argument :math:`x`, then the value of the function
at that argument is :math:`f(x)`. A probability model for such a function
must provide a probability distribution for :math:`f(x)`, for any possible
argument value :math:`x`. Furthermore, if we consider a set of possible
values for the argument, which we can denote by :math:`x_1, x_2, ..., x_N`,
then the probability model must provide a joint probability distribution
for the corresponding function values :math:`f(x_1), f(x_2),\cdots,
f(x_N)`.

In :ref:`MUCM<DefMUCM>` methods, an :ref:`emulator<DefEmulator>`
is (at least in the fully Bayesian approach) a probability model for the
corresponding :ref:`simulator<DefSimulator>`. The simulator is
regarded as a function, with the simulator inputs comprising the
function's argument and the simulator output(s) comprising the function
value.

A GP is a particular probability model in which the distribution for a
single function value is a normal distribution (also often called a
Gaussian distribution), and the joint distribution of a set of function
values is multivariate normal. In the same way that normal distributions
play a central role in statistical practice by virtue of their
mathematical simplicity, the GP plays a central role in (fully Bayesian)
MUCM methods.

Just as a normal distribution is identified by its mean and its
variance, a GP is identified by its mean function and its covariance
function. The mean function is :math:`\text{E}[f(x)]`, regarded as a
function of :math:`x`, and the covariance function is :math:`\text{Cov}[f(x_1),
f(x_2)]`, regarded as a function of both :math:`x_1` and :math:`x_2`. Note in
particular that the variance of :math:`f(x)` is the value of the covariance
function when both :math:`x_1` and :math:`x_2` are equal to :math:`x` .

A GP emulator represents beliefs about the corresponding simulator as a
GP, although in practice this is always conditional in the sense that
the mean and covariance functions are specified in terms of uncertain
parameters. The representation of beliefs about simulator output values
as a GP implies that the probability distributions for those outputs are
normal, and this is seen as an assumption or an approximation. In the
Bayes linear approach in MUCM, the assumption of normality is not made,
and the mean and covariance functions alone comprise the Bayes linear
emulator.
