.. _DiscGPBasedEmulator:

Discussion: Forms of GP-Based Emulators
=======================================

Description and Background
--------------------------

In the fully :ref:`Bayesian<DefBayesian>` approach to
:ref:`emulating<DefEmulator>` the output(s) of a
:ref:`simulator<DefSimulator>`, the emulation is based on the use of
a :ref:`Gaussian process<DefGP>` (GP) to represent the simulator
output as a function of its inputs. However, the underlying model
represents the output as a GP *conditional* on some
:ref:`hyperparameters<DefHyperparameter>`. This means that a fully
Bayesian (GP-based) emulator in the toolkit has two parts. The parts
themselves can have a variety of forms, which we discuss here.

Discussion
----------

An emulator in the fully Bayesian approach is a full probability
specification for the output(s) of the simulator as a function of its
inputs, that has been trained on a :ref:`training
sample<DefTrainingSample>`. Formally, the emulator is the
Bayesian posterior distribution of the simulator output function
:math:`f(\cdot)`. Within the toolkit, a GP-based emulator specifies this
posterior distribution in two parts. The first part is the posterior
distribution of :math:`f(\cdot)` conditional on the hyperparameters. The
second is a posterior specification for the hyperparameters.

First part
~~~~~~~~~~

In the simplest form, the first part is a GP. It is defined by
specifying

-  The hyperparameters :math:`\theta` upon which the GP is conditioned
-  The posterior mean function :math:`m^*(\cdot)` and covariance function
   :math:`v^*(\cdot,\cdot)`, which will be functions of (some or all of)
   the hyperparameters in :math:`\theta`

However, in some situations it will be possible to integrate out some of
the full set of hyperparameters, in which case the conditional
distribution may be, instead of a GP, a :ref:`t
process<DefTProcess>`. The definition now specifies

-  The hyperparameters :math:`\theta` upon which the t process is
   conditioned
-  The posterior degrees of freedom :math:`b^*`, mean function
   :math:`m^*(\cdot)` and covariance function :math:`v^*(\cdot,\cdot)`, which
   will be functions of (some or all of) the hyperparameters in
   :math:`\theta`

Second part
~~~~~~~~~~~

The full probabilistic specification is now completed by giving the
posterior distribution :math:`\pi_\theta(\theta)` for the hyperparameters
on which the first part is conditioned. The second part could consist
simply of this posterior distribution. However, :math:`\pi_\theta(\cdot)`
is not generally a simple distribution, and in particular not a member
of any of the standard families of distributions that are widely used
and understood in statistics. For computational reasons, we therefore
augment this abstract statement of the posterior distribution with one
or more specific values of :math:`\theta` designed to provide a discrete
representation of this distribution.

We can denote these sample values of :math:`\theta` by :math:`\theta^{(j)}`,
for :math:`j=1,2,\ldots,s`, where :math:`s` is the number of hyperparameter
sets provided in this second part. At one extreme, :math:`s` may equal just
1, so that we are using a single point value for :math:`\theta`. Clearly,
in this case we have a representative value (that should be chosen as a
"best" value in some sense), but the representation does not give any
idea of posterior uncertainty about :math:`\theta`. Nevertheless, this
simple form is widely used, particularly when it is believed that
accounting for uncertainty in :math:`\theta` is unimportant in the context
of the uncertainty expressed in the first part.

At the other extreme, we may have a very large random sample which will
provide a full and representative coverage of the range of uncertainty
about :math:`\theta` expressed in :math:`\pi_\theta(\cdot)`. In this case, the
sample may comprise :math:`s` independent draws from
:math:`\pi_\theta(\theta)`, or more usually is a Markov chain Monte Carlo
(MCMC) sample (the members of which will be correlated but should still
provide a representative coverage of the posterior distribution).

The set of :math:`\theta` values provided may be fixed (and in particular
this will be the case when :math:`s=1`), or it may be possible to increase
the size of the sample.

Additional Comments
-------------------

The way in which the sample of :math:`\strut \\theta^{(j)}` values is used
for computation of specific tasks, and in particular the way in which
the number :math:`\strut s` of :math:`\theta` values is matched to the number
required for the computation, is considered in the discussion page on
Monte Carlo estimation, sample sizes and emulator hyperparameter sets
(:ref:`DiscMonteCarlo<DiscMonteCarlo>`).
