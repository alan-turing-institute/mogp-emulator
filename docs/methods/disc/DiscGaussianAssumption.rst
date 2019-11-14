.. _DiscGaussianAssumption:

Discussion: The Gaussian assumption
===================================

Description and Background
--------------------------

The fully :ref:`Bayesian<DefBayesian>` methods in
:ref:`MUCM<DefMUCM>` are based on the use of a :ref:`Gaussian
process<DefGP>` :ref:`emulator<DefEmulator>`. In contrast to
the :ref:`Bayes linear<DefBayesLinear>` methods, this adds an
assumption of a Gaussian (i.e. normal) distribution to represent
uncertainty about the :ref:`simulator<DefSimulator>`. Whilst most of
this discussion concerns fully Bayesian emulators, some aspects are
relevant also to the Baykes linear approach.

Discussion
----------

This discussion looks at why the Gaussian assumption is made, when it
might be unrealistic in practice, and what might be done in that case.
Following these comments, which are from the perspective of the fully
Bayesian approach, some remarks are made about analogous issues in the
Bayes linear approach.

Why the Gaussian assumption
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fully Bayesian approach to statistics involves specifications of
uncertainty that are complete probability distributions. Thus, a fully
Bayesian emulator must make predictions of simulator outputs in the form
of (joint) probability distributions. In the MUCM toolkit, an emulator
is a Gaussian process (conditional on any
:ref:`hyperparameters<DefHyperparameter>` that might be in the mean
and variance functions). Using a Gaussian process implies an assumption
that uncertainty about simulator outputs may be represented
(conditionally) by normal (also known as Gaussian) distributions.

In principle, the fully Bayesian approach could involve emulators based
on any other kind of joint probability distributions, but the Gaussian
process is by far the simplest form of probability distribution that one
could use. It might be technically feasible to develop tools based on
other distributional assumptions, but those tools would certainly end up
being much more complex. So the MUCM toolkit does not attempt to address
any other kind of specification.

When the assumption might not be realistic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, it should be said that the assumption becomes unimportant if we
are able to make enough runs of the simulator in order to produce a very
accurate emulator. The point is that if there is very little uncertainty
in the emulator predictions, then the precise nature of the probability
distribution that we use to represent that uncertainty does not matter.
So it is not necessary to question the Gaussian assumption in these
circumstances.

In practice, however, the MUCM approach has been developed to work with
computationally intensive simulators, for which we cannot make
arbitrarily large numbers of runs. So we will often be in the situation
where the Gaussian assumption is not irrelevant.

Probably the most important context where the assumption may be
unrealistic is when the simulator output being emulated has a restricted
range of possible values. For instance, many simulator outputs are
necessarily positive. Suppose that the emulator estimates such an output
(for some given input configuration) with a mean of 10 and a standard
deviation of 10, then the Gaussian assumption implies that the emulator
gives a nontrivial probability that the true output will be negative.
Then the emulator is clearly making unrealistic statements, and in such
a situation the Gaussian assumption would be inappropriate.

(Incidentally, we can see here why the assumption can be ignored when we
are able to achieve high emulator accuracy, because that means the
standard deviation of any prediction will always be small. Then the
existence of restrictions on the range of the output will not be a
problem.)

What to do if the assumption is unrealistic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have seen that the Gaussian assumption becomes unrealistic if the
emulator gives a nontrivial probability to the output lying outside its
possible range, and that this may arise if the emulator has sufficient
uncertainty. The simplest remedy for this is one that is widely used to
deal with assumptions of normality in other areas of Statistics, namely
to transform the variable.

Suppose, for example, that our simulator output is the amount of water
flowing in a stream. Then it is clear that this output must be positive.
If we were simulating the flow in a river, then we might find that the
output over all plausible configurations of the simulator inputs does
not get close to zero, and then the Gaussian assumption would not give
cause for concern. In the case of a stream, however, there may be some
input configurations in which the flow is large and others (e.g. drought
conditions) that give very low flow. Then we might indeed meet the
problem of unrealistic predictions. A solution is to define the output
not to be the flow itself but the logarithm of the flow. The log-flow
output is not constrained, and so the Gaussian assumption should be
valid for the log-flow when it is not acceptable for the flow.

The log transformation is widely used in statistics to make a normality
assumption acceptable for a variable that must be positive and which has
sufficiently large uncertainty.

Notice, however, that now the emulator makes predictions about log-flow,
whereas we will generally wish to predict the actual output, i.e. flow.
If the emulator prediction for log-flow (for some given input
configuration) has a normal distribution with mean :math:`m` and variance
:math:`v`, for instance, then flow has mean :math:`\exp(m+v/2)`. In general,
if we transform the output in order to make the Gaussian assumption
acceptable, then the question arises of how to construct appropriate
inferences about the untransformed variable. In the fully Bayesian
approach this is a technical question that can readily be addressed
(although it may involve substantial extra computations).

Details of how to deal with transformed outputs can be found in the the
procedure page for transforming outputs
(:ref:`ProcOutputTransformation<ProcOutputTransformation>`).

Bayes linear methods
~~~~~~~~~~~~~~~~~~~~

There is a Bayes linear formulation in which simulator outputs are
characterised in terms of their means, variances and covariances,
modelling these in similar ways to a full Bayesian analysis. The basic
Bayes linear updating formulae will then produce analogous results to
the full Bayesian posterior means, variances and covariances.
Probability distributions are not required in the Bayes linear approach,
so the specification of means, variances and covariances will not be
extended by making the Gaussian assumption. From the Bayes linear
viewpoint, this analysis is meaningful without that assumption. From the
perspective of the full Bayesian approach, however, the Gaussian
assumption is necessary, and results cannot be legitimately obtained
without asserting probability distributions.

Within the Bayes linear approach we might also prefer to build an
emulator for the logarithm or some other transformation of the output of
interest, rather than the output itself. Then the question of making
statements about the untransformed output based on an emulator of the
transformed output is more complex because the mean and variance of the
original output are not implied by the mean and variance of the
transformed output. We therefore have to build a joint belief
specification for both the output and the transformed output, which
poses technical difficulties.

Additional Comments
-------------------

This discussion is relevant to all MUCM methods, since they all involve
emulation. See in particular the core threads
:ref:`ThreadCoreGP<ThreadCoreGP>` and
:ref:`ThreadCoreBL<ThreadCoreBL>`.
