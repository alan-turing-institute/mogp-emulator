.. _DefEmulator:

Definition of Term: Emulator
============================

An emulator is a statistical representation of a
:ref:`simulator<DefSimulator>`. For any given configuration of input
values for the simulator, the emulator provides a probabilistic
prediction of one or more of the outputs that the simulator would
produce if it were run at those inputs.

Furthermore, for any set of input configurations, the emulator will
provide a joint probabilistic prediction of the corresponding set of
simulator outputs.

*Example*: A simulator of a nuclear power station reactor requires
inputs that specify the flow of gas through the reactor, and produces an
output which is the steady state mean temperature of the reactor core. A
simulator of this output would provide probabilistic predictions of the
mean temperature (as output by the actual simulator) at any single
configuration of gas flow inputs, or at any set of such configurations.

The probabilistic predictions may take one of two forms depending on the
approach used to build the emulator. In the fully
:ref:`Bayesian<DefBayesian>` approach, the predictions are complete
probability distributions.

*Example*: In the previous example, the fully Bayesian approach would
provide a complete probability distribution for the output from any
single configuration of inputs. This would give the probability that the
output (mean temperature) lies in any required range. In particular, it
would also provide any desired summaries of the probability
distribution, such as the mean or variance. For a given set of input
configurations, it would produce a joint probability distribution for
the corresponding set of outputs, and in particular would give means,
variances and covariances.

In the :ref:`Bayes linear<DefBayesLinear>` approach, the emulator's
probabilistic specification of outputs comprises (adjusted) means,
variances and covariances.

*Example*: The Bayes linear emulator in the above example would provide
the (adjusted) mean and (adjusted) variance for the simulator output
(mean temperature) from a given single configuration of gas flow inputs.
When the emulator is used to predict a set of outputs from more than one
input configuration, it will also provide (adjusted) covariances between
each pair of outputs.

Strictly, these 'adjusted' means, variances and covariances have a
somewhat different meaning from the means, variances and covariances in
a fully Bayesian emulator. Nevertheless, they are in practice
interpreted the same way - thus, the (adjusted) mean is a point estimate
of the simulator output and the square-root of the (adjusted) variance
is a measure of accuracy for that estimate in the sense of being a
root-mean-square distance from the estimate to the true simulator
output. In practice, we would drop the word 'adjusted' and simply call
them means, variances and covariances, but the distinction can be
important.
