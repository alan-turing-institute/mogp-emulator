.. _ProcPredictGP:

Procedure: Predict simulator outputs using a GP emulator
========================================================

Description and Background
--------------------------

A :ref:`Gaussian process<DefGP>` (GP) :ref:`emulator<DefEmulator>`
is a statistical representation of knowledge about the outputs of a
:ref:`simulator<DefSimulator>` based on the Gaussian process as a
probability distribution for an unknown function. The unknown function
in this case is the simulator, viewed as a function that takes inputs
and produces one or more outputs. One use for the emulator is to predict
what the simulator would produce as output when run at one or several
different points in the input space. This procedure describes how to
derive such predictions in the case of a GP emulator built with the
procedure for a single output
(:ref:`ProcBuildCoreGP<ProcBuildCoreGP>`) or the procedure for
multiple outputs
(:ref:`ProcBuildMultiOutputGP<ProcBuildMultiOutputGP>`) . The
multiple output case will be emphasised only where this differs from the
single output case.

Inputs
------

-  An emulator
-  A single point :math:`x^\prime` or a set of points :math:`x^\prime_1,
   x^\prime_2,\ldots,x^\prime_{n^\prime}` at which predictions are
   required for the simulator output(s)

Outputs
-------

-  Predictions in the form of statistical quantities, such as the
   expected value of the output(s), the variance (matrix) of the
   outputs, the probability that an output exceeds some threshold, or a
   sample of values from the predictive distribution of the output(s)

Procedure
---------

The emulator, as developed for instance in
:ref:`ProcBuildCoreGP<ProcBuildCoreGP>` or
:ref:`ProcBuildMultiOutputGP<ProcBuildMultiOutputGP>`, has two parts.
The first is a distribution, either a GP or its cousin the
:ref:`t-process<DefTProcess>`, for the simulator output function
conditional on some hyperparameters. The second is generally a
collection of sets of values of the hyperparameters being conditioned
on, but may be just a single set of values. We let :math:`s` denote
the number of hyperparameter sets provided with the emulator. When we
have a single set of hyperparameter values, :math:`s=1`. See the
discussion page on the forms of GP based emulators
(:ref:`DiscGPBasedEmulator<DiscGPBasedEmulator>`) for more details.

The procedure for computing predictions generally takes the form of
computing the appropriate predictions from the GP or t-process, given
that the hyperparameters takes each of the :math:`s` sets of values
in turn, and if :math:`s>1` then combining the resulting :math:`s`
predictions. See also the discussion page on Monte Carlo estimation
(:ref:`DiscMonteCarlo<DiscMonteCarlo>`), where this approach is
treated in more detail and in particular there is consideration of how
many sets of hyperparameters should be used.

Predictive mean
~~~~~~~~~~~~~~~

The conditional mean of the output at :math:`x^\prime`, given the
hyperparameters, is obtained by evaluating the mean function of the GP
or t-process at that point. When :math:`s=1`, this is done with the
hyperparameters fixed at the single set of values. When :math:`s>1`,
we evaluate the mean function using each of the :math:`s`
hyperparameter sets and the predictive mean is the average of those
:math:`s` conditional means.

If required for a set of points :math:`x^\prime_1,
x^\prime_2,\ldots,x^\prime_{n^\prime}`, the predictive mean of the
output vector is the vector of predictive means obtained by applying the
above procedure to each :math:`x^\prime_j` separately. In the multi-ouput
case things are somewhat more complicated. Each output is itself a :math:`1
\times r` vector, so the outputs at several input configurations can
be treated in two ways, either as as vector of vectors (that is a
:math:`{n^\prime} \times r` matrix) or more helpfully stacked into a
:math:`{n^\prime} r \times 1` vector of outputs at each input
configuration. This vector can then be treated as described above.

Predictive variance (matrix)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider the case where we wish to derive the predictive variance matrix
of the output vector at a set of points :math:`x^\prime_1,
x^\prime_2,\ldots,x^\prime_{n^\prime}`. If :math:`s=1` the
predictive variance matrix is just the matrix of conditional variances
and covariances from the GP or t process, using the single set of
hyperparameter values.

If :math:`s>1` the procedure is more complex, and requires several
steps as follows.

#. For :math:`i=1,2,\ldots,s` fix the hyperparameters at the i-th set, and
   given these values of the hyperparameters, compute the vector of
   conditional means :math:`E_i` and the matrix :math:`V_i` of conditional
   variances and covariances from the GP or t-process.
#. Compute the average values :math:`\bar E =
   s^{-1}{\scriptstyle\sum_{i=1}^s}E_i` and :math:`\bar V =
   s^{-1}{\scriptstyle\sum_{i=1}^s}V_i`. (Note that :math:`\bar E` is the
   predictive mean described above.)
#. Compute the variance matrix of the conditional means, :math:`W =
   s^{-1}{\scriptstyle\sum_{i=1}^s}(E_i-\bar E)(E_i-\bar E)^{\rm T}`.
#. The predictive variance matrix is :math:`\bar V + W`.

Prediction at a single point :math:`x^\prime` is the special case
:math:`n^\prime=1` of this procedure. In brief, the predictive variance is
either just the conditional variance evaluated with the single set of
hyperparameter values, or if :math:`s>1` the average of the
conditional variances plus the variance of the conditional means. To
handle the multi-output case the most simple approach is to pursue the
vectorisation of the outputs to a :math:`{n^\prime} r \times 1` vector. In
this case the variance matrix is :math:`{n^\prime} r \times {n^\prime} r`
(with a range of possible simplifications if the covariance is assumed
to be :ref:`separable<DefSeparable>`. This (potentially very large)
variance matrix can be treated identically to the single output case.

Probability of exceeding a threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The conditional probability of exceeding a threshold can be computed
from the GP or t-process for any given set of hyperparameter values. For
a GP, this means computing the probability of exceeding a given value
for a normal random variable with given mean and variance. For the
t-process it is the probability of exceeding that value for a t random
variable with given mean, variance and degrees of freedom. For
:math:`s>1`, the predictive probability is the average of the conditional
probabilities.

For multiple outputs this is more complex, since it is possible to ask
more complex questions, such as the joint probability of two or more
outputs exceeding some threshold. The complexity depends on the
assumptions made in constructing the multivariate emulator, and is
discussed in the alternatives page on approaches to multiple outputs
(:ref:`AltMultipleOutputsApproach<AltMultipleOutputsApproach>`). For
example if separate independent emulators are used, then the probability
of all outputs lying above some threshold will be the product of the
individual probabilities of each output being above the threshold. This
will not be true if the outputs are correlated and the full multivariate
GP or t-process should be used.

Sample of predictions
~~~~~~~~~~~~~~~~~~~~~

Suppose we wish to draw a sample of :math:`N` values from the predictive
distribution of the simulator output at the input :math:`x^\prime`, or of
the outputs at the points :math:`x^\prime_1,
x^\prime_2,\ldots,x^\prime_{n^\prime}`. This means using :math:`N` sets of
hyperparameter values. If :math:`N<s`, then we select a subset of the full
set of available hyperparameter sets. (These will usually have been
produced by Markov chain Monte Carlo sampling, in which case the subset
should be chosen by thinning the sequence of hyperparameter sets, e.g.
if :math:`N=s/2` we could take only even numbered hyperparameter sets.)

If :math:`N>s` we will need to reuse some hyperparameter sets. Although
this is generally undesirable, in the case :math:`s=1` it is
unavoidable! However, it may be feasible to obtain a larger sample of
hyperparameter sets: see :ref:`DiscMonteCarlo<DiscMonteCarlo>`.

For each chosen hyperparameter set, we make a *single* draw from the
conditional distribution of the output(s) given by the GP or t-process,
conditional on that hyperparameter set. Procedures for generating random
outputs are described in :ref:`ProcOutputSample<ProcOutputSample>`.

Additional Comments
-------------------

It is possible to develop procedures for other kinds of predictions, but
not all will be simple. For instance to output a predictive credible
interval would be a more complex procedure.
