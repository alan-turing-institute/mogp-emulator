.. _DefBayesLinear:

Definition of Term: Bayes linear
================================

Like the fully :ref:`Bayesian<DefBayesian>` approach in
:ref:`MUCM<DefMUCM>`, the Bayes linear approach is founded on a
personal (or subjective) interpretation of probability. Unlike the
Bayesian approach, however, probability is not the fundamental primitive
concept in the Bayes linear philosophy. Instead, expectations are
primitive and updating is carried out by orthogonal projection. In the
simple case of a vector of real quantities, some of which are observed,
this projection is equivalent to minimising expected quadratic loss over
a linear combination of the observed quantities, and the adjusted mean
and variance that result are computationally the same as the Bayesian
conditional mean and variance of a Gaussian vector. If the observation
vector consists of the indicator functions for a partition, then the
Bayes linear adjustment is equivalent to Bayesian conditioning on the
partition.

Further details on the theory behind the Bayes linear approach are given
in the discussion page on theoretical aspects of Bayes linear
(:ref:`DiscBayesLinearTheory<DiscBayesLinearTheory>`).
