.. _DiscExpertAssessMD:

Discussion: Expert Assessment of Model Discrepancy
==================================================

Description and Background
--------------------------

When linking a model to reality, an essential ingredient is the :ref:`model
discrepancy<DefModelDiscrepancy>` term :math:`d`. The
:ref:`assessment<DefAssessment>` of this term is rarely
straightforward, but a variety of methods are available. This page
describes the use of Expert Assessment. Here we use the term model
synonymously with the term :ref:`simulator<DefSimulator>`.

Discussion
----------

As is introduced in the variant thread on linking models to reality
using model discrepancy
(:ref:`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`),
the difference between the model :math:`f(x)` and the real system
:math:`y` is given by the model discrepancy term :math:`d`, a
precise definition of which is discussed in page
:ref:`DiscBestInput<DiscBestInput>`. :math:`d` represents the
deficiencies of the model: possible types are described in the model
discrepancy discussion page
(:ref:`DiscWhyModelDiscrepancy<DiscWhyModelDiscrepancy>`).

We use the model discrepancy :math:`d` to represent a difference
about which we may have very little information. For example, if there
are only a small number of observations with which to compare the model,
then we may not have enough data to accurately assess :math:`d`.
The informal and formal methods for assessing :math:`d`, discussed
in pages :ref:`DiscInformalAssessMD<DiscInformalAssessMD>` and
:ref:`DiscFormalAssessMD<DiscFormalAssessMD>`, may therefore not be
appropriate. Of particular importance in this case is the method of
Expert Assessment, whereby the subjective beliefs of the expert
regarding the model discrepancy are represented by statements of
uncertainty. These statements would usually be expressed in the form of
statistical properties of :math:`d`, or more likely, statistical
properties of the distribution that :math:`d` may be considered a
realisation of.

While scientists are very familiar with the notion of uncertainty, they
are generally unaccustomed to representing it probabilistically. This
subjective approach (which underlies Subjective Bayesian Analysis) is
extremely useful, especially in the case of model discrepancy when often
very little data is available. The expert (the scientist) usually has
reasonably detailed knowledge as to the deficiencies of the model, which
the model discrepancy :math:`d` is supposed to represent. All that
is required is to convert this knowledge into statistical statements
about :math:`d`.

Assessment Procedure
~~~~~~~~~~~~~~~~~~~~

A typical assessment might proceed along the following lines. Usually,
one of the most important stages in the assessment of model discrepancy
is to consider the defects in the model of which the scientist is
already aware. As a first step, the scientist would be asked to list the
main areas in which the model could be improved. If possible,
improvements that are orthogonal/independent should be identified, that
is improvements that could reasonably be considered independent of each
other, perhaps due to them dealing with different physical parts of the
model. Possible examples of improvements to the model might include:
improved solution techniques, better treatment of aspects of the physics
which are currently modelled, impact of aspects of physics currently
ignored and inaccuracies in forcing functions. The more that each of
these features can be broken down into manageable sub-units the better.
This list of improvements can be based partly on upgrade plans for the
model itself and partly on the scientific literature.

The scientist would then be asked to consider, for each such
improvement, how much difference it would be likely to make to the
computer output for a general input. It should be noted that the
improved model sits between the current model and reality, as by
definition, it must be considered to be closer to reality than the
current model. Such considerations of improved models may lead to a
fuller description of model discrepancy (which occurs in the Reification
approach described in pages :ref:`DiscReification<DiscReification>`
and :ref:`DiscReificationTheory<DiscReificationTheory>`), but for the
purposes of this section they can be used to specify order of magnitude
effects (for example standard deviations) for each of the contributions
to the components of the model discrepancy :math:`d`. If the
scientist is unwilling to specify exact numbers for some of the required
statistical quantities, an imprecise approach can be taken whereby only
ranges are required for each quantity.

This general approach is useful for a single output, but, perhaps even
more helpful for the multivariate case where we require a joint
specification over many outputs. Judgements as to how each improvement
to the model would affect subsets of the outputs would help determine
the correlation structure over the outputs.

An important point to note is that the procedure of specifying model
discrepancy is a serious scientific activity which requires careful
documentation and a similar level of care to that for each other aspect
of the modelling process.

Univariate Output
~~~~~~~~~~~~~~~~~

If we are dealing with a univariate output of the model (which is the
case for the core model: see :ref:`ThreadCoreGP<ThreadCoreGP>` and
:ref:`ThreadCoreBL<ThreadCoreBL>`), then we would follow the above
procedure and would be required to assess each of the univariate
contributions to the random quantity :math:`d=y-f(x^+)`. In the
fully probabilistic case, this involves
:ref:`eliciting<DefElicitation>` the full distribution for each
contribution. To achieve this, an appropriate choice of one of the
standard distributions might be made (e.g the Normal Distribution), and
the scientist would then be asked a series of questions designed to
identify the parameters of the distribution (in this case the mean
:math:`\mu` and standard deviation :math:`\sigma`). Often
these questions will concern certain quantiles, although there are many
possible approaches.

In the Bayes Linear case only the two numbers, which give the
expectation and variance of each contribution, are required. Often, the
expectations are assumed equal to zero, which states that the modeller
has symmetric beliefs about the deficiencies of the model. Obtaining an
assessment of the variance of each contribution requires more thought,
but various methods can be used including: assessing the standard
deviation directly, or assuming approximate distributions combined with
beliefs about quantiles.

Multivariate Output
~~~~~~~~~~~~~~~~~~~

If the model produces several outputs that can be compared to observed
data, then we can choose to consider each of the :math:`r` outputs
separately and assess the model discrepancy for each individual output
as described in the previous sections.

If the joint structure of the outputs is considered important, as is
most likely the case, a more rigorous approach is to assess the full
multivariate model discrepancy. In this case :math:`y`,
:math:`f(x)`, :math:`d` and :math:`{\rm E}[d]` are all
vectors of length :math:`r` and :math:`{\rm Var}[d]` is an
:math:`r\times r` covariance matrix. Assessing either the full
multivariate distribution for :math:`d` (in the fully Bayesian
case) or :math:`{\rm E}[d]` and :math:`{\rm Var}[d]` (in the
Bayes linear case), can be a complex task. However, as outlined above,
consideration of improvements to the model can suggest a natural
correlation structure for :math:`d`, which when combined with
consideration of the structures of the physical processes described by
the model can suggest corresponding low dimension parameterisations for
the expectation :math:`{\rm E}[d]` and covariance matrix
:math:`{\rm Var}[d]`. Then only a small number of parameter values
need be assessed which can be done directly, by using the techniques
described in the previous sections or by use of purpose built
elicitation tools. For discussion and examples of possible model
discrepancy structures and parameterisations see
:ref:`DiscStructuredMD<DiscStructuredMD>` and for an example of an
elicitation tool see Vernon et al (2010) or Bower et al (2009).

Additional Comments
-------------------

Note that several of the formal methods of assessing the model
discrepancy discussed in page
:ref:`DiscFormalAssessMD<DiscFormalAssessMD>` involve the
specification of either prior distributions for :math:`d`, or
expectations and variances of :math:`d`. These prior specifications
would most likely be given using the methods outlined in this page.

References
----------

Vernon, I., Goldstein, M., and Bower, R. (2010), “Galaxy Formation: a
Bayesian Uncertainty Analysis,” MUCM Technical Report 10/03

Bower, R., Vernon, I., Goldstein, M., et al. (2009), “The Parameter
Space of Galaxy Formation,” MUCM Technical Report 10/02,
