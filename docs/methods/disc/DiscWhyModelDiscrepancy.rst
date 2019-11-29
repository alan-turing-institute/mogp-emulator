.. _DiscWhyModelDiscrepancy:

Discussion: Why Model Discrepancy?
==================================

Description and Background
--------------------------

Often, the purpose of a model, :math:`f(x)`, of a physical process,
is to make statements about the corresponding real world process
:math:`y` (see the variant thread on linking models to reality
using model discrepancy
:ref:`ThreadVariantModelDiscrepancy<ThreadVariantModelDiscrepancy>`).
In order for this to be possible, the difference between the model and
the real system must be :ref:`assessed<DefAssessment>` and
incorporated into the analysis. This difference is known as the :ref:`model
discrepancy<DefModelDiscrepancy>`, :math:`d`, and this page
discusses the reasons why such a quantity is essential. Here we use the
term model synonymously with the term :ref:`simulator<DefSimulator>`.

Discussion
----------

A model is an imperfect representation of the way system properties
affect system behaviour. Whenever models are used, this imperfection
should be accounted for within our analysis, by careful assessment of
Model Discrepancy.

Model discrepancy is a feature of computer model analysis which is
relatively unfamiliar to most scientists. However carefully we have
constructed our model, there will always be a difference between the
system and the simulator. A climate model will never be precisely the
same as climate, and nor would we expect it to be. Inevitably, there
will be simplifications in the physics, based on features that are too
complicated for us to include, features that we do not know that we
should include, mismatches between the scales on which the model and the
system operate, simplifications and approximations in solving the
equations determining the system, and uncertainty in the forcing
functions, boundary conditions and initial conditions.

Incorporating the difference between the model and the real system
within our analysis will allow us

-  to make meaningful predictions about future behaviour of the system,
   rather than the model,
-  to perform a proper :ref:`calibration<DefCalibration>` of the
   model inputs to historical observation.

Obviously we will be uncertain about this difference, :math:`d`,
and it is therefore natural to express such judgements
probabilistically, to be incorporated within a Bayesian analysis,
employing either the fully probabilistic or the Bayes Linear approach.

Neglecting the model discrepancy would make all further statements in
the analysis conditional on the model being a perfect representation of
reality, and would result in overconfident predictions of system
behaviour. In fact, ignoring :math:`d` leads to a smaller variance
of (and hence overconfidence in) the observed data :math:`z`, which
leads to over-fitting in calibration (when we learn about
:math:`x`). This exacerbates the over-confidence effect in
subsequent predictions. Similarly, ignoring the correlation structure
within :math:`d` leads to further prediction inaccuracies.
Fundamental to the scientific process is the concern that without
:math:`d`, one can wrongly rule out a potentially useful model.
Unfortunately, all these mistakes are commonly made.

The simplest and most common strategy for incorporating model
discrepancy is known as the :ref:`Best Input<DefBestInput>` Approach,
which is discussed in detail in page
:ref:`DiscBestInput<DiscBestInput>`, and gives a simple and intuitive
definition of :math:`d`. More careful methods have been developed
that go beyond the limitations of the Best Input approach, one such
approach involves the idea of Reification (see the discussion pages on
reification (:ref:`DiscReification<DiscReification>`) and its theory
(:ref:`DiscReificationTheory<DiscReificationTheory>`)).

Often, understanding the size and structure of the model discrepancy
will be one of the most challenging aspects of the analysis. There are,
however, a variety of methods available to assess :math:`d`: for
Expert, Informal and Formal methods see the discussion pages
:ref:`DiscExpertAssessMD<DiscExpertAssessMD>`,
:ref:`DiscInformalAssessMD<DiscInformalAssessMD>` and
:ref:`DiscFormalAssessMD<DiscFormalAssessMD>` respectively.

Additional Comments
-------------------

The assessment of :math:`d` can be a difficult and subtle task. We
generally use the random quantity :math:`d` to statistically model
a difference that is, in reality, extremely complex. In this sense,
there is no 'true' value of :math:`d` itself, instead it should be
viewed as a useful tool for representing an important feature (the
difference between model and reality) in a simple and tractable manner.
Before an assessment is made, we should be clear about which features of
the model discrepancy we are interested in. For example, do we want to
learn about the realised values of :math:`d` itself, or do we wish
to learn about the parameters of a distribution that :math:`d` may
be considered a realisation of. Such considerations are of particular
importance in problems where the match between the model output and
historical data is used to inform judgements about the likely values of
model discrepancy for future system features.
