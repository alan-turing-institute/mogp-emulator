.. _AltGPorBLEmulator:

Alternatives: Gaussian Process or Bayes Linear Emulators
========================================================

Overview
--------

The :ref:`MUCM<DefMUCM>` technology of quantifying and managing
uncertainty in complex simulation models rests on the idea of
:ref:`emulation<DefEmulator>`. This toolkit deals with two basic ways
to construct and use emulators - :ref:`Gaussian process<DefGP>`
emulators and :ref:`Bayes linear<DefBayesLinear>` emulators. The two
forms have much in common, but there are also fundamental differences.
These differences have impact on the ways in which the two approaches
are used and the kinds of applications to which they are applied.

Choosing the Alternatives
-------------------------

-  To choose the route of Gaussian process emulation, the core thread is
   :ref:`ThreadCoreGP<ThreadCoreGP>`.
-  To choose the route of Bayes linear emulation, the core thread is
   :ref:`ThreadCoreBL<ThreadCoreBL>`.
-  The core threads deal with emulation of a basic kind of problem
   called the :ref:`core problem<DiscCore>`. For more complex
   situations, the relevant variant thread deals with how to modify the
   core to tackle new features.

The Nature of the Alternatives
------------------------------

An emulator is a statistical representation of a
:ref:`simulator<DefSimulator>`. For any given values of the
simulator's inputs, we can obtain the simulator output(s) by running the
simulator itself, but we can instead use an emulator to predict what the
output(s) would be. MUCM methods use the emulator to address questions
about the simulator far more efficiently than methods which rely on
running the simulator itself.

Perhaps the most fundamental difference between GP and BL emulators is
the nature of the predictions.

-  A GP emulator provides a full probability distribution for the
   output(s) as its prediction. In particular, the mean of the
   distribution is the natural estimate, and the variance (or its square
   root, the standard deviation) provides a measure of accuracy. Because
   it provides a full predictive distribution, a GP emulator can also
   provide credible intervals for the outputs.
-  A BL emulator provides an estimate and a dispersion measure that are
   analogous to the mean and variance of a GP emulator, although in
   principle they have somewhat different interpretations. BL emulator
   predictions, however, are not full probability distributions, and are
   confined to the estimate and dispersion measure.

These differences result from a difference in underlying philosophy. GP
emulators are based in conventional :ref:`Bayesian<DefBayesian>`
statistical theory, in which data are represented by their sampling
distributions and unknown parameters by prior or posterior
distributions. Bayes linear theory is based on considering a set of
uncertain quantities and defining a belief specification for them which
comprises an estimate and a measure of dispersion for each, together
with an association measure for each pair. From the perspective of
conventional Bayesian statistics, it is natural to interpret these as
the means, variances and covariances of the uncertain quantities, and in
some ways this interpretation conforms with how these measures are
elicited and used in Bayes linear methods. However, the Bayes linear
view is philosophically different, and in its most abstract form
interprets them geometrically - the estimates define a point in an
abstract metric space with a metric defined by the dispersion and
association measures. The Bayes linear analogue of the use of Bayes'
theorem in Bayesian theory, to update beliefs in the light of additional
information, is projection in the metric space.

The two theories coincide if in the Bayes linear analysis the set of
uncertain quantities include the indicator functions of every possible
value for all the random variables that are given probability
distributions in the fully Bayesian approach. However, this equivalence
is achieved by the Bayes linear analyst effectively defining full
probability distributions, and in practice Bayes linear analyses will
only exceptionally include such a full probabilistic specification.
Adherents of the Bayes linear view argue, correctly, that one can never
make so many judgements about a problem if those judgements are to
represent carefully considered beliefs. In a fully Bayesian analysis,
distributions are not specified by thinking about every single
probability that makes up that distribution. In reality, they are
specified by a convenient (and more loosely considered) completion of a
much smaller number of carefully considered judgements. Proponents of
the Bayes linear approach decline in principle to make judgements that
are not individually considered and elicited.

The choice between the two approaches can then be reduced to choosing
between two alternative ways of making statistical inferences in the
context of a relatively small, finite number of actual, carefully
considered judgements.

#. In the full Bayesian approach, those judgements are expanded to
   produce full probability distributions, by a choice of distributional
   form that is partly based on informal judgement but also partly on
   convenience. Then updating via Bayes' theorem produces full posterior
   distributions for inference.
#. In the Bayes linear approach, a complete Bayes linear belief
   specification is required for the chosen set of uncertain quantities,
   but no other judgements are added (and in particular distributions
   are not required). Updating is via projection and yields an adjusted
   belief specification.

The two approaches differ in their interpretation of the meaning of the
belief analysis, but each approach may be regarded, from the other
viewpoint, as an approximation to what would be produced using their
preferred approach.

It is not the purpose of this page to go deeply into the philosophy.
Although Bayes linear methods have only limited support in the Bayesian
community generally, they are accepted in the field of computer model
uncertainty, and have made important contributions to that field.

The choice between a GP and a BL emulator is principally a matter of
philosophical viewpoint but also partly pragmatic. The computations
required for BL methods are based on manipulating variance matrices, and
can remain tractable even in very complex applications, where sometimes
the computations using the GP emulator become infeasible. On the other
hand, it may be difficult to formulate the complete belief specification
in a complex BL application, and not having full posterior distributions
makes it more difficult to deliver some kinds of inferences in the BL
approach.
