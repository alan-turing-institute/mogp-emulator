.. _DefWeakPrior:

Definition of Term: Weak prior distribution
===========================================

In :ref:`Bayesian<DefBayesian>` statistics, prior knowledge about a
parameter is specified in the form of a probability distribution called
its prior distribution (or simply its prior). The use of prior
information is a feature of Bayesian statistics, and one which is
contentious in the field of statistical inference. Most opponents of the
Bayesian approach disagree with the use of prior information. In this
context, there has been considerable study of the notion of a prior
distribution that represents prior ignorance, or at least a state of
very weak prior information, since by using such a prior it may be
possible to evade this particular criticism of the Bayesian approach. In
this toolkit, we will call such prior distributions "weak priors,"
although they may be found answering to many other names in the
literature (such as reference priors, noninformative priors, default
priors or objective priors).

The use of weak priors has itself been criticised on various grounds.
Strict adherents of the Bayesian view argue that genuine prior
information invariably exists (i.e. a state of prior ignorance is
unrealistic) and that to deny prior information is wasteful. On more
pragmatic grounds, it is clear that despite all the research into weak
priors there is nothing like a consensus on what is *the* weak prior to
use in any situation, and there are numerous competing theories leading
to alternative weak prior formulations. On theoretical grounds, others
point to logical inconsistencies in any systematic use of weak priors.

In the :ref:`MUCM<DefMUCM>` toolkit, we adopt a Bayesian approach (or
a variant known as the :ref:`Bayes Linear<DefBayesLinear>` approach)
and so take the view that prior information should be used. Nevertheless
we see a pragmatic value in weak priors. When prior information is
genuinely weak relative to the information that may be gained from the
data in a statistical analysis, it can be shown that the precise choice
of prior distribution has little effect on the statistical results
(inferences). Then the use of a weak prior can be justified as being an
adequate replacement for spending unnecessary effort on specifying the
prior. (And in this context, the existence of alternative weak prior
formulations is not a problem because it should not matter which we
use.)

Note that conventional weak priors are often
:ref:`improper<DefProper>`.
